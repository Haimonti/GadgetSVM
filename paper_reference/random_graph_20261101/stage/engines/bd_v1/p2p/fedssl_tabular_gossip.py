"""Actual one-hop gossip SSL, sharing the same transport as FedAvg/SDCA."""
from dataclasses import replace
from paper_reference import ssl_tabular as backend
from src.peersim_python.gossip_protocol import GossipProtocol
from src.peersim_python.cdsim import CDState
from src.peersim_python.aggregator import Aggregator


class EncoderAverageAggregator(Aggregator):
    def __init__(self,config):self.config=config

    def aggregate(self,current,peers):
        records=[(p['state'],p['n'],0.) for p in [current,*peers]]
        state,_=backend.aggregate_states(records,self.config)
        return {'state':state,'n':current['n']}


class TabularSSLGossipProtocol(GossipProtocol):
    def __init__(self,config,gossip_k=3):
        super().__init__(gossip_k,EncoderAverageAggregator(config))
        self.config=config;self.data_ready=False;self.head=None;self.test_evaluations=0
        self.metrics=[];self.completed_rounds=0

    def clone(self):return type(self)(replace(self.config),self.gossip_k)

    def set_data(self,x,labeled,test,node_id):
        self.X=backend.input_tensor(x);self.n=len(self.X);self.labeled=labeled;self.test=test
        self.node_id=node_id;self.model=backend.new_encoder(self.config);self.data_ready=True

    def ready(self):return self.data_ready

    def current_state(self):return {'state':self.model.state_dict(),'n':self.n}

    def outgoing_payload(self):
        # Own latest encoder only; no all-origin tables and no relay flooding.
        return {'state':{k:v.detach().clone() for k,v in self.model.state_dict().items()},'n':self.n}

    def set_state(self,merged):self.model.load_state_dict(merged['state'])

    def local_update(self):
        state,self.local_loss=backend.train_local(self.model.state_dict(),self.X,self.config,self.node_id,CDState.getCycle())
        self.model.load_state_dict(state);self.completed_rounds+=1

    def finish(self,reason):
        self.model.eval()
        for p in self.model.parameters():p.requires_grad_(False)
        self.head=backend.fit_local_heads(self.model,[self.labeled],self.config)[0]
        self.stopped=True;self.stop_reason=reason

    def accuracy(self):
        if self.head is None or not self.stopped:raise RuntimeError('Final test requested before stop')
        if self.test_evaluations:raise RuntimeError('Test already evaluated')
        value=backend.evaluate_local_heads(self.model,[self.head],[self.test])[0]
        self.test_evaluations+=1;return value
