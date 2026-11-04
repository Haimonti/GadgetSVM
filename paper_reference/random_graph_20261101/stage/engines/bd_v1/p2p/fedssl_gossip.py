"""Author FedSSL local solver on PeerSim nodes with synchronous P2P flooding.

Every node aggregates received records locally. A fixed n-1-hop budget per
training epoch preserves the synchronous update without a model server.
"""
from paper_reference import ssl as iq_backend
from paper_reference.ssl import SSLConfig,aggregate_states
from src.peersim_python.cdsim import CDProtocol,CDState


class FedSSLGossipProtocol(CDProtocol):
    LINKABLE_PID=0
    backend=iq_backend

    def __init__(self, config=None):
        self.config=config or SSLConfig()
        self.config.validate()
        self.inbox=[]
        self.comm_bytes=0
        self.messages=0
        self.metrics=[]
        self.stopped=False
        self.data_ready=False
        self.node_id=None
        self.epoch=-1
        self.head=None
        self.test_evaluations=0

    def clone(self):
        from dataclasses import replace
        return type(self)(replace(self.config))

    def set_iq_data(self,unlabeled,labeled,test,node_id,n_workers):
        self.X=self.backend.input_tensor(unlabeled) if hasattr(self.backend,"input_tensor") else self.backend.iq_tensor(unlabeled)
        self.labeled=labeled
        self.test=test  # not read by the training loop
        self.node_id=int(node_id)
        self.n_workers=int(n_workers)
        self.n=len(self.X)
        self.model=self.backend.new_encoder(self.config)
        self.data_ready=True
        self.period=self.n_workers
        self.completed_epochs=0

    @staticmethod
    def record_bytes(record):
        # int64 origin/epoch/count headers plus float64 local loss.
        return 32+sum(v.numel()*v.element_size() for v in record[0].values())

    def nextCycle(self,node,pid):
        if not self.data_ready or self.stopped:return
        cycle=CDState.getCycle()
        epoch,offset=divmod(cycle,self.period)
        epoch+=getattr(self,"start_epoch",0)
        if epoch>=self.config.rounds:return
        if offset==0:
            self.epoch=epoch
            self.records={}
            self.sent={}
            state,loss=self.backend.train_local(self.model.state_dict(),self.X,self.config,self.node_id,epoch)
            self.records[self.node_id]=(state,self.n,loss)
        for message in self.inbox:
            if message['epoch']!=epoch:continue
            for origin,record in message['records'].items():
                if not 0<=origin<self.n_workers:raise ValueError('Invalid SSL contribution origin')
                self.records.setdefault(origin,record)
        self.inbox=[]
        if offset==self.period-1:
            if len(self.records)!=self.n_workers:
                raise RuntimeError('SSL P2P epoch incomplete: graph must be strongly connected and reliable')
            aggregate,loss=aggregate_states([self.records[i] for i in range(self.n_workers)],self.config)
            self.model.load_state_dict(aggregate)
            self.completed_epochs+=1
            self.metrics.append(dict(training_round=epoch+1,link_cycle=cycle+1,
                train_triplet_loss=loss,comm_bytes=self.comm_bytes))
            if self.completed_epochs==self.config.rounds and self.backend is iq_backend:
                self.finish('author_fixed_round_budget')
            return
        links=node.getProtocol(self.LINKABLE_PID)
        for j in range(links.degree()):
            peer_node=links.getNeighbor(j)
            peer_id=peer_node.getIndex()
            sent=self.sent.setdefault(peer_id,set())
            delta={origin:r for origin,r in self.records.items() if origin not in sent}
            if not delta:continue
            # Detached origin records are immutable; load_state_dict copies.
            peer_node.getProtocol(pid).inbox.append({'epoch':epoch,'records':dict(delta)})
            self.comm_bytes+=sum(self.record_bytes(r) for r in delta.values())
            self.messages+=1
            sent.update(delta)

    def finish(self,reason):
        self.model.eval()
        for parameter in self.model.parameters():parameter.requires_grad_(False)
        self.head=self.backend.fit_local_heads(self.model,[self.labeled],self.config)[0]
        self.stopped=True
        self.stop_reason=reason

    def accuracy(self):
        if not self.stopped or self.head is None:
            raise RuntimeError('Final test requested before SSL training and head fitting completed')
        if self.test_evaluations:raise RuntimeError('SSL test set has already been evaluated')
        value=self.backend.evaluate_local_heads(self.model,[self.head],[self.test])[0]
        self.test_evaluations+=1
        return value

    def metadata(self):
        result=self.backend.source_metadata(self.config,self.model)
        result.update(training_rounds=self.completed_epochs,link_cycles=self.completed_epochs*self.period,
            communication_bytes=self.comm_bytes,messages=self.messages,
            transport='PeerSim synchronous epoch delta flooding',
            stop_reason=getattr(self,'stop_reason','max_link_cycles'),converged=None)
        if self.head is not None:
            result.update(selected_svm_C=float(self.head.C),local_svm_iterations=int(self.head.n_iter_))
        return result
