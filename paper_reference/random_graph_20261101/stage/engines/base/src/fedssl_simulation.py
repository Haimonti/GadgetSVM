"""I/Q FedSSL-AMC on the same PeerSim runtime as the SVM comparisons."""
from p2p.fedssl_gossip import FedSSLGossipProtocol
from src.peersim_python.core import Network,GeneralNode,CommonState,Control
from src.peersim_python.idle_protocol import IdleProtocol
from src.peersim_python.cdsim import CDSimulator
from src.peersim_python.dynamics import WireRing,WireFull,WireKOut


class IQInitializer(Control):
    def __init__(self,u,l,t):self.u,self.l,self.t=u,l,t

    def execute(self):
        for i in range(len(self.u)):
            Network.get(i).getProtocol(1).set_iq_data(self.u[i],self.l[i],self.t[i],i,len(self.u))
        return False


class SSLFinished(Control):
    def execute(self):
        return all(Network.get(i).getProtocol(1).stopped for i in range(Network.size()))


class FedSSLSimulation:
    protocol_type=FedSSLGossipProtocol
    def __init__(self,unlabeled,labeled,test,config,topology='ring',activation='shuffle'):
        if not unlabeled or not len(unlabeled)==len(labeled)==len(test):
            raise ValueError('Aligned nonempty I/Q client collections required')
        if activation not in ('ordered','shuffle'):
            raise ValueError('SSL synchronous epochs require one activation per node per cycle')
        self.u,self.l,self.t=unlabeled,labeled,test
        self.config=config
        self.topology=topology
        self.activation=activation
        self.sim=None

    def build(self):
        CommonState.initializeRandom(self.config.seed)
        Network.reset(len(self.u),GeneralNode([IdleProtocol(),self.protocol_type(self.config)]))
        if self.topology=='ring':wire=WireRing(0,undir=True)
        elif self.topology=='full':wire=WireFull(0,undir=True)
        elif self.topology=='random_kout':wire=WireKOut(0,min(3,len(self.u)-1),undir=True)
        else:raise ValueError('SSL topology must be ring/full/random_kout')
        self.sim=CDSimulator(self.config.rounds*len(self.u),
            [wire,IQInitializer(self.u,self.l,self.t)],[SSLFinished()],self.activation)
        return self

    def run(self):
        if hasattr(self,'protos'):raise RuntimeError('This SSL experiment has already run; create a fresh simulation')
        if self.sim is None:self.build()
        cycles=self.sim.nextExperiment()
        self.protos=[Network.get(i).getProtocol(1) for i in range(Network.size())]
        if not all(p.stopped for p in self.protos):
            raise RuntimeError('SSL PeerSim budget exhausted before all clients finished')
        return dict(link_cycles=cycles,training_rounds=self.config.rounds,
            total_comm_bytes=sum(p.comm_bytes for p in self.protos),
            per_node_comm_bytes=[p.comm_bytes for p in self.protos],
            history=self.protos[0].metrics,per_node_metadata=[p.metadata() for p in self.protos])

    def evaluate_final(self):
        if not hasattr(self,'protos'):raise RuntimeError('Run training before final test evaluation')
        return [p.accuracy() for p in self.protos]
