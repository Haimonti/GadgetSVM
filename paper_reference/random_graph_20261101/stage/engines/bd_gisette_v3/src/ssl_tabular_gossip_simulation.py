"""Controlled gossip tabular SSL on the common PeerSim engine."""
from p2p.fedssl_tabular_gossip import TabularSSLGossipProtocol
from paper_reference import ssl_tabular as backend
from src.peersim_python.core import Network,GeneralNode,CommonState,Control
from src.peersim_python.idle_protocol import IdleProtocol
from src.peersim_python.cdsim import CDSimulator,CDState
from src.fedavg_simulation import FedAvgSimulation
from src.train_convergence import WindowStability
from src.peersim_python.communication import POLICY


class Initialize(Control):
    def __init__(self,u,l,t):self.u,self.l,self.t=u,l,t
    def execute(self):
        for i in range(len(self.u)):Network.get(i).getProtocol(1).set_data(self.u[i],self.l[i],self.t[i],i)
        return False


class Observe(Control):
    def __init__(self,config,n):
        self.config=config;self.rule=WindowStability(n,config.stability_window,config.stability_checks,config.rtol)
        self.history=[];self.converged=False
    def execute(self):
        ps=[Network.get(i).getProtocol(1) for i in range(Network.size())]
        r=CDState.getCycle()+1
        losses=[backend.monitor_loss(p.model,p.X,self.config,i) for i,p in enumerate(ps)]
        self.converged=self.rule.update(losses,r)
        self.history.append(dict(round=r,training_round=r,link_cycle=r,train_contrastive_loss=sum(losses)/len(losses),
            per_node_monitor_loss=losses,total_comm_bytes=sum(p.comm_bytes for p in ps)))
        if r%50==0 or self.converged or r==self.config.rounds:
            print(f'gossip SSL round={r} loss={sum(losses)/len(losses):.7g} all_peer_plateau={self.converged}',flush=True)
        if self.converged or r==self.config.rounds:
            reason='train_contrastive_plateau' if self.converged else 'max_training_rounds'
            for p in ps:p.finish(reason)
            return True
        return False


class GossipTabularSSLSimulation:
    def __init__(self,u,l,t,config,topology='random_kout',activation='shuffle',gossip_k=3):
        if not u or not len(u)==len(l)==len(t):raise ValueError('Aligned nonempty client collections required')
        if activation not in ['ordered','shuffle']:raise ValueError('One activation per node per cycle required')
        self.u,self.l,self.t,self.config=u,l,t,config
        self.topology,self.activation,self.gossip_k=topology,activation,gossip_k
        self.sim=None
    def build(self):
        CommonState.initializeRandom(self.config.seed)
        Network.reset(len(self.u),GeneralNode([IdleProtocol(),TabularSSLGossipProtocol(self.config,self.gossip_k)]))
        wire=FedAvgSimulation({'GOSSIP_K':self.gossip_k},[])._topology(self.topology)
        self.observer=Observe(self.config,len(self.u))
        self.sim=CDSimulator(self.config.rounds,[wire,Initialize(self.u,self.l,self.t)],[self.observer],self.activation)
        return self
    def run(self):
        if hasattr(self,'protos'):raise RuntimeError('Create a fresh experiment')
        if self.sim is None:self.build()
        cycles=self.sim.nextExperiment()
        self.protos=[Network.get(i).getProtocol(1) for i in range(Network.size())]
        self.adjacency=[sorted(Network.get(i).getProtocol(0).getNeighbor(j).getIndex() for j in range(Network.get(i).getProtocol(0).degree())) for i in range(Network.size())]
        metadata=[dict(backend.source_metadata(self.config,p.model),transport='shared one-hop GossipProtocol',
            selected_svm_C=float(p.head.C),local_svm_iterations=int(p.head.n_iter_)) for p in self.protos]
        return dict(link_cycles=cycles,training_rounds=cycles,rounds=cycles,converged=self.observer.converged,
            stop_reason=self.protos[0].stop_reason,total_comm_bytes=sum(p.comm_bytes for p in self.protos),
            per_node_comm_bytes=[p.comm_bytes for p in self.protos],history=self.observer.history,
            per_node_metadata=metadata,topology=self.topology,activation=self.activation,gossip_k=self.gossip_k,
            adjacency=self.adjacency,communication_policy=POLICY,
            messages_sent=sum(p.messages_sent for p in self.protos),
            transport='shared one-hop gossip; current encoder only; no flooding or exact global averaging')
    def evaluate_final(self):return [p.accuracy() for p in self.protos]
