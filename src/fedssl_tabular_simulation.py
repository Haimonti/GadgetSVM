"""Tabular backend on the validated FedSSL PeerSim transport."""
from paper_reference import ssl_tabular
from p2p.fedssl_gossip import FedSSLGossipProtocol
from src.fedssl_simulation import FedSSLSimulation
from src.peersim_python.core import Network,Control
from src.train_convergence import WindowStability


class TabularFedSSLProtocol(FedSSLGossipProtocol):
    backend=ssl_tabular


class TabularStop(Control):
    """Offline train-only simulation observer, same deployment caveat as SVMs.

    Reads only local training feature tensors and each final aggregate encoder.
    No test data, labels, classifier fitting or hidden model exchange in stop.
    Its hypothetical deployment traffic is excluded; CPU time is included.
    """
    def __init__(self,config):
        self.config=config
        self.rule=WindowStability(1,config.stability_window,config.stability_checks,config.rtol)
        self.history=[];self.last_epoch=0;self.converged=False

    def execute(self):
        ps=[Network.get(i).getProtocol(1) for i in range(Network.size())]
        epoch=min(p.completed_epochs for p in ps)
        if epoch<=self.last_epoch:return False
        self.last_epoch=epoch
        losses=[ssl_tabular.monitor_loss(p.model,p.X,self.config,i) for i,p in enumerate(ps)]
        weights=[1/len(ps)]*len(ps) if self.config.aggregation=='uniform' else [p.n/sum(q.n for q in ps) for p in ps]
        value=sum(w*v for w,v in zip(weights,losses))
        self.converged=self.rule.update([value],epoch)
        self.history.append(dict(training_round=epoch,link_cycle=epoch*len(ps),
            train_contrastive_loss=value,total_comm_bytes=sum(p.comm_bytes for p in ps),
            per_node_monitor_loss=losses))
        if epoch%10==0 or self.converged or epoch==self.config.rounds:
            print(f'tabular SSL round={epoch} fixed_train_loss={value:.7g} plateau={self.converged}',flush=True)
        if self.converged or epoch==self.config.rounds:
            reason='train_contrastive_plateau' if self.converged else 'max_training_rounds'
            for p in ps:p.finish(reason)
            return True
        return False


class TabularFedSSLSimulation(FedSSLSimulation):
    protocol_type=TabularFedSSLProtocol

    def __init__(self,*args,resume=None,**kwargs):
        super().__init__(*args,**kwargs)
        self.resume=resume

    def build(self):
        super().build()
        self.observer=TabularStop(self.config)
        if self.resume is not None:
            state,previous=self.resume
            start=previous['training_rounds']
            if previous['converged'] or not 0<start<self.config.rounds:
                raise ValueError('Only extend an unfinished encoder run with a larger round budget')
            self.observer.history=list(previous['history'])
            self.observer.last_epoch=start
            for item in self.observer.history:
                self.observer.rule.update([item['train_contrastive_loss']],item['training_round'])
            self.sim.initializers.append(ResumeEncoder(state,previous))
            self.sim.cycles=(self.config.rounds-start)*len(self.u)
        self.sim.controls=[self.observer]
        return self

    def run(self):
        result=super().run()
        result.update(link_cycles=self.protos[0].completed_epochs*len(self.protos),
            training_rounds=self.protos[0].completed_epochs,
            converged=self.observer.converged,stop_reason=self.protos[0].stop_reason,
            history=self.observer.history,
            monitoring='fixed local training feature samples, offline observer; excludes deployment monitor traffic')
        return result


class ResumeEncoder(Control):
    def __init__(self,state,previous):self.state,self.previous=state,previous

    def execute(self):
        for i in range(Network.size()):
            p=Network.get(i).getProtocol(1)
            p.model.load_state_dict(self.state)
            p.start_epoch=p.completed_epochs=self.previous['training_rounds']
            p.comm_bytes=self.previous['per_node_comm_bytes'][i]
            p.messages=self.previous['per_node_metadata'][i]['messages']
        return False
