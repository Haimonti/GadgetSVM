"""PeerSim assemblies with barriers for the author CoLA/CHOCO round semantics."""
from src.peersim_python.core import Network, GeneralNode, CommonState
from src.peersim_python.idle_protocol import IdleProtocol
from src.peersim_python.observers import DataInitializer
from src.peersim_python.cdsim import CDSimulator, CDState
from src.peersim_python.cdsim.full_next_cycle import FullNextCycle
from src.peersim_python.communication import activation_order
from src.cocoa_simulation import CoCoASimulation
from p2p.linear_baselines import LinearBaselineProtocol, metropolis_matrix


class BarrierCycle:
    """All snapshots sent before any local update; deltas applied after all updates."""
    def execute(self):
        nodes = [Network.get(i) for i in range(Network.size())]
        for n in nodes:
            n.getProtocol(1).prepare_cycle(n)
        # Fixed order is safe because learners consume immutable round snapshots.
        for n in nodes:
            CommonState.setNode(n);CommonState.setPid(1)
            n.getProtocol(1).nextCycle(n, 1)
        for n in nodes:
            n.getProtocol(1).finish_cycle(n)
        for n in nodes:
            n.getProtocol(1).complete_cycle()
        return False


class MixingInitializer:
    def execute(self):
        adjacency = [[Network.get(i).getProtocol(0).getNeighbor(j).getIndex()
                      for j in range(Network.get(i).getProtocol(0).degree())]
                     for i in range(Network.size())]
        W = metropolis_matrix(adjacency)
        for i in range(Network.size()):
            Network.get(i).getProtocol(1).set_mixing(W)
        return False


class LinearBaselineSimulation:
    def __init__(self, config, worker_data, method):
        self.config, self.worker_data, self.method = config, worker_data, method
        self.sim = None

    def build(self):
        c = self.config
        CommonState.initializeRandom(c['SEED'])
        prototype = GeneralNode([IdleProtocol(), LinearBaselineProtocol(self.method,
            choco_lr=c.get('CHOCO_INITIAL_LR', .1),
            choco_estimate=c.get('CHOCO_ESTIMATE', 'final'))])
        Network.reset(c['NUM_WORKERS'], prototype)
        topology = CoCoASimulation(c, self.worker_data)._topology(c['TOPOLOGY'])
        init = [topology, DataInitializer(1, self.worker_data, c['LAMBDA'],
                c['GOSSIP_K'], warm_start=False, local_steps=c['SDCA_LOCAL_STEPS']),
                MixingInitializer()]
        self.sim = CDSimulator(0, initializers=init, activation=c.get('ACTIVATION','shuffle'))
        if self.method in ('cola', 'choco'):
            self.sim.runner = BarrierCycle()
        return self
