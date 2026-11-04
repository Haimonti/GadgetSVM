"""ColaSimulation — COLA on the shared PeerSim engine, same setup as every tenant.

Same seed, graph (`make_topology`), shards and metrics as linear SDCA / GADGET;
COLA's own pieces — Metropolis W on that graph, synchronous all-neighbour rounds
(`SynchronousBarrier`), the LinearSVM local solver — are the algorithm.

    sim = ColaSimulation(CONFIG, worker_data)
    sim.run()
"""


from src.peersim_python.core import Network, GeneralNode, CommonState
from src.peersim_python.idle_protocol import IdleProtocol
from src.peersim_python.dynamics import make_topology
from src.peersim_python.cdsim import CDSimulator
from src.peersim_python.sync_protocol import SynchronousBarrier
from src.peersim_python.logger import logger
from src.baselines.linear_evaluator import SyncLinearEvaluator
from src.baselines.cola.cola_protocol import ColaProtocol
from src.baselines.cola.observers import ColaDataInitializer, cola_dual, cola_extras


class ColaSimulation:

    LINKABLE_PID = 0
    COLA_PID = 1

    def __init__(self, config, worker_data):
        self.config = config
        self.worker_data = worker_data
        self.sim = None

    def _protos(self):
        return [Network.get(i).getProtocol(self.COLA_PID) for i in range(Network.size())]

    def build(self):
        cfg = self.config
        if cfg.get("ACTIVATION", "shuffle") == "getpair_rand":
            raise ValueError("COLA needs synchronous rounds; ACTIVATION=getpair_rand is not allowed")
        CommonState.initializeRandom(cfg["SEED"])
        Network.reset(cfg["NUM_WORKERS"], GeneralNode([IdleProtocol(), ColaProtocol()]))
        logger.info("network", f"{cfg['NUM_WORKERS']} nodes built (protocol 0=Linkable, 1=COLA)")
        initializers = [
            make_topology(cfg, self.LINKABLE_PID),
            ColaDataInitializer(self.COLA_PID, self.LINKABLE_PID, self.worker_data, cfg),
        ]
        controls = [
            SynchronousBarrier(self.COLA_PID),
            SyncLinearEvaluator(
                self.COLA_PID, cfg["LAMBDA"], cfg["GAP_THRESHOLD"],
                eval_every=cfg.get("EVAL_EVERY", 1),
                stop_on_threshold=cfg.get("STOP_ON_THRESHOLD", False),
                stop_on_plateau=cfg.get("STOP_ON_PLATEAU", False),
                plateau_patience=cfg.get("PLATEAU_PATIENCE", 20),
                plateau_rel_eps=cfg.get("PLATEAU_REL_EPS", 1e-3),
                dual_fn=cola_dual(cfg["LAMBDA"]), extra_fn=cola_extras(self._protos),
                tag="cola"),
        ]
        self.sim = CDSimulator(cycles=0, initializers=initializers, controls=controls,
                               activation=cfg.get("ACTIVATION", "shuffle"))
        return self

    def run(self, cycles=None):
        cfg = self.config
        # COLA's gap shrinks ~O(1/t): measured on a9a (lambda*n ~ 3) it needs ~7k
        # rounds to reach a 1e-4 gap, so its budget is max(ROUNDS, COLA_ROUNDS);
        # rounds are cheap, evaluation is thinned to <= ~500 points.
        cycles = cycles if cycles is not None else max(cfg["ROUNDS"], cfg.get("COLA_ROUNDS", 20000))
        self.budget = cycles
        if self.sim is None:
            self.build()
        self.sim.cycles = cycles
        for c in self.sim.controls:
            if isinstance(c, SyncLinearEvaluator):
                c.total_cycles = cycles
                c.eval_every = max(c.eval_every, cycles // 500)
        logger.info("main", f"Training COLA — topology={cfg['TOPOLOGY']}, max_cycles={cycles}")
        stopped_at = self.sim.nextExperiment()
        logger.info("main", "Training complete")
        return stopped_at
