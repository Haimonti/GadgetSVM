"""ChocoSimulation — CHOCO-SGD on the shared PeerSim engine, same setup as every tenant.

Same seed, graph (`make_topology`), shards and metrics as linear SDCA / GADGET;
CHOCO's own pieces — Metropolis W on that graph, synchronous all-neighbour rounds
(`SynchronousBarrier`), SGD + compressed CHOCO-Gossip — are the algorithm.

    sim = ChocoSimulation(CONFIG, worker_data)
    sim.run()
"""


from src.peersim_python.core import Network, GeneralNode, CommonState
from src.peersim_python.idle_protocol import IdleProtocol
from src.peersim_python.dynamics import make_topology
from src.peersim_python.cdsim import CDSimulator
from src.peersim_python.sync_protocol import SynchronousBarrier
from src.peersim_python.logger import logger
from src.baselines.linear_evaluator import SyncLinearEvaluator
from src.baselines.choco.choco_protocol import ChocoProtocol
from src.baselines.choco.observers import ChocoDataInitializer


class ChocoSimulation:

    LINKABLE_PID = 0
    CHOCO_PID = 1

    def __init__(self, config, worker_data):
        self.config = config
        self.worker_data = worker_data
        self.sim = None

    def _protos(self):
        return [Network.get(i).getProtocol(self.CHOCO_PID) for i in range(Network.size())]

    def build(self):
        cfg = self.config
        if cfg.get("ACTIVATION", "shuffle") == "getpair_rand":
            raise ValueError("CHOCO-SGD needs synchronous rounds; ACTIVATION=getpair_rand is not allowed")
        CommonState.initializeRandom(cfg["SEED"])
        Network.reset(cfg["NUM_WORKERS"], GeneralNode([IdleProtocol(), ChocoProtocol()]))
        logger.info("network", f"{cfg['NUM_WORKERS']} nodes built (protocol 0=Linkable, 1=CHOCO)")
        initializers = [
            make_topology(cfg, self.LINKABLE_PID),
            ChocoDataInitializer(self.CHOCO_PID, self.LINKABLE_PID, self.worker_data, cfg),
        ]
        controls = [
            SynchronousBarrier(self.CHOCO_PID),
            SyncLinearEvaluator(
                self.CHOCO_PID, cfg["LAMBDA"], cfg["GAP_THRESHOLD"],
                eval_every=cfg.get("EVAL_EVERY", 1),
                stop_on_threshold=cfg.get("STOP_ON_THRESHOLD", False),
                stop_on_plateau=False,              # fixed budget (see run())
                plateau_patience=cfg.get("PLATEAU_PATIENCE", 20),
                plateau_rel_eps=cfg.get("PLATEAU_REL_EPS", 1e-3),
                extra_fn=lambda p: {"lr": p.lr(max(p.t - 1, 0))},
                tag="choco"),
        ]
        self.sim = CDSimulator(cycles=0, initializers=initializers, controls=controls,
                               activation=cfg.get("ACTIVATION", "shuffle"))
        return self

    def run(self, cycles=None):
        """CHOCO-SGD takes CHOCO_BATCH samples per round and, as in the official
        experiments, runs a FIXED budget (no early stop: a noisy SGD objective trips
        a plateau rule long before convergence — measured on a9a). The budget is
        max(CHOCO_EPOCHS epochs, CHOCO_MIN_ROUNDS SGD steps, ROUNDS): SGD accuracy
        depends on the number of steps, so small shards still get enough of them
        (tiny a9a: ~45k steps reach ~2% of P*). Evaluation is thinned to <= ~500 points."""
        import math
        cfg = self.config
        if cycles is None:
            n_local = max(w["n_local"] for w in self.worker_data)
            by_epochs = math.ceil(cfg.get("CHOCO_EPOCHS", 20) * n_local / cfg.get("CHOCO_BATCH", 1))
            cycles = max(cfg["ROUNDS"], by_epochs, cfg.get("CHOCO_MIN_ROUNDS", 50000))
        self.budget = cycles
        if self.sim is None:
            self.build()
        self.sim.cycles = cycles
        for c in self.sim.controls:
            if isinstance(c, SyncLinearEvaluator):
                c.total_cycles = cycles
                c.eval_every = max(c.eval_every, cycles // 500)
        logger.info("main", f"Training CHOCO-SGD — topology={cfg['TOPOLOGY']}, max_cycles={cycles}")
        stopped_at = self.sim.nextExperiment()
        logger.info("main", "Training complete")
        return stopped_at
