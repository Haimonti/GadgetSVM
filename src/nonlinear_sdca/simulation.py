"""KernelSDCASimulation — orchestrator for the non-linear (kernel) tenant.

Same role as `src/peersim_python/simulation.py` and `src/baselines/gadget/simulation.py`,
but it wires the RBF kernel SDCA learner (`KernelSDCAProtocol`) and the kernel
observers onto the *same* PeerSim engine (`core/`, `cdsim/`), topology and config.
SDCA's and GADGET's orchestrators are untouched; all three share the engine and
differ only in the model they assemble.

    sim = KernelSDCASimulation(CONFIG, worker_data)
    sim.run()                        # or sim.run(cycles) to cap cycles
"""

from src.peersim_python.core import Network, GeneralNode, CommonState
from src.peersim_python.idle_protocol import IdleProtocol
from src.peersim_python.dynamics import (
    WireKOut, WireRing, WireFull, WireStar, WireMesh,
)
from src.peersim_python.cdsim import CDSimulator
from src.peersim_python.logger import logger

from src.nonlinear_sdca.kernel_sdca_protocol import KernelSDCAProtocol
from src.nonlinear_sdca.observers import KernelDataInitializer, KernelEvaluator


class KernelSDCASimulation:
    """Assembles and runs one RBF-kernel gossip-SDCA experiment over DSRR projections."""

    LINKABLE_PID = 0   # IdleProtocol (neighbour list)
    KERNEL_PID = 1     # KernelSDCAProtocol (learner)

    def __init__(self, config, worker_data):
        self.config = config
        self.worker_data = worker_data
        self.sim = None

    def _topology(self, name):
        """Map a CONFIG topology name to the matching Wire* control (undirected)."""
        if name == "random_kout":
            return WireKOut(self.LINKABLE_PID, self.config["GOSSIP_K"], undir=True)
        if name == "ring":
            return WireRing(self.LINKABLE_PID, undir=True)
        if name == "full":
            return WireFull(self.LINKABLE_PID, undir=True)
        if name == "star":
            return WireStar(self.LINKABLE_PID, undir=True)
        if name == "mesh":
            return WireMesh(self.LINKABLE_PID, undir=True)
        raise ValueError(
            f"Unknown topology '{name}'. Choose: random_kout | ring | full | star | mesh"
        )

    def build(self):
        """Seed RNG, build the network, and assemble initializers + controls."""
        cfg = self.config

        # Shared RNG (drives wiring, node visiting order, gossip peer choice). The
        # DSRR projection matrix A is seeded separately from the same SEED inside
        # each node's set_data, so every node builds the identical A.
        CommonState.initializeRandom(cfg["SEED"])

        prototype = GeneralNode([IdleProtocol(), KernelSDCAProtocol()])
        Network.reset(cfg["NUM_WORKERS"], prototype)
        logger.info("network",
                    f"{cfg['NUM_WORKERS']} nodes built (protocol 0=Linkable, 1=KernelSDCA)")

        initializers = [
            self._topology(cfg["TOPOLOGY"]),
            KernelDataInitializer(
                self.KERNEL_PID, self.worker_data, cfg["LAMBDA"], cfg["GOSSIP_K"],
                seed=cfg["SEED"], m=cfg["PROJECTION_DIM"], dist=cfg["PROJECTION_DIST"],
                gamma=cfg["RBF_GAMMA"], tau=cfg.get("DSRR_TAU", 0.0),
                local_steps=cfg.get("SDCA_LOCAL_STEPS"),
                step_scale=cfg.get("SDCA_STEP_SCALE"),
            ),
        ]
        # control.* — global measurement. Two independent early stops (both opt-in):
        # convergence (every node's kernel gap < GAP_THRESHOLD) and plateau.
        controls = [KernelEvaluator(
            self.KERNEL_PID, cfg["GAP_THRESHOLD"],
            eval_every=cfg.get("EVAL_EVERY", 1),
            stop_on_threshold=cfg.get("STOP_ON_THRESHOLD", False),
            stop_on_plateau=cfg.get("STOP_ON_PLATEAU", False),
            plateau_patience=cfg.get("PLATEAU_PATIENCE", 20),
            plateau_rel_eps=cfg.get("PLATEAU_REL_EPS", 1e-3),
        )]

        self.sim = CDSimulator(
            cycles=0,  # set in run()
            initializers=initializers,
            controls=controls,
            activation=cfg.get("ACTIVATION", "shuffle"),
        )
        return self

    def run(self, cycles=None):
        """Build (if needed) and run the experiment; returns the stopping cycle."""
        cfg = self.config
        cycles = cycles if cycles is not None else cfg["ROUNDS"]
        if self.sim is None:
            self.build()
        self.sim.cycles = cycles
        for control in self.sim.controls:
            if isinstance(control, KernelEvaluator):
                control.total_cycles = cycles
        logger.info(
            "main",
            f"Training kernel-SDCA — topology={cfg['TOPOLOGY']}, k={cfg['GOSSIP_K']}, "
            f"max_cycles={cycles}, gap_threshold={cfg['GAP_THRESHOLD']}",
        )
        stopped_at = self.sim.nextExperiment()
        logger.info("main", "Training complete")
        return stopped_at
