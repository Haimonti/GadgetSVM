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
from src.peersim_python.dynamics import make_topology
from src.nonlinear_sdca.kernel import make_kernel
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


    def _kernel_cfg(self):
        """Kernel settings with the polynomial scale fixed to one number.

        The projection preserves inner products (E<Ax, Ax'> = <x, x'>), so the
        projected-space kernel and the original-space kernel of the recovery step
        must use the SAME g; POLY_GAMMA=None resolves to 1/d of the raw data, not to
        sklearn's per-input 1/n_features (which would differ between x and Ax).
        """
        cfg = dict(self.config)
        if cfg.get("KERNEL", "rbf") == "poly" and cfg.get("POLY_GAMMA") is None:
            cfg["POLY_GAMMA"] = 1.0 / self.worker_data[0]["X_csr"].shape[1]
        return cfg

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
            make_topology(cfg, self.LINKABLE_PID),
            KernelDataInitializer(
                self.KERNEL_PID, self.worker_data, cfg["LAMBDA"], cfg["GOSSIP_K"],
                seed=cfg["SEED"], m=cfg["PROJECTION_DIM"], dist=cfg["PROJECTION_DIST"],
                gamma=cfg["RBF_GAMMA"], tau=cfg.get("DSRR_TAU", 0.0),
                local_steps=cfg.get("SDCA_LOCAL_STEPS"),
                step_scale=cfg.get("SDCA_STEP_SCALE"),
                kernel=make_kernel(self._kernel_cfg()),
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
            recovery=cfg.get("DSRR_RECOVERY", True),
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
        self.budget = cycles
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
