"""NDSVMSimulation — orchestrator for the MoM-NDSVM baseline.

Built exactly like the other tenants' simulations
(``src/baselines/gadget/simulation.py``): the same ``CommonState.initializeRandom``
seed, the same topology control as the *first* initializer (so the identical
undirected graph), the same ``CDSimulator`` with the same activation setting and
cycle budget. Only the model differs. The tenant draws its own randomness from
numpy generators seeded by ``SEED``, never from ``CommonState.r``, so it cannot
perturb the wiring or the activation stream.

    sim = NDSVMSimulation(CONFIG, worker_data)
    sim.run()                        # or sim.run(cycles) to cap cycles
"""

from src.peersim_python.core import Network, GeneralNode, CommonState
from src.peersim_python.idle_protocol import IdleProtocol
from src.peersim_python.dynamics import (
    WireKOut, WireRing, WireFull, WireStar, WireMesh,
)
from src.peersim_python.cdsim import CDSimulator
from src.peersim_python.logger import logger
from src.nonlinear_sdca.kernel import rbf_gram

from src.baselines.mom_ndsvm.ndsvm_protocol import NDSVMProtocol
from src.baselines.mom_ndsvm.observers import (
    NDSVMDataInitializer, SynchronousRoundBarrier, NDSVMEvaluator,
)


class NDSVMSimulation:
    """Assembles and runs one MoM-NDSVM experiment."""

    LINKABLE_PID = 0   # IdleProtocol (neighbour list)
    NDSVM_PID = 1      # NDSVMProtocol (learner)

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
        cfg = self.config
        activation = cfg.get("ACTIVATION", "shuffle")
        if activation == "getpair_rand":
            raise ValueError("MoM-NDSVM needs every node once per round; "
                             "ACTIVATION='getpair_rand' samples nodes with replacement")

        # Shared RNG (drives wiring and node visiting order), seeded as in every tenant
        CommonState.initializeRandom(cfg["SEED"])

        prototype = GeneralNode([IdleProtocol(), NDSVMProtocol()])
        Network.reset(cfg["NUM_WORKERS"], prototype)
        logger.info("network",
                    f"{cfg['NUM_WORKERS']} nodes built (protocol 0=Linkable, 1=MoM-NDSVM)")

        gamma = cfg["RBF_GAMMA"]
        initializers = [
            self._topology(cfg["TOPOLOGY"]),
            NDSVMDataInitializer(
                self.NDSVM_PID, self.LINKABLE_PID, self.worker_data, cfg["LAMBDA"],
                kernel=lambda A, B: rbf_gram(A, B, gamma),
                L=cfg["NDSVM_L"], eta=cfg["NDSVM_ETA"], seed=cfg["SEED"],
                qp_tol=cfg["NDSVM_QP_TOL"], qp_max_passes=cfg["NDSVM_QP_MAX_SWEEPS"],
                features=cfg.get("NDSVM_FEATURES", "raw"),
                projection={"m": cfg.get("PROJECTION_DIM"), "dist": cfg.get("PROJECTION_DIST")},
            ),
        ]
        # Barrier first (Algorithm 3's broadcast + dual step), then measurement.
        controls = [
            SynchronousRoundBarrier(self.NDSVM_PID),
            NDSVMEvaluator(
                self.NDSVM_PID, cfg["NDSVM_RESIDUAL_TOL"],
                eval_every=cfg.get("EVAL_EVERY", 1),
                stop_on_threshold=cfg.get("STOP_ON_THRESHOLD", False),
                stop_on_plateau=cfg.get("STOP_ON_PLATEAU", False),
                plateau_patience=cfg.get("PLATEAU_PATIENCE", 20),
                plateau_rel_eps=cfg.get("PLATEAU_REL_EPS", 1e-3),
            ),
        ]

        self.sim = CDSimulator(
            cycles=0,  # set in run()
            initializers=initializers,
            controls=controls,
            activation=activation,
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
            if isinstance(control, NDSVMEvaluator):
                control.total_cycles = cycles
        logger.info(
            "main",
            f"Training MoM-NDSVM — topology={cfg['TOPOLOGY']}, k={cfg['GOSSIP_K']}, "
            f"max_cycles={cycles}, L={cfg['NDSVM_L']}, eta={cfg['NDSVM_ETA']}",
        )
        stopped_at = self.sim.nextExperiment()
        logger.info("main", "Training complete")
        return stopped_at
