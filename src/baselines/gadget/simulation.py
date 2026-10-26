"""GadgetSimulation — orchestrator for the GADGET SVM baseline.

Same role as `src/peersim_python/simulation.py`, but it wires the Pegasos
weight-averaging learner (`PegasosProtocol`) and the GADGET observers onto the
*same* PeerSim engine (`core/`, `cdsim/`), topology and config. SDCA's own
`Simulation` is untouched, so both drivers share the engine and only differ in
the model they assemble.

    sim = GadgetSimulation(CONFIG, worker_data)
    sim.run()                        # or sim.run(cycles) to cap cycles
"""

from src.peersim_python.core import Network, GeneralNode, CommonState
from src.peersim_python.idle_protocol import IdleProtocol
from src.peersim_python.dynamics import (
    WireKOut, WireRing, WireFull, WireStar, WireMesh,
)
from src.peersim_python.cdsim import CDSimulator
from src.peersim_python.logger import logger

from src.baselines.gadget.pegasos_protocol import PegasosProtocol
from src.baselines.gadget.observers import PegasosDataInitializer, PegasosEvaluator


class GadgetSimulation:
    """Assembles and runs one GADGET (Pegasos + weight-averaging) experiment."""

    LINKABLE_PID = 0   # IdleProtocol (neighbour list)
    PEGASOS_PID = 1    # PegasosProtocol (learner)

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

        # Shared RNG (drives wiring, node visiting order, gossip peer choice)
        CommonState.initializeRandom(cfg["SEED"])

        # Each node = [IdleProtocol(links), PegasosProtocol(learner)]
        prototype = GeneralNode([IdleProtocol(), PegasosProtocol()])
        Network.reset(cfg["NUM_WORKERS"], prototype)
        logger.info("network",
                    f"{cfg['NUM_WORKERS']} nodes built (protocol 0=Linkable, 1=Pegasos)")

        # init.* — wire topology, then assign data shards
        initializers = [
            self._topology(cfg["TOPOLOGY"]),
            PegasosDataInitializer(
                self.PEGASOS_PID, self.worker_data, cfg["LAMBDA"], cfg["GOSSIP_K"],
                local_steps=cfg.get("SDCA_LOCAL_STEPS"),
            ),
        ]
        # control.* — global measurement. Two independent early stops (both
        # opt-in): convergence (every node's objective settled) and plateau
        # (the global mean objective stops improving).
        controls = [PegasosEvaluator(
            self.PEGASOS_PID, cfg["OBJ_THRESHOLD"],
            convergence_count=cfg.get("CONVERGENCE_COUNT", 10),
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
            if isinstance(control, PegasosEvaluator):
                control.total_cycles = cycles
        logger.info(
            "main",
            f"Training GADGET — topology={cfg['TOPOLOGY']}, k={cfg['GOSSIP_K']}, "
            f"max_cycles={cycles}, obj_threshold={cfg['OBJ_THRESHOLD']}",
        )
        stopped_at = self.sim.nextExperiment()
        logger.info("main", "Training complete")
        return stopped_at
