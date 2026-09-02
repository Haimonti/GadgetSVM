"""Simulation — the orchestrator that wires the PeerSim components together.

The single entry class that assembles a run: seed the shared RNG, build the
prototype node, clone the network, wire the topology, hand out data shards, set
up the observer controls, and drive the cycle-driven loop. It owns *assembly*
(the job PeerSim's config parser does); the per-cycle *loop* stays in
`CDSimulator`, which this class constructs and delegates to.

It receives already-made data shards (from `src/data_sharding.py`) and never
loads a dataset itself, so the engine stays data-agnostic.

    sim = Simulation(CONFIG, worker_data)
    sim.run()                        # or sim.run(cycles) to cap cycles
"""

from src.peersim_python.core import Network, GeneralNode, CommonState
from src.peersim_python.idle_protocol import IdleProtocol
from src.peersim_python.sdca_protocol import SDCAProtocol
from src.peersim_python.dynamics import (
    WireKOut, WireRing, WireFull, WireStar, WireMesh,
)
from src.peersim_python.observers import DataInitializer, ConvergenceObserver
from src.peersim_python.cdsim import CDSimulator
from src.peersim_python.logger import logger


class Simulation:
    """Assembles and runs one gossip-SDCA PeerSim experiment."""

    LINKABLE_PID = 0   # IdleProtocol (neighbour list)
    SDCA_PID = 1       # SDCAProtocol (learner)

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

        # Shared RNG (drives wiring, node visiting order, gossip peer choice)
        CommonState.initializeRandom(cfg["SEED"])

        # Each node = [IdleProtocol(links), SDCAProtocol(learner)]
        prototype = GeneralNode([IdleProtocol(), SDCAProtocol()])
        Network.reset(cfg["NUM_WORKERS"], prototype)
        logger.info("network",
                    f"{cfg['NUM_WORKERS']} nodes built (protocol 0=Linkable, 1=SDCA)")

        # init.* — wire topology, then assign data shards
        initializers = [
            self._topology(cfg["TOPOLOGY"]),
            DataInitializer(
                self.SDCA_PID, self.worker_data, cfg["LAMBDA"], cfg["T0_FRACTION"],
                cfg["GOSSIP_K"], sgd_init=cfg.get("SGD_INIT", False),
            ),
        ]
        # control.* — convergence-threshold stop
        controls = [ConvergenceObserver(self.SDCA_PID, cfg["GAP_THRESHOLD"])]

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
        logger.info(
            "main",
            f"Training — topology={cfg['TOPOLOGY']}, k={cfg['GOSSIP_K']}, "
            f"max_cycles={cycles}, gap_threshold={cfg['GAP_THRESHOLD']}",
        )
        stopped_at = self.sim.nextExperiment()
        logger.info("main", "Training complete")
        return stopped_at
