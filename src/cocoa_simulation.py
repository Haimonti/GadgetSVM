"""Assemble and run one gossip-CoCoA / CoCoA+ experiment, mirroring StallAwareSimulation.

Same shared RNG seed, same topology controls, same DataInitializer, same
config as SDCA, BDSVM and FedAvg — only the learner in the node prototype
differs. CoCoA is primal-dual like SDCA, so it is measured and stopped by the
same `StallAwareGapEvaluator` (global duality gap, or the iterate stalling, and
only once every origin has been heard from), at SDCA's stall tolerance.
"""

from src.peersim_python.core import Network, GeneralNode, CommonState
from src.peersim_python.idle_protocol import IdleProtocol
from src.peersim_python.dynamics import (
    WireKOut, WireRing, WireFull, WireStar, WireMesh,
)
from src.peersim_python.observers import DataInitializer
from src.peersim_python.cdsim import CDSimulator
from src.peersim_python.logger import logger

from src.sdca_observers import StallAwareGapEvaluator
from p2p.cocoa_gossip import CoCoAGossipProtocol, CoCoAPlusGossipProtocol

# CoCoA's own aggregation parameters, at the values of cocoa/cocoa.py and
# cocoa/cocoa_plus.py. Everything else (lambda, H, seed, topology) is CONFIG.
COCOA_PARAMS = {"beta": 1.0, "gamma": 1.0}
# SDCA's stall tolerance (src/sdca_observers.py), so the two primal-dual
# learners stop by the same rule.
COCOA_STALL_TOL = 1e-3

VARIANTS = {
    "cocoa": CoCoAGossipProtocol,
    "cocoa_plus": CoCoAPlusGossipProtocol,
}


class CoCoASimulation:
    LINKABLE_PID = 0
    COCOA_PID = 1

    def __init__(self, config, worker_data, variant="cocoa", params=None,
                 stall_tol=COCOA_STALL_TOL):
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant '{variant}'. Choose: {sorted(VARIANTS)}")
        self.config = config
        self.worker_data = worker_data
        self.variant = variant
        self.params = dict(COCOA_PARAMS if params is None else params)
        self.stall_tol = stall_tol
        self.sim = None

    def _topology(self, name):
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
        raise ValueError(f"Unknown topology '{name}'")

    def build(self):
        cfg = self.config
        CommonState.initializeRandom(cfg["SEED"])
        learner = VARIANTS[self.variant](**self.params)
        learner.gossip_k = cfg["GOSSIP_K"]
        prototype = GeneralNode([IdleProtocol(), learner])
        Network.reset(cfg["NUM_WORKERS"], prototype)
        logger.info("network",
                    f"{cfg['NUM_WORKERS']} nodes built "
                    f"(protocol 0=Linkable, 1={self.variant})")
        initializers = [
            self._topology(cfg["TOPOLOGY"]),
            DataInitializer(
                self.COCOA_PID, self.worker_data, cfg["LAMBDA"], cfg["GOSSIP_K"],
                warm_start=cfg.get("WARM_START", False),
                local_steps=cfg.get("SDCA_LOCAL_STEPS"),
                step_scale=cfg.get("SDCA_STEP_SCALE"),
            ),
        ]
        controls = [StallAwareGapEvaluator(
            self.COCOA_PID, cfg["GAP_THRESHOLD"],
            eval_every=cfg.get("EVAL_EVERY", 1),
            stop_on_threshold=cfg.get("STOP_ON_THRESHOLD", False),
            stall_tol=self.stall_tol,
        )]
        self.sim = CDSimulator(cycles=0, initializers=initializers,
                               controls=controls,
                               activation=cfg.get("ACTIVATION", "shuffle"))
        return self

    def run(self, cycles=None):
        cfg = self.config
        cycles = cycles if cycles is not None else cfg["ROUNDS"]
        if self.sim is None:
            self.build()
        self.sim.cycles = cycles
        for control in self.sim.controls:
            if isinstance(control, StallAwareGapEvaluator):
                control.total_cycles = cycles
        logger.info("main",
                    f"Training — {self.variant}, topology={cfg['TOPOLOGY']}, "
                    f"k={cfg['GOSSIP_K']}, max_cycles={cycles}, "
                    f"stall_tol={self.stall_tol}, params={self.params}, "
                    f"local_steps={cfg.get('SDCA_LOCAL_STEPS')}, lambda={cfg['LAMBDA']}")
        stopped_at = self.sim.nextExperiment()
        logger.info("main", "Training complete")
        return stopped_at
