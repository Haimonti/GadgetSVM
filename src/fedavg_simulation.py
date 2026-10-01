"""Assemble and run one gossip-FedAvg experiment, mirroring BDSVMSimulation.

Same shared RNG seed, same topology controls, same DataInitializer, same
config — only the learner in the node prototype and the evaluator differ.
"""

from src.peersim_python.core import Network, GeneralNode, CommonState
from src.peersim_python.idle_protocol import IdleProtocol
from src.peersim_python.dynamics import (
    WireKOut, WireRing, WireFull, WireStar, WireMesh,
)
from src.peersim_python.observers import DataInitializer
from src.peersim_python.cdsim import CDSimulator
from src.peersim_python.logger import logger

from src.fedavg_observers import FedAvgEvaluator
from p2p.fedavg_gossip import FedAvgGossipProtocol

# The stall tolerance SDCA settled on (see src/sdca_observers.py): 1e-4 was
# never reached by a still-converging gossip iterate, 1e-3 stops at unchanged
# accuracy. FedAvg is the same kind of primal iterate, so it starts there.
FEDAVG_STALL_TOL = 1e-3


class FedAvgSimulation:
    LINKABLE_PID = 0
    FEDAVG_PID = 1

    def __init__(self, config, worker_data, stall_tol=FEDAVG_STALL_TOL):
        self.config = config
        self.worker_data = worker_data
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
        prototype = GeneralNode([
            IdleProtocol(),
            FedAvgGossipProtocol(gossip_k=cfg["GOSSIP_K"]),
        ])
        Network.reset(cfg["NUM_WORKERS"], prototype)
        logger.info("network",
                    f"{cfg['NUM_WORKERS']} nodes built (protocol 0=Linkable, 1=FedAvg)")
        initializers = [
            self._topology(cfg["TOPOLOGY"]),
            DataInitializer(
                self.FEDAVG_PID, self.worker_data, cfg["LAMBDA"], cfg["GOSSIP_K"],
                warm_start=False,
                local_steps=cfg.get("SDCA_LOCAL_STEPS"),
                step_scale=cfg.get("SDCA_STEP_SCALE"),
            ),
        ]
        controls = [FedAvgEvaluator(
            self.FEDAVG_PID, stall_tol=self.stall_tol,
            eval_every=cfg.get("EVAL_EVERY", 1),
            stop_on_threshold=cfg.get("STOP_ON_THRESHOLD", False),
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
            if isinstance(control, FedAvgEvaluator):
                control.total_cycles = cycles
        logger.info("main",
                    f"Training — topology={cfg['TOPOLOGY']}, k={cfg['GOSSIP_K']}, "
                    f"max_cycles={cycles}, stall_tol={self.stall_tol}, "
                    f"local_steps={cfg.get('SDCA_LOCAL_STEPS')}, lambda={cfg['LAMBDA']}")
        stopped_at = self.sim.nextExperiment()
        logger.info("main", "Training complete")
        return stopped_at
