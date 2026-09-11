"""Assembly for a P2P-BDSVM run — the BDSVM twin of `Simulation`.

Identical to `src/peersim_python/simulation.py` in every respect that is not
the learner: same shared RNG seed, same topology controls, same
`DataInitializer`, same cycle-driven loop. Only two things differ, and both are
forced by the algorithm:

  * the node prototype carries `BDSVMGossipProtocol` instead of `SDCAProtocol`,
  * the measuring Control is `BDSVMEvaluator`, because BDSVM has no duality gap
    to stop on and uses the paper's eta on the iterate instead.

Keeping the assembly parallel is the point: anything that differs between an
SDCA curve and a BDSVM curve is then the algorithm, not the harness.
"""

from src.peersim_python.core import Network, GeneralNode, CommonState
from src.peersim_python.idle_protocol import IdleProtocol
from src.peersim_python.dynamics import (
    WireKOut, WireRing, WireFull, WireStar, WireMesh,
)
from src.peersim_python.observers import DataInitializer
from src.peersim_python.cdsim import CDSimulator
from src.peersim_python.logger import logger

from src.bdsvm_observers import BDSVMEvaluator
from p2p.bdsvm_gossip import BDSVMGossipProtocol

# BDSVM's own hyperparameters, as selected in docs_hyperparameters.md. They are
# not in CONFIG because CONFIG is Sreekar's file and describes SDCA's knobs;
# these are the BDSVM column of the same experiment.
BDSVM_PARAMS = {
    "P": 100,          # budget — number of pre-image vectors (covtype)
    "C": 30.0,         # penalty in the weighting rule, chosen on validation AUC
    "rho": 0.5,        # mixing weight in beta <- rho*beta + (1-rho)*beta_new
    "gamma": None,     # None -> median heuristic
    "preimage": "uniform",
    "arch_seed": 0,    # shared, so every node regenerates the same pre-images
}
BDSVM_ETA = 5e-3       # Algorithm 2 step 9; unchanged from the paper


class BDSVMSimulation:
    """Assembles and runs one gossip-BDSVM PeerSim experiment."""

    LINKABLE_PID = 0   # IdleProtocol (neighbour list)
    BDSVM_PID = 1      # BDSVMGossipProtocol (learner)

    def __init__(self, config, worker_data, params=None, eta=BDSVM_ETA):
        self.config = config
        self.worker_data = worker_data
        self.params = dict(BDSVM_PARAMS if params is None else params)
        self.eta = eta
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

        CommonState.initializeRandom(cfg["SEED"])

        prototype = GeneralNode([
            IdleProtocol(),
            BDSVMGossipProtocol(gossip_k=cfg["GOSSIP_K"], **self.params),
        ])
        Network.reset(cfg["NUM_WORKERS"], prototype)
        logger.info("network",
                    f"{cfg['NUM_WORKERS']} nodes built (protocol 0=Linkable, 1=BDSVM)")

        initializers = [
            self._topology(cfg["TOPOLOGY"]),
            DataInitializer(
                self.BDSVM_PID, self.worker_data, cfg["LAMBDA"], cfg["GOSSIP_K"],
                warm_start=cfg.get("WARM_START", False),
                local_steps=cfg.get("SDCA_LOCAL_STEPS"),
                step_scale=cfg.get("SDCA_STEP_SCALE"),
            ),
        ]
        controls = [BDSVMEvaluator(
            self.BDSVM_PID, eta=self.eta,
            eval_every=cfg.get("EVAL_EVERY", 1),
            stop_on_threshold=cfg.get("STOP_ON_THRESHOLD", False),
        )]

        self.sim = CDSimulator(
            cycles=0,
            initializers=initializers,
            controls=controls,
            activation=cfg.get("ACTIVATION", "shuffle"),
        )
        return self

    def run(self, cycles=None):
        cfg = self.config
        cycles = cycles if cycles is not None else cfg["ROUNDS"]
        if self.sim is None:
            self.build()
        self.sim.cycles = cycles
        for control in self.sim.controls:
            if isinstance(control, BDSVMEvaluator):
                control.total_cycles = cycles
        logger.info(
            "main",
            f"Training — topology={cfg['TOPOLOGY']}, k={cfg['GOSSIP_K']}, "
            f"max_cycles={cycles}, eta={self.eta}, "
            f"P={self.params['P']}, C={self.params['C']}, rho={self.params['rho']}",
        )
        stopped_at = self.sim.nextExperiment()
        logger.info("main", "Training complete")
        return stopped_at
