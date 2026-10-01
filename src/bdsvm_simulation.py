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

# BDSVM's own hyperparameters. They are not in CONFIG because CONFIG is
# Sreekar's file and describes SDCA's knobs; these are the BDSVM column of the
# same experiment.
#
# Two of them have to be chosen per dataset, and `_make_preimages`' own
# docstring says why:
#
#   preimage  A dense pre-image in a sparse high-dimensional space is nearly
#             orthogonal to every data point, so no gamma can make it
#             discriminate — measured on rcv1, a dense unit pre-image puts every
#             ||x-p||^2 inside [1.957, 2.039]. Sparse data therefore needs
#             "sparse"; data already min-max scaled into [0,1] takes "uniform".
#   P         The budget has to scale with the intrinsic dimension: 100 was
#             enough for covtype's 54 features, rcv1's 47k needed 1000.
#
# C and the gamma multiplier are selected per dataset by tune_bdsvm.py — a grid
# over C x P x gamma on a validation split carved from the P2P run's own
# training shards, chosen by validation AUC at the eta stop, with the trajectory
# checked so a configuration that peaks at epoch 1 and decays is not mistaken
# for a good one. covtype's C = 30 is the original docs_hyperparameters.md
# selection. Grids are under results/tune/bdsvm_<dataset>.csv.
#
# Carrying covtype's C = 30 / gamma x1 onto the other three was what produced
# gisette's falling accuracy (peak at epoch 1, then decay) and real-sim's
# 5000-cycle limit cycle. Re-tuning fixed gisette's decay and rcv1, but not
# real-sim: at the tuned values it still collapsed every 40 cycles in the
# gossip run while the same values were stable centrally. The mechanism is
# Eq (9)'s weight a_i = 2C/(e_i y_i), unbounded as a sample approaches the
# margin. Centralised IRWLS self-corrects the resulting blow-up in one
# synchronous epoch; the gossip version *publishes* the blown-up (C_k, d_k)
# and every peer that merges it blows up too, so the collapse propagates and
# takes ~20 cycles of fresh versions to flush. `ey_floor` bounds the weight at
# 2C/ey_floor. At 1e-2, real-sim converges monotonically and early-stops at
# cycles 40-50; rho = 0.9 only slowed the collapse. covtype keeps the paper's
# unbounded rule because its results were produced with it and are not rerun.
BDSVM_DEFAULTS = {
    "rho": 0.5,        # mixing weight in beta <- rho*beta + (1-rho)*beta_new
    "gamma": None,     # None -> gamma_mult x median heuristic
    "arch_seed": 0,    # shared, so every node regenerates the same pre-images
}
BDSVM_PER_DATASET = {
    # dataset      P     preimage    C       gamma   ey_floor   selected on
    "covtype":   (100,  "uniform",  30.0,   1.0,    1e-12),  # docs_hyperparameters.md; paper's rule
    "gisette":   (1000, "uniform",  1000.0, 0.5,    1e-2),   # val AUC 0.9878, peak@12, decay 0.0012
    "real-sim":  (1000, "sparse",   100.0,  0.5,    1e-2),   # val AUC 0.9438, peak@62, decay 0.0005
    "rcv1":      (1000, "sparse",   300.0,  1.0,    1e-2),   # val AUC 0.9489, peak@98, decay 0.0016
    # The four sreekar@1f73416 datasets. Pre-image kind was part of the grid
    # here (all three tried), and P is the smallest within 0.003 AUC of the
    # best — on a9a and webspam that is a 100x / 11x cheaper payload for a
    # difference inside the noise.
    "ijcnn1":    (1000, "unit",     300.0,  2.0,    1e-2),   # val AUC 0.9901; unit > sparse > uniform
    "a9a":       (100,  "uniform",  300.0,  1.0,    1e-2),   # val AUC 0.9003 vs 0.9011 best; flat plateau
    "w8a":       (1000, "uniform",  1000.0, 1.0,    1e-2),   # val AUC 0.9431; uniform >> sparse, unit
    "webspam":   (300,  "sparse",   1000.0, 2.0,    1e-2),   # val AUC 0.9702 vs 0.9723 best; sparse > unit
}
BDSVM_ETA = 5e-3       # Algorithm 2 step 9; unchanged from the paper


def params_for(dataset):
    """BDSVM hyperparameters for one dataset name (as CONFIG['DATASET'] holds it)."""
    if dataset not in BDSVM_PER_DATASET:
        raise ValueError(f"No BDSVM pre-image setting for dataset '{dataset}'. "
                         f"Known: {sorted(BDSVM_PER_DATASET)}")
    P, preimage, C, gamma_mult, ey_floor = BDSVM_PER_DATASET[dataset]
    return {**BDSVM_DEFAULTS, "P": P, "preimage": preimage,
            "C": C, "gamma_mult": gamma_mult, "ey_floor": ey_floor}


# Back-compat for callers that just want the covtype column.
BDSVM_PARAMS = params_for("covtype")


class BDSVMSimulation:
    """Assembles and runs one gossip-BDSVM PeerSim experiment."""

    LINKABLE_PID = 0   # IdleProtocol (neighbour list)
    BDSVM_PID = 1      # BDSVMGossipProtocol (learner)

    def __init__(self, config, worker_data, params=None, eta=BDSVM_ETA):
        self.config = config
        self.worker_data = worker_data
        self.params = dict(params if params is not None
                           else params_for(config["DATASET"]))
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
            f"P={self.params['P']}, preimage={self.params['preimage']}, "
            f"C={self.params['C']}, gamma_mult={self.params['gamma_mult']}, "
            f"ey_floor={self.params['ey_floor']}, rho={self.params['rho']}",
        )
        stopped_at = self.sim.nextExperiment()
        logger.info("main", "Training complete")
        return stopped_at
