"""CentralSDCASimulation — centralized SDCA on the PeerSim engine as a 1-node network.

    sim = CentralSDCASimulation(CONFIG, worker_data)
    sim.run()
"""

from src.peersim_python.core import Network, GeneralNode, CommonState
from src.peersim_python.cdsim import CDSimulator
from src.peersim_python.logger import logger
from src.baselines.central_sdca.central_protocol import CentralSDCAProtocol
from src.baselines.central_sdca.observers import (CentralDataInitializer, CentralEvaluator,
                                                  reference_optimum)


class CentralSDCASimulation:

    CENTRAL_PID = 0      # the only protocol: no Linkable, the node has no neighbours

    def __init__(self, config, worker_data):
        self.config = config
        self.worker_data = worker_data
        self.sim = None
        self.p_star = None

    def build(self):
        cfg = self.config
        CommonState.initializeRandom(cfg["SEED"])
        Network.reset(1, GeneralNode([CentralSDCAProtocol()]))
        logger.info("network", "1 node built (protocol 0=CentralSDCA, no links)")
        self.sim = CDSimulator(
            cycles=0,
            initializers=[CentralDataInitializer(self.CENTRAL_PID, self.worker_data, cfg)],
            controls=[CentralEvaluator(
                self.CENTRAL_PID, cfg.get("CENTRAL_GAP_TOL", 1e-6),
                eval_every=cfg.get("EVAL_EVERY", 1),
                stop_on_plateau=cfg.get("STOP_ON_PLATEAU", True),
                plateau_patience=cfg.get("PLATEAU_PATIENCE", 20),
                plateau_rel_eps=cfg.get("PLATEAU_REL_EPS", 1e-3))],
            activation=cfg.get("ACTIVATION", "shuffle"),
        )
        return self

    def run(self, cycles=None):
        cfg = self.config
        cycles = cycles if cycles is not None else cfg["ROUNDS"]
        self.budget = cycles
        if self.sim is None:
            self.build()
        self.sim.cycles = cycles
        for c in self.sim.controls:
            if isinstance(c, CentralEvaluator):
                c.total_cycles = cycles
        logger.info("main", f"Training centralized SDCA — max_cycles={cycles}, "
                            f"tol={cfg.get('CENTRAL_GAP_TOL', 1e-6)}")
        stopped_at = self.sim.nextExperiment()
        p = Network.get(0).getProtocol(self.CENTRAL_PID)
        # exact optimum for every linear model's P(w) - P*; independent of this loop
        self.p_star = reference_optimum(p.X, p.y, p.lam)
        logger.info("main", f"Training complete; P* (liblinear) = {self.p_star}")
        return stopped_at

    def extra_json(self):
        return {"p_star": self.p_star,
                "p_star_solver": "liblinear (LinearSVC hinge, dual CD, tol 1e-10)",
                "network": "1 PeerSim node, no links (all NUM_WORKERS shards stacked); "
                           "num_workers/topology fields are the P2P runs' setup, unused here",
                "central_gap_tol": self.config.get("CENTRAL_GAP_TOL", 1e-6)}
