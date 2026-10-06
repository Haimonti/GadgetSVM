"""Assemble FDR-SVM on the same PeerSim engine and shards as SDCA."""
from p2p.fdr_svm_gossip import FDRSVMGossipProtocol
from src.fedavg_simulation import FedAvgSimulation
from src.peersim_python.core import Network, GeneralNode, CommonState
from src.peersim_python.idle_protocol import IdleProtocol
from src.peersim_python.observers import DataInitializer
from src.peersim_python.cdsim import CDSimulator
from src.fedavg_observers import FedAvgEvaluator


class FDRSVMSimulation(FedAvgSimulation):
    FDR_PID = 1

    def build(self):
        cfg = self.config
        CommonState.initializeRandom(cfg["SEED"])
        prototype = GeneralNode([
            IdleProtocol(),
            FDRSVMGossipProtocol(
                local_steps=cfg.get("FDR_LOCAL_STEPS", 100),
                rho=cfg.get("FDR_RHO", 1.0),
                eps_scale=cfg.get("FDR_EPS_SCALE", 1.0),
            ),
        ])
        Network.reset(cfg["NUM_WORKERS"], prototype)
        initializers = [
            self._topology(cfg["TOPOLOGY"]),
            DataInitializer(self.FDR_PID, self.worker_data, cfg["LAMBDA"],
                            cfg["GOSSIP_K"], warm_start=False),
        ]
        controls = [FedAvgEvaluator(
            self.FDR_PID, stall_tol=self.stall_tol,
            eval_every=cfg.get("EVAL_EVERY", 1),
            stop_on_threshold=cfg.get("STOP_ON_THRESHOLD", False),
        )]
        self.sim = CDSimulator(cycles=0, initializers=initializers,
                               controls=controls,
                               activation=cfg.get("ACTIVATION", "shuffle"))
        return self

    def run(self, cycles=None):
        if self.sim is None:
            self.build()
        self.sim.cycles = cycles if cycles is not None else self.config["ROUNDS"]
        self.sim.controls[0].total_cycles = self.sim.cycles
        return self.sim.nextExperiment()
