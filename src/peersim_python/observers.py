from src.peersim_python.core import Control, Network
from src.peersim_python.cdsim import CDState
from src.peersim_python.logger import logger
"""Controls for the gossip-SDCA simulation: data assignment + convergence stop.

DataInitializer plays PeerSim's NodeInitializer role — it hands each node its
data shard after the network is cloned. ConvergenceObserver is the threshold
Control: it watches every node's duality gap and stops the whole simulation once
all nodes are below the target (mirroring the user's "stop only at a threshold").
"""



class DataInitializer(Control):
    """Assign each node's SDCA protocol its own data shard (run once, at init)."""
    def __init__(self, pid, worker_data, lambda_reg, t0_fraction, gossip_k, sgd_init=False):
        self.pid = pid
        self.worker_data = worker_data
        self.lambda_reg = lambda_reg
        self.t0_fraction = t0_fraction
        self.gossip_k = gossip_k
        self.sgd_init = sgd_init

    def execute(self):
        for i, wd in enumerate(self.worker_data):
            proto = Network.get(i).getProtocol(self.pid)
            proto.gossip_k = self.gossip_k
            proto.set_data(
                wd["X_csr"], wd["y"], wd["X_test"], wd["y_test"],
                self.lambda_reg, self.t0_fraction,
            )
            if self.sgd_init:
                proto.sgd_init()   # Stage 1: local Modified-SGD warm start
            logger.info(
                "init",
                f"Node {i} loaded — {wd['n_local']} local samples"
                + ("  (+SGD warm start)" if self.sgd_init else ""),
            )
        return False


class ConvergenceObserver(Control):
    """Stop when every node's duality gap drops below `gap_threshold`.

    Runs each cycle after training. Logs mean/max gap so per-node convergence is
    visible during the simulation. Returns True (→ stop) once all nodes converge.
    """
    def __init__(self, pid, gap_threshold):
        self.pid = pid
        self.gap_threshold = gap_threshold

    def execute(self):
        gaps = []
        for i in range(Network.size()):
            m = Network.get(i).getProtocol(self.pid).metrics
            if m:
                gaps.append(m[-1]["duality_gap"])
        if not gaps:
            return False
        mean_gap = sum(gaps) / len(gaps)
        max_gap = max(gaps)
        logger.info(
            "observer",
            f"cycle={CDState.getCycle()}  mean_gap={mean_gap:.6f}  max_gap={max_gap:.6f}",
        )
        return all(g < self.gap_threshold for g in gaps)
