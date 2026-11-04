"""CHOCO-SGD data/graph initialiser."""

import time

import numpy as np

from src.peersim_python.core import Control, Network
from src.peersim_python.compression import Compressor
from src.peersim_python.mixing import adjacency_from_network, metropolis_weights
from src.peersim_python.logger import logger

INIT_WEIGHT_STD = 0.01      # convex_code/constants.py


class ChocoDataInitializer(Control):
    """Shards, W rows, and CHOCO's hyper-parameters.

    Initialisation as `LogisticDecentralizedSGD.fit`: one x0 ~ N(0, 0.01^2) tiled
    to every node, and x_hat = x0 (so no initial exchange is needed).
    Defaults follow `experiment_epsilon_final.py`: lr_type="decay",
    initial_lr=0.1, tau=d; the regulariser is our LAMBDA (they use 1/n).
    """

    def __init__(self, pid, linkable_pid, worker_data, cfg):
        self.pid = pid
        self.linkable_pid = linkable_pid
        self.worker_data = worker_data
        self.cfg = cfg
        self.beta = None

    def execute(self):
        cfg = self.cfg
        adj = adjacency_from_network(self.linkable_pid)
        W, self.beta = metropolis_weights(adj)
        d = self.worker_data[0]["X_csr"].shape[1]
        x0 = np.random.default_rng(cfg["SEED"]).normal(0.0, INIT_WEIGHT_STD, size=d)
        tau = cfg.get("CHOCO_TAU") or d
        spec = cfg.get("CHOCO_COMPRESSOR", "full")
        for k, shard in enumerate(self.worker_data):
            p = Network.get(k).getProtocol(self.pid)
            p.set_data(shard["X_csr"], shard["y"], shard["X_test"], shard["y_test"])
            weights = {k: W[k, k], **{j: W[k, j] for j in adj[k]}}
            p.configure(k, adj[k], weights, x0, cfg["LAMBDA"],
                        lr_type=cfg.get("CHOCO_LR_TYPE", "decay"),
                        initial_lr=cfg.get("CHOCO_INITIAL_LR", 0.1), tau=tau,
                        consensus_lr=cfg.get("CHOCO_CONSENSUS_LR", 1.0),
                        batch=cfg.get("CHOCO_BATCH", 1),
                        compressor=Compressor(spec, np.random.default_rng([cfg["SEED"], k, 3])),
                        rng=np.random.default_rng([cfg["SEED"], k, 2]))
            p.start = time.time()
        logger.info("choco", f"{len(self.worker_data)} nodes, beta(W)={self.beta:.4f}, "
                             f"compressor={spec}, consensus_lr={cfg.get('CHOCO_CONSENSUS_LR', 1.0)}, "
                             f"lr={cfg.get('CHOCO_LR_TYPE', 'decay')}({cfg.get('CHOCO_INITIAL_LR', 0.1)}, tau={tau})")
        return False
