"""COLA observers: data/graph initialiser and COLA-specific evaluator fields."""

import time

import numpy as np

from src.peersim_python.core import Control, Network
from src.peersim_python.compression import Compressor
from src.peersim_python.mixing import adjacency_from_network, metropolis_weights
from src.peersim_python.logger import logger


class ColaDataInitializer(Control):
    """Hand each node its shard, its row of W, and COLA's constants.

    C = 1/(lambda * n_global) makes COLA's LinearSVM the same problem as ours.
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
        n_global = sum(w["n_local"] for w in self.worker_data)
        C = 1.0 / (cfg["LAMBDA"] * n_global)
        K = len(self.worker_data)
        for k, shard in enumerate(self.worker_data):
            p = Network.get(k).getProtocol(self.pid)
            p.set_data(shard["X_csr"], shard["y"], shard["X_test"], shard["y_test"])
            weights = {k: W[k, k], **{j: W[k, j] for j in adj[k]}}
            p.configure(k, adj[k], weights, K, C,
                        gamma=cfg.get("COLA_GAMMA", 1.0),
                        local_iters=cfg.get("COLA_LOCAL_ITERS", 1.0),
                        theta=cfg.get("COLA_THETA", 0.0),
                        random_state=cfg["SEED"],
                        compressor=Compressor(cfg.get("COMPRESSOR", "full"),
                                              np.random.default_rng([cfg["SEED"], k, 3])))
            p.initial_exchange()
            p.start = time.time()
        logger.info("cola", f"{K} nodes, C={C:.4g}, beta(W)={self.beta:.4f}, "
                            f"gamma={cfg.get('COLA_GAMMA', 1.0)}, "
                            f"local_iters={cfg.get('COLA_LOCAL_ITERS', 1.0)}")
        return False


def cola_dual(lam):
    """Global dual in our scaling from the true alphas (COLA's xk) of all nodes."""
    def fn(protos):
        n = sum(p.n for p in protos)
        v_true = np.sum([p.Akxk for p in protos], axis=0)        # A x
        w_true = v_true / (lam * n)
        a_sum = sum(float(np.dot(p.xk, p.y)) for p in protos)    # sum of unsigned a_i
        return a_sum / n - 0.5 * lam * float(np.dot(w_true, w_true))
    return fn


def cola_extras(protos_getter):
    """COLA's own monitor quantities: cv2 = ||v_k - A x||^2 and the local solver gap."""
    def fn(p):
        v_true = np.sum([q.Akxk for q in protos_getter()], axis=0)
        return {"cv2": float(np.sum((p.v - v_true) ** 2)),
                "local_gap": p.solver.last_gap,
                "local_iters_done": p.solver.last_iters}
    return fn
