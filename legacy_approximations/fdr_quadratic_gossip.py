"""PeerSim gossip FDR-SVM using the original FDR local proximal update.

Each node maintains a local primal vector w, scaled dual u, and neighbourhood
consensus estimate z. The ambiguity radius eps_i = eps_scale/sqrt(n_i) is local.
The proximal solver is the same factored sparse Pegasos recursion used by the
earlier P2P implementation, with curvature lambda + eps_i + rho.
"""
import time

import numpy as np

from p2p._prox import prox_pegasos
from src.peersim_python.cdsim.cd_protocol import CDProtocol
from src.peersim_python.core.common_state import CommonState


class FDRSVMGossipProtocol(CDProtocol):
    LINKABLE_PID = 0

    def __init__(self, local_steps=100, rho=1.0, eps_scale=1.0):
        self.local_steps = int(local_steps)
        self.rho = float(rho)
        self.eps_scale = float(eps_scale)
        self.inbox = []
        self.metrics = []
        self.comm_bytes = 0
        self.data_ready = False
        self.stopped = False
        self.stop_cycle = None
        self.stop_reason = None

    def clone(self):
        return FDRSVMGossipProtocol(self.local_steps, self.rho, self.eps_scale)

    def set_data(self, X_csr, y, X_test, y_test, lambda_reg):
        self.X = X_csr.tocsr()
        self.y = np.asarray(y, dtype=np.float64)
        self.X_test = X_test.tocsr()
        self.y_test = np.asarray(y_test, dtype=np.float64)
        self.n, self.d = self.X.shape
        self.lambda_reg = float(lambda_reg)
        self.eps = self.eps_scale / max(np.sqrt(self.n), 1.0)
        self.w = np.zeros(self.d, dtype=np.float32)
        self.z = np.zeros(self.d, dtype=np.float32)
        self.u = np.zeros(self.d, dtype=np.float32)
        self.rng = np.random.default_rng(CommonState.r.randrange(2 ** 31))
        self.start = time.time()
        self.data_ready = True

    def configure_network(self, node_id, n_global, n_workers,
                          local_steps=None, step_scale=None):
        self.node_id = int(node_id)
        self.n_global = int(n_global)

    def warm_start(self):
        pass

    def nextCycle(self, node, pid):
        if not self.data_ready or self.stopped:
            return
        total = float(self.n)
        accum = (self.w.astype(np.float64) + self.u) * self.n
        for peer_wu, peer_n in self.inbox:
            accum += peer_wu * peer_n
            total += peer_n
        if total > 0:
            self.z = (accum / total).astype(np.float32)
        self.u += self.w - self.z
        self.inbox.clear()
        if self.n:
            centre = self.z - self.u
            self.w = prox_pegasos(
                self.X, self.y, centre, self.lambda_reg + self.eps + self.rho,
                self.rho, self.local_steps, self.rng,
            )
        link = node.getProtocol(self.LINKABLE_PID)
        payload = (self.w + self.u).copy()
        for j in range(link.degree()):
            peer = link.getNeighbor(j).getProtocol(pid)
            if not peer.stopped:
                peer.inbox.append((payload, self.n))
                self.comm_bytes += payload.nbytes + 8

    def accuracy(self):
        if not self.y_test.size:
            return float("nan")
        scores = np.asarray(self.X_test.dot(self.w)).ravel()
        return float(np.mean(np.where(scores >= 0, 1.0, -1.0) == self.y_test))
