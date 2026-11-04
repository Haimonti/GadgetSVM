"""CentralSDCAProtocol — single-node SDCA learner (a cycle-driven protocol).

The node holds the whole training set (all shards stacked) and, every cycle, runs
`per_round` closed-form dual coordinate ascent steps on

    P(w) = lambda/2 ||w||^2 + (1/n) sum_i max(0, 1 - y_i x_i^T w)
    D(a) = (1/n) sum_i a_i - lambda/2 ||w(a)||^2,   w(a) = X^T (y*a) / (lambda n)

with the same step as `SDCAProtocol._run_local_steps`, at step scale 1 (one node,
nothing to damp). There are no neighbours, so nothing is sent.
"""

import time

import numpy as np

from src.peersim_python.cdsim.cd_protocol import CDProtocol


class CentralSDCAProtocol(CDProtocol):

    def __init__(self):
        self.node_id = 0
        self.ready = False
        self.metrics = []
        self.comm_bytes = 0
        self.stopped = False
        self.stop_cycle = None
        self.start = None

    def clone(self):
        return CentralSDCAProtocol()

    def set_data(self, X, y, X_test, y_test, lambda_reg, per_round, seed):
        self.X = X.tocsr()
        self.y = np.asarray(y, dtype=np.float64)
        self.X_test = X_test.tocsr()
        self.y_test = np.asarray(y_test, dtype=np.float64)
        self.n, self.d = self.X.shape
        self.lam = float(lambda_reg)
        self.per_round = int(per_round)
        self.alpha = np.zeros(self.n)
        self.w = np.zeros(self.d)
        self.rng = np.random.default_rng(seed)
        self.start = time.time()
        self.ready = True

    def nextCycle(self, node, protocolID):
        if self.ready and not self.stopped:
            self._steps(self.per_round)

    def _steps(self, count):
        X, y, alpha, w = self.X, self.y, self.alpha, self.w
        indptr, indices, data = X.indptr, X.indices, X.data
        lam_n = self.lam * self.n
        for i in self.rng.integers(0, self.n, size=count):
            s, e = indptr[i], indptr[i + 1]
            cols, vals = indices[s:e], data[s:e]
            norm_sq = float(np.dot(vals, vals))
            if norm_sq < 1e-12:
                continue
            yi = float(y[i])
            margin = yi * float(np.dot(w[cols], vals))
            unsigned_new = min(1.0, max(0.0, alpha[i] * yi + (1.0 - margin) * lam_n / norm_sq))
            delta = yi * unsigned_new - alpha[i]
            alpha[i] += delta
            w[cols] += (delta / lam_n) * vals

    def objectives(self):
        hinge = float(np.mean(np.maximum(0.0, 1.0 - self.y * self.X.dot(self.w))))
        reg = 0.5 * self.lam * float(np.dot(self.w, self.w))
        primal = hinge + reg
        dual = float(np.dot(self.alpha, self.y)) / self.n - reg
        return primal, dual, hinge

    def accuracy(self):
        preds = np.where(self.X_test.dot(self.w) >= 0.0, 1.0, -1.0)
        return float(np.mean(preds == self.y_test))
