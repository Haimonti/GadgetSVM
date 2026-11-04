"""CHOCO-SGD node — `convex_code/logistic.py` (method="choco") on synchronous rounds.

Official loop body, per node i at step t (all nodes in lock-step):

    lr      = self.lr(t)                                   # base_logistic.lr
    x_plus  = lr * minus_grad(x_i, one local sample)        # SGD step
    x_plus += x_i
    x_i     = x_plus + consensus_lr * sum_j W_ij (x_hat_j - x_hat_i)   # x_hat.dot(W - I)
    q_i     = Q(x_i - x_hat_i)                              # CHOCO-Gossip compression
    x_hat_i += q_i            (and every neighbour adds q_i to its copy of x_hat_i)

Phase A computes x_i and q_i from the public copies held at the end of round
t-1; the barrier sends q_i to all neighbours; commit adds own and received q's
to the public copies. CHOCO-Gossip is exactly the last three lines — it is the
communication step of CHOCO-SGD, not a separate learner.

The one deliberate change: the official convex code is logistic regression; the
gradient is replaced by the hinge subgradient of our objective
    minus_grad = y a [y a.x < 1] - lambda x
(CHOCO-SGD's analysis covers any convex loss with bounded (sub)gradients).
"""

import numpy as np

from src.peersim_python.sync_protocol import SyncGossipProtocol


class ChocoProtocol(SyncGossipProtocol):

    def clone(self):
        return ChocoProtocol()

    def set_data(self, X, y, X_test, y_test):
        self.X = X.tocsr()
        self.y = np.asarray(y, dtype=np.float64)
        self.X_test = X_test.tocsr()
        self.y_test = np.asarray(y_test, dtype=np.float64)
        self.n, self.d = self.X.shape

    def configure(self, node_id, neighbours, weights, x0, lam, lr_type, initial_lr, tau,
                  consensus_lr, batch, compressor, rng):
        self.node_id = int(node_id)
        self.neighbours = list(neighbours)
        self.weights = dict(weights)                  # W_ij for j in N(i) u {i}
        self.lam = float(lam)
        self.lr_type = lr_type
        self.initial_lr = float(initial_lr)
        self.tau = float(tau)
        self.consensus_lr = float(consensus_lr)
        self.batch = int(batch)
        self.compressor = compressor
        self.rng = rng
        self.t = 0
        self.x = x0.copy()
        # public copies: own x_hat and one per neighbour, all equal to x0 at start
        self.x_hat = {j: x0.copy() for j in [self.node_id, *self.neighbours]}
        self.ready = True

    @property
    def w(self):
        return self.x

    def lr(self, t):
        """`BaseLogistic.lr` (t = global step; epoch-decay is not used here)."""
        if self.lr_type == "constant":
            return self.initial_lr
        if self.lr_type == "decay":
            return self.initial_lr / (self.lam * (t + self.tau))
        if self.lr_type == "bottou":
            return self.initial_lr / (1 + self.initial_lr * self.lam * t)
        raise ValueError(f"unknown CHOCO_LR_TYPE '{self.lr_type}'")

    def _minus_grad(self, x):
        idx = self.rng.choice(self.n, size=self.batch)            # np.random.choice(indices)
        g = np.zeros(self.d)
        for i in idx:
            s, e = self.X.indptr[i], self.X.indptr[i + 1]
            cols, vals = self.X.indices[s:e], self.X.data[s:e]
            yi = self.y[i]
            if yi * float(np.dot(vals, x[cols])) < 1.0:           # hinge subgradient
                g[cols] += yi * vals
        return g / self.batch - self.lam * x

    # ---- SyncGossipProtocol hooks ---------------------------------------------
    def compute_round(self):
        lr = self.lr(self.t)
        x_plus = lr * self._minus_grad(self.x) + self.x
        own = self.x_hat[self.node_id]
        mix = np.zeros(self.d)
        for j, wij in self.weights.items():
            mix += wij * (self.x_hat[j] - own)
        self._x_new = x_plus + self.consensus_lr * mix
        q, self._nbytes = self.compressor(self._x_new - own)
        return q

    def message_nbytes(self, msg):
        return self._nbytes

    def apply_round(self, msg):
        self.x = self._x_new
        self.x_hat[self.node_id] += msg
        for j in self.neighbours:
            self.x_hat[j] += self.inbox.pop(j)
        self.t += 1

    def accuracy(self):
        preds = np.where(self.X_test.dot(self.x) >= 0.0, 1.0, -1.0)
        return float(np.mean(preds == self.y_test))
