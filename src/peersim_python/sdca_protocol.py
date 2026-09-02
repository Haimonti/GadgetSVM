"""Gossip-SDCA as a tenant of the generic GossipProtocol.

The SDCA learner supplies only what is SDCA-specific — its state (the primal
weight w), its local training step (one closed-form dual coordinate-ascent
epoch), and its metrics — while the inbox, the send, and the aggregation are
inherited from `GossipProtocol`. It uses CoCoA-style communication: it gossips
the per-round INCREMENT delta-w (not the absolute weight) and the
`IncrementAggregator` ADDS neighbours' increments onto w. Because increments
shrink to zero at convergence, this preserves SDCA's primal-dual invariant
(unlike averaging absolute weights, which diverges). The dual variables alpha
stay local; only delta-w travels between nodes.
"""

import time

import numpy as np

from src.peersim_python.gossip_protocol import GossipProtocol
from src.peersim_python.aggregator import IncrementAggregator


class SDCAProtocol(GossipProtocol):
    """Gossip-SDCA SVM learner — a GossipProtocol tenant (payload = weight w)."""

    def __init__(self):
        # State is empty on the prototype; a DataInitializer fills each node's
        # shard after the network is cloned (PeerSim NodeInitializer pattern).
        # CoCoA-style: gossip the per-round increment delta-w and ADD neighbours'
        # increments (IncrementAggregator) instead of averaging absolute weights.
        super().__init__(gossip_k=1, aggregator=IncrementAggregator())
        self.data_ready = False
        self.metrics: list = []
        self.start = None
        self.sgd_init_done = False  # Stage-1 SGD warm start run yet?
        self.last_delta = None      # this node's weight change from its last epoch

    def clone(self):
        # Fresh, empty protocol per node — never deep-copy the (large) shard.
        c = SDCAProtocol()
        c.gossip_k = self.gossip_k
        return c

    # ---- set-up (called once per node by DataInitializer) -------------------
    def set_data(self, X_csr, y, X_test, y_test, lambda_reg, t0_fraction):
        self.X = X_csr.tocsr()
        self.y = np.asarray(y, dtype=np.float32)
        self.X_test = X_test
        self.y_test = np.asarray(y_test, dtype=np.float32)
        self.n, self.d = self.X.shape
        self.lambda_reg = lambda_reg
        self.t0 = max(1, int(t0_fraction * self.n))

        self.alpha = np.zeros(self.n, dtype=np.float32)
        self.w = np.zeros(self.d, dtype=np.float32)
        self.w_avg = np.zeros(self.d, dtype=np.float32)
        self.avg_cnt = 0
        self.step = 0
        self.start = time.time()
        self.data_ready = True

    # ---- GossipProtocol hooks (the only SDCA-specific communication glue) ---
    def ready(self):
        return self.data_ready

    def current_state(self):
        return self.w                       # the aggregator folds increments into w

    def outgoing_payload(self):
        # Gossip the per-round INCREMENT delta-w (what this node's local epoch
        # changed), not the absolute weight. Zero before the first epoch.
        if self.last_delta is None:
            return np.zeros(self.d, dtype=np.float32)
        return self.last_delta

    def set_state(self, merged):
        self.w = merged                     # adopt weight after folding in neighbours

    def local_update(self):
        # Track the weight change from this node's own local SDCA epoch — that
        # increment (delta-w) is what gets gossiped and additively aggregated.
        w_before = self.w.copy()
        self._local_epoch()
        self.last_delta = self.w - w_before

    def record(self):
        self._record()

    def payload_nbytes(self):
        return self.d * 4                   # one float32 weight vector

    # ---- local SDCA epoch (sparse, closed-form) -----------------------------
    def _local_epoch(self):
        X, y, w, alpha = self.X, self.y, self.w, self.alpha
        indptr, indices, data = X.indptr, X.indices, X.data
        lam_n = self.lambda_reg * self.n
        for i in range(self.n):
            s, e = indptr[i], indptr[i + 1]
            cols = indices[s:e]
            vals = data[s:e]
            xi_norm_sq = float(np.dot(vals, vals))
            if xi_norm_sq < 1e-12:
                continue
            yi = float(y[i])
            score = float(np.dot(w[cols], vals)) * yi
            denom = xi_norm_sq / lam_n
            new_alpha_yi = max(0.0, min(1.0, (1.0 - score) / denom + alpha[i] * yi))
            delta = yi * new_alpha_yi - alpha[i]
            alpha[i] += delta
            w[cols] += (delta / lam_n) * vals
            self.step += 1
            if self.step >= self.t0:
                self.w_avg += w
                self.avg_cnt += 1

    # ---- per-round metrics (primal/dual/gap on the local shard) -------------
    def _record(self):
        w_np = (self.w_avg / self.avg_cnt) if self.avg_cnt > 0 else self.w
        scores = self.X.dot(w_np)
        hinge = float(np.mean(np.maximum(0.0, 1.0 - self.y * scores)))
        reg = float((self.lambda_reg / 2.0) * np.dot(w_np, w_np))
        primal = hinge + reg
        w_alpha = self.X.T.dot(self.alpha) / (self.lambda_reg * self.n)
        dual = float(np.mean(self.alpha * self.y)) - \
            float((self.lambda_reg / 2.0) * np.dot(w_alpha, w_alpha))
        gap = primal - dual
        test_preds = np.sign(self.X_test.dot(w_np))
        accuracy = float(np.mean(test_preds == self.y_test))
        self.metrics.append({
            "round":       len(self.metrics) + 1,
            "primal":      primal,
            "dual":        dual,
            "duality_gap": gap,
            "hinge_loss":  hinge,
            "accuracy":    accuracy,
            "wall_time":   time.time() - self.start,
            "comm_bytes":  self.comm_bytes,
        })

    # ---- evaluation ---------------------------------------------------------
    def accuracy(self):
        w_np = (self.w_avg / self.avg_cnt) if self.avg_cnt > 0 else self.w
        preds = np.sign(self.X_test.dot(w_np))
        return float(np.mean(preds == self.y_test))

    # ---- Stage 1: SGD initialization (Shalev-Shwartz & Zhang 2013, Sec. 4) ---
    def sgd_init(self):
        """One Modified-SGD pass over the local shard to warm-start alpha and w.

        Procedure Modified-SGD from the SDCA paper: a single pass (t = 1..n) with
        the *growing* scaling lambda*t (an SGD-style decaying step), the hinge
        closed form, and alpha_t starting at 0:

            a_t     = clip_[0,1]( (1 - y_t x_t^T w^(t-1)) / (||x_t||^2 / (lambda*t)) )
            alpha_t = y_t * a_t
            w^(t)   = (1/(lambda*t)) * sum_{i<=t} alpha_i x_i

        Since the final step has t = n, the result satisfies the SDCA invariant
        w = (1/(lambda*n)) sum alpha_i x_i, so Stage 2 (the gossip-SDCA loop)
        continues consistently. This cures the cold-start slowness of vanilla SDCA.
        """
        if self.sgd_init_done or not self.data_ready:
            return
        X = self.X
        indptr, indices, data = X.indptr, X.indices, X.data
        y = self.y
        lam = self.lambda_reg
        A = np.zeros(self.d, dtype=np.float64)   # running sum  sum_{i<=t} alpha_i x_i
        prev_scale = 0.0                         # 1/(lambda*(t-1)); w^(0) = 0
        for t in range(1, self.n + 1):
            i = t - 1
            s, e = indptr[i], indptr[i + 1]
            cols = indices[s:e]
            vals = data[s:e]
            xi_norm_sq = float(np.dot(vals, vals))
            if xi_norm_sq >= 1e-12:
                xw = prev_scale * float(np.dot(A[cols], vals))   # x_i^T w^(t-1)
                yi = float(y[i])
                denom = xi_norm_sq / (lam * t)
                a_i = max(0.0, min(1.0, (1.0 - yi * xw) / denom))
                alpha_i = yi * a_i
                self.alpha[i] = alpha_i
                A[cols] += alpha_i * vals
            prev_scale = 1.0 / (lam * t)         # scale of w^(t) for next step
        self.w = (A / (lam * self.n)).astype(np.float32)   # w = w^(n), SDCA-consistent
        self.sgd_init_done = True
