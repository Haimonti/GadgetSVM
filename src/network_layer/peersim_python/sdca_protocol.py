"""Gossip-SDCA as a PeerSim CDProtocol.

Each node owns a data shard and runs, per cycle:
  1. drain its inbox and merge received models (age-weighted, per-node,
     order-independent → asynchronous aggregation),
  2. one local SDCA training epoch over its shard (closed-form dual coordinate
     ascent — identical math to `src/model.py`),
  3. push its current primal weight to a random neighbour's inbox (gossip).

Weights (the primal vector w) are what travels between nodes; the dual variables
alpha stay local. This mirrors `own_network/` (SDCA model + GossipAggregator)
but as a pure-Python PeerSim protocol with no p2pfl/Lightning dependency.
"""

import time

import numpy as np

from src.network_layer.peersim_python.cdsim import CDProtocol
from src.network_layer.peersim_python.core import CommonState


class SDCAProtocol(CDProtocol):
    LINKABLE_PID = 0  # protocol id of the IdleProtocol holding neighbours

    def __init__(self):
        # State is empty on the prototype; a DataInitializer fills each node's
        # shard after the network is cloned (PeerSim NodeInitializer pattern).
        self.data_ready = False
        self.gossip_k = 1
        self.inbox: list = []       # received (w, age) pairs — the async mailbox
        self.metrics: list = []
        self.comm_bytes = 0
        self.start = None

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
        self.age = self.n          # model age t starts at local sample count
        self.start = time.time()
        self.data_ready = True

    # ---- one cycle ----------------------------------------------------------
    def nextCycle(self, node, pid):
        if not self.data_ready:
            return
        self._merge_inbox()        # 1. asynchronous receive + aggregate
        self._local_epoch()        # 2. local SDCA training
        self._gossip_push(node, pid)  # 3. send to random neighbour(s)
        self._record()             # 4. log per-round metrics

    # ---- (1) age-weighted merge — GossipAggregator Algorithm 2 --------------
    def _merge_inbox(self):
        if not self.inbox:
            return
        total_age = float(self.age)
        accum = self.w.astype(np.float64) * self.age
        for peer_w, peer_age in self.inbox:
            accum += peer_w.astype(np.float64) * peer_age
            total_age += peer_age
        self.w = (accum / total_age).astype(np.float32)
        self.age = total_age
        self.inbox = []

    # ---- (2) local SDCA epoch (sparse, closed-form) -------------------------
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

    # ---- (3) gossip push through the Linkable interface ---------------------
    def _gossip_push(self, node, pid):
        link = node.getProtocol(self.LINKABLE_PID)
        deg = link.degree()
        if deg == 0:
            return
        for _ in range(min(self.gossip_k, deg)):
            peer = link.getNeighbor(CommonState.r.randint(0, deg - 1))
            peer.getProtocol(pid).inbox.append((self.w.copy(), self.age))
            self.comm_bytes += self.d * 4  # one float32 weight vector sent

    # ---- (4) per-round metrics (primal/dual/gap on the local shard) ---------
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
        self.metrics.append({
            "round":       len(self.metrics) + 1,
            "primal":      primal,
            "dual":        dual,
            "duality_gap": gap,
            "hinge_loss":  hinge,
            "wall_time":   time.time() - self.start,
            "comm_bytes":  self.comm_bytes,
        })

    # ---- evaluation ---------------------------------------------------------
    def accuracy(self):
        w_np = (self.w_avg / self.avg_cnt) if self.avg_cnt > 0 else self.w
        preds = np.sign(self.X_test.dot(w_np))
        return float(np.mean(preds == self.y_test))
