"""P2P-FedAvg-SVM on the refactored gossip layer, alongside SDCA and BDSVM.

FedAvg is the simplest of the three: every node runs local Pegasos on its own
shard and then averages its weight vector with its neighbours', weighted by
shard size. There is no dual variable, no contribution table, no certificate.
That simplicity is the point of including it — it is the method the non-IID
literature shows collapsing under label skew, so it anchors the comparison.

What differs from the other two learners on this engine
--------------------------------------------------------
Payload is `{"w": w, "n": n_k}`: the primal weight plus the shard size, so
the merge can be size-weighted as FedAvg defines it. It is a dict rather than
a tuple because the engine's `_gossip_push` shallow-copies every payload with
`dict(...)`, which is how the contribution tables travel; a tuple would be
read as key/value pairs. The engine's PlainAverageAggregator is equal-weight
and takes bare arrays, so a small size-weighted aggregator lives here. Under
equal shards the two coincide; under Dirichlet they do not.

Local work per cycle is `local_steps` Pegasos updates, taken from the same
config key SDCA uses for its coordinate steps, so the two do comparable work
per cycle. Pegasos' step-size counter `t` runs across the whole simulation
(textbook), not restarting each cycle: a restart reopens every cycle at
eta = 1/lambda, which at lambda=1e-4 is a step of 10^4 and was measured
earlier to blow the model up before local training could settle.

Stopping is on the iterate stalling — the same primal rule SDCA's second
condition uses — since there is no gap to stop on. The threshold is the
observer's; the protocol only measures `rel_change`.
"""

import time

import numpy as np

from src.peersim_python.gossip_protocol import GossipProtocol
from src.peersim_python.aggregator import Aggregator


class SizeWeightedAverageAggregator(Aggregator):
    """FedAvg's merge: w <- sum_k n_k w_k / sum_k n_k over self and received.

    Payloads are `{"w": ..., "n": ...}`. A node with no data has n = 0 and
    therefore contributes nothing, which is the right outcome — averaging in a
    zero vector from an empty shard would only drag its neighbours toward zero.
    """

    def aggregate(self, current, peers):
        w_self, n_self = current["w"], current["n"]
        total = float(n_self)
        accum = np.asarray(w_self, dtype=np.float64) * n_self
        for p in peers:
            accum += np.asarray(p["w"], dtype=np.float64) * p["n"]
            total += p["n"]
        if total <= 0:
            return current
        return {"w": accum / total, "n": n_self}


class FedAvgGossipProtocol(GossipProtocol):
    """Gossip-FedAvg with a linear SVM trained by Pegasos."""

    def __init__(self, gossip_k=1, local_steps=None, t_global=True):
        super().__init__(gossip_k=gossip_k,
                         aggregator=SizeWeightedAverageAggregator())
        self.local_steps = local_steps
        self.t_global = t_global
        self.data_ready = False
        self.metrics: list = []
        self.node_id = None
        self.n_global = None
        self.steps_done = 0
        self.rel_change = float("inf")
        self.stop_cycle = None
        self.start = None

    def clone(self):
        return FedAvgGossipProtocol(gossip_k=self.gossip_k,
                                    local_steps=self.local_steps,
                                    t_global=self.t_global)

    # ---- set-up ------------------------------------------------------------
    def set_data(self, X_csr, y, X_test, y_test, lambda_reg):
        self.X = X_csr.tocsr()
        self.y = np.asarray(y, dtype=np.float64)
        self.X_test = X_test.tocsr()
        self.y_test = np.asarray(y_test, dtype=np.float64)
        self.n, self.d = self.X.shape
        self.lambda_reg = float(lambda_reg)
        self.w = np.zeros(self.d, dtype=np.float64)
        self.start = time.time()
        self.data_ready = True

    def configure_network(self, node_id, n_global, n_workers,
                          local_steps=None, step_scale=None):
        """`local_steps` is shared with SDCA's config key so the two do the
        same amount of local work per cycle. `step_scale` is SDCA's dual
        damping and has no meaning here."""
        self.node_id = int(node_id)
        self.n_global = int(n_global)
        if self.local_steps is None:
            self.local_steps = max(1, int(local_steps or self.n or 1))
        # Per-node RNG derived from the shared PeerSim stream, so a run is
        # reproducible from the single configured seed.
        from src.peersim_python.core import CommonState
        self.rng = np.random.default_rng(CommonState.r.randrange(2 ** 31))

    def warm_start(self):
        return None

    # ---- GossipProtocol hooks ---------------------------------------------
    def ready(self):
        return self.data_ready and self.node_id is not None

    def current_state(self):
        return {"w": self.w, "n": self.n}

    def set_state(self, merged):
        self.w = np.asarray(merged["w"], dtype=np.float64)

    def outgoing_payload(self):
        return {"w": self.w.copy(), "n": self.n}

    def payload_nbytes(self):
        return self.d * 8 + 8      # one float64 vector plus the shard size

    def local_update(self):
        """`local_steps` of Pegasos on the local shard.

            eta_t = 1 / (lambda * t)
            w    <- (1 - eta*lambda) * w   (+ eta * y_i * x_i  if margin < 1)

        w is kept factored as s*u so the shrink is a scalar update and only the
        hinge term touches memory: O(nnz of one row) per step instead of O(d).
        Numerically identical to methods/fedavg_svm.py's step (3e-08).
        """
        if self.n == 0:
            self.rel_change = 0.0
            return
        w_prev = self.w.copy()
        X, y, lam = self.X, self.y, self.lambda_reg
        indptr, indices, data = X.indptr, X.indices, X.data
        u = self.w.astype(np.float64, copy=True)
        s = 1.0
        base = self.steps_done if self.t_global else 0
        for j in range(1, self.local_steps + 1):
            eta = 1.0 / (lam * (base + j))
            i = int(self.rng.integers(self.n))
            st, e = indptr[i], indptr[i + 1]
            cols, vals = indices[st:e], data[st:e]
            if len(cols) == 0:
                continue
            margin = float(y[i]) * float(np.dot(s * u[cols], vals))
            s *= (1.0 - eta * lam)
            if abs(s) < 1e-12:      # exact: at t=1, eta*lambda == 1 kills s*u
                u[:] = 0.0
                s = 1.0
            if margin < 1.0:
                u[cols] += (eta * float(y[i]) / s) * vals
        self.w = s * u
        self.steps_done += self.local_steps
        denom = float(np.linalg.norm(w_prev))
        self.rel_change = (float(np.linalg.norm(self.w - w_prev)) / denom
                           if denom > 0 else float("inf"))

    # ---- evaluation --------------------------------------------------------
    def accuracy(self):
        if self.X_test.shape[0] == 0:
            return float("nan")
        s = np.asarray(self.X_test.dot(self.w)).ravel()
        return float(np.mean(np.where(s >= 0, 1.0, -1.0) == self.y_test))
