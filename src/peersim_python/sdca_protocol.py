"""Gossip-SDCA as a tenant of the generic GossipProtocol.

The learner supplies only what is SDCA-specific — its state, its local training
step, and its payload — while the inbox, the send and the merge are inherited
from `GossipProtocol`.

The dual decomposes across workers. Worker k owns a disjoint block alpha_k of
the global dual variable, and the global primal is the sum of per-worker terms

    w = sum_k  X_k.T @ alpha_k / (lambda * n_global)          (*)

with alpha_k signed (alpha_i = y_i * a_i, a_i in [0, 1]). Each node therefore
publishes one identified, versioned *contribution* — its own term of (*) — and
holds a table `origin -> (version, contribution, alpha_sum)`. Its weight vector
is that table summed, rebuilt after every merge rather than accumulated. Two
consequences matter:

  * duplicate or out-of-order delivery cannot corrupt `w`, because a version
    either replaces an older entry or is dropped;
  * once every node holds every origin's latest entry, all nodes hold the same
    `w`, which is what consensus means here.

Two scalings are easy to get wrong and are both global: the coordinate step
divides by `lambda * n_global` (not the local shard size), and the dual
objective divides the summed alpha mass by `n_global`. Using the local `n`
solves a different problem per worker, and those problems have different optima.
"""

import time

import numpy as np

from src.peersim_python.aggregator import VersionedContributionAggregator
from src.peersim_python.core.common_state import CommonState
from src.peersim_python.gossip_protocol import GossipProtocol


class SDCAProtocol(GossipProtocol):
    """Gossip-SDCA SVM learner — payload = versioned per-origin contributions."""

    def __init__(self):
        # State is empty on the prototype; DataInitializer fills each node's
        # shard after the network is cloned (PeerSim NodeInitializer pattern).
        super().__init__(gossip_k=1, aggregator=VersionedContributionAggregator())
        self.data_ready = False
        self.metrics: list = []
        self.start = None
        self.warm_start_done = False
        self.node_id = None
        self.n_global = None
        self.n_workers = 1
        self.local_steps = 1
        self.step_scale = 1.0
        self.version = 0
        self.contributions = {}

    def clone(self):
        # Fresh, empty protocol per node — never deep-copy the (large) shard.
        c = SDCAProtocol()
        c.gossip_k = self.gossip_k
        return c

    # ---- set-up (called once per node by DataInitializer) -------------------
    def set_data(self, X_csr, y, X_test, y_test, lambda_reg):
        self.X = X_csr.tocsr()
        self.y = np.asarray(y, dtype=np.float64)
        self.X_test = X_test.tocsr()
        self.y_test = np.asarray(y_test, dtype=np.float64)
        self.n, self.d = self.X.shape
        self.lambda_reg = float(lambda_reg)
        self.alpha = np.zeros(self.n, dtype=np.float64)
        self.w = np.zeros(self.d, dtype=np.float64)
        self.start = time.time()
        self.data_ready = True

    def configure_network(self, node_id, n_global, n_workers,
                          local_steps=None, step_scale=None):
        """Hand the node the constants of the *global* objective.

        `local_steps` is how many coordinate updates one gossip cycle performs
        (None = one full pass over the shard); `step_scale` damps each dual
        step (None = the conservative CoCoA+ value 1/K).
        """
        self.node_id = int(node_id)
        self.n_global = int(n_global)
        self.n_workers = int(n_workers)
        self.local_steps = max(1, int(local_steps or self.n or 1))
        self.step_scale = float(
            (1.0 / max(self.n_workers, 1)) if step_scale is None else step_scale
        )
        if not 0.0 < self.step_scale <= 1.0:
            raise ValueError("SDCA_STEP_SCALE must be in (0, 1]")
        self._publish_contribution(bump=False)

    # ---- GossipProtocol hooks ----------------------------------------------
    def ready(self):
        return self.data_ready and self.node_id is not None

    def current_state(self):
        return self.contributions

    def outgoing_payload(self):
        """Relay everything this node knows — its own entry and every origin it
        has heard about. Forwarding others' entries is what lets a sparse
        overlay reach agreement without an all-to-all round."""
        return dict(self.contributions)

    def set_state(self, merged):
        self.contributions = merged
        self._rebuild_w()

    def payload_nbytes(self):
        # per entry: one float64 vector + an int version + a float alpha sum
        return len(self.contributions) * (self.d * 8 + 16)

    def local_update(self):
        self._run_local_steps(self.local_steps, random_order=True)
        self._publish_contribution()

    # ---- contribution bookkeeping ------------------------------------------
    def _contribution(self):
        """This node's term of (*) — its dual block mapped into primal space."""
        if not self.n_global:
            return np.zeros(self.d, dtype=np.float64)
        return np.asarray(
            self.X.T.dot(self.alpha) / (self.lambda_reg * self.n_global)
        ).ravel()

    def _publish_contribution(self, bump=True):
        if bump:
            self.version += 1
        self.contributions[self.node_id] = (
            self.version,
            self._contribution(),
            float(np.dot(self.alpha, self.y)),   # sum of unsigned a_i, for the dual
        )
        self._rebuild_w()

    def _rebuild_w(self):
        if not self.contributions:
            self.w = np.zeros(self.d, dtype=np.float64)
            return
        self.w = np.sum(
            [entry[1] for entry in self.contributions.values()],
            axis=0, dtype=np.float64,
        )

    # ---- local SDCA on this node's dual block -------------------------------
    def _run_local_steps(self, count, random_order):
        """`count` closed-form dual coordinate ascent steps on the local shard.

        Within a pass `w` is updated in place, so later coordinates already see
        earlier ones (Gauss-Seidel inside the block); the neighbours' terms in
        `w` stay fixed at whatever this node last heard, which is what makes the
        step a well-defined local subproblem.
        """
        if self.n == 0:
            return
        X, y, alpha, w = self.X, self.y, self.alpha, self.w
        indptr, indices, data = X.indptr, X.indices, X.data
        lam_n = self.lambda_reg * self.n_global
        if lam_n <= 0:
            raise ValueError("lambda and the global sample count must be positive")

        if random_order:
            order = [CommonState.r.randrange(self.n) for _ in range(count)]
        else:
            order = [i % self.n for i in range(count)]

        for i in order:
            start, end = indptr[i], indptr[i + 1]
            cols, vals = indices[start:end], data[start:end]
            norm_sq = float(np.dot(vals, vals))
            if norm_sq < 1e-12:
                continue
            yi = float(y[i])
            margin = yi * float(np.dot(w[cols], vals))
            unsigned_old = alpha[i] * yi
            unsigned_new = min(1.0, max(0.0, unsigned_old
                                        + (1.0 - margin) * lam_n / norm_sq))
            delta = self.step_scale * (yi * unsigned_new - alpha[i])
            alpha[i] += delta
            w[cols] += (delta / lam_n) * vals

    # ---- evaluation ---------------------------------------------------------
    def accuracy(self):
        preds = np.where(self.X_test.dot(self.w) >= 0.0, 1.0, -1.0)
        return float(np.mean(preds == self.y_test))

    def warm_start(self):
        """One deterministic sweep over the shard before any gossip.

        The SDCA paper warm-starts with a Modified-SGD pass; a plain in-order
        dual sweep serves the same purpose and, unlike an SGD pass, ends on a
        state that already satisfies (*), so the first contribution a node
        publishes is consistent with the alpha behind it.
        """
        if self.warm_start_done or not self.ready():
            return
        self._run_local_steps(self.n, random_order=False)
        self._publish_contribution()
        self.warm_start_done = True
