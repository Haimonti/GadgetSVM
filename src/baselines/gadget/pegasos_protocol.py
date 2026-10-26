"""GADGET SVM as a tenant of the generic GossipProtocol.

The learner supplies only what is Pegasos-specific --- its weight vector, its
local sub-gradient step, and its payload --- while the inbox, the send and the
merge are inherited from `GossipProtocol`. This mirrors `SDCAProtocol`, so both
learners run on the identical gossip transport.

Model (unchanged from the GADGET / SPegasosGadget source):

    pick a random example i, with step counter t and learning rate lambda
        eta    = 1 / (lambda * t)
        scale  = 1 - 1/t
        w      <- scale * w                     # regularisation shrink
        if y_i <w, x_i> < 1:                     # hinge sub-gradient
            w  <- w + eta * y_i * x_i
        t      <- t + 1

Nodes gossip the *absolute* weight vector w and combine it by plain averaging
(GADGET's ``pushsum1``: a pair replaces both weights with their mean). There is
no dual variable and no duality gap here --- that is exactly the axis on which
this baseline differs from the gossip-SDCA learner.
"""

import time

import numpy as np

from src.peersim_python.core.common_state import CommonState
from src.peersim_python.gossip_protocol import GossipProtocol
from src.peersim_python.aggregator import PlainAverageAggregator


class PegasosProtocol(GossipProtocol):
    """GADGET SVM learner — payload = the weight vector w, merged by averaging."""

    def __init__(self):
        # State is empty on the prototype; DataInitializer fills each node's
        # shard after the network is cloned (PeerSim NodeInitializer pattern).
        super().__init__(gossip_k=1, aggregator=PlainAverageAggregator())
        self.data_ready = False
        self.metrics: list = []
        self.start = None
        self.node_id = None
        self.n_global = None
        self.n_workers = 1
        self.local_steps = 1
        self.t = 1                      # Pegasos step counter (global across cycles)
        # GADGET objective-change convergence bookkeeping (filled by the observer)
        self.prev_obj = None
        self.converge_count = 0
        self.stop_cycle = None

    def clone(self):
        # Fresh, empty protocol per node — never deep-copy the (large) shard.
        c = PegasosProtocol()
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
        self.w = np.zeros(self.d, dtype=np.float64)
        self.start = time.time()
        self.data_ready = True

    def configure_network(self, node_id, n_global, n_workers, local_steps=None):
        """Hand the node the constants of the run.

        `local_steps` is how many Pegasos sub-gradient updates one gossip cycle
        performs (None = one full pass over the shard); reusing the SDCA
        per-cycle budget keeps the two learners on equal compute per round.
        """
        self.node_id = int(node_id)
        self.n_global = int(n_global)
        self.n_workers = int(n_workers)
        self.local_steps = max(1, int(local_steps or self.n or 1))
        self.t = 1

    # ---- GossipProtocol hooks ----------------------------------------------
    def ready(self):
        return self.data_ready and self.node_id is not None

    def current_state(self):
        return self.w

    def set_state(self, merged):
        self.w = np.asarray(merged, dtype=np.float64)

    def outgoing_payload(self):
        # Model averaging: gossip the absolute weight vector.
        return self.w

    def _copy_payload(self, payload):
        # The payload is a plain weight vector, not a dict — copy as an array.
        return payload.copy()

    def payload_nbytes(self):
        # one float64 weight vector of length d
        return self.d * 8

    def local_update(self):
        self._pegasos_steps(self.local_steps)

    # ---- the Pegasos sub-gradient solver (GADGET's local model) -------------
    def _pegasos_steps(self, count):
        """`count` Pegasos sub-gradient steps on the local shard.

        Each step picks a random example, shrinks w by (1 - 1/t), and if the
        example violates the margin adds eta * y_i * x_i with eta = 1/(lambda t).
        `t` is the global step counter, so the learning rate decays as training
        proceeds, exactly as in the GADGET source.
        """
        if self.n == 0:
            return
        X, y, w = self.X, self.y, self.w
        indptr, indices, data = X.indptr, X.indices, X.data
        lam = self.lambda_reg
        for _ in range(count):
            i = CommonState.r.randrange(self.n)
            start, end = indptr[i], indptr[i + 1]
            cols, vals = indices[start:end], data[start:end]
            yi = float(y[i])
            eta = 1.0 / (lam * self.t)
            scale = 1.0 - 1.0 / self.t
            w *= scale
            margin = yi * float(np.dot(w[cols], vals))
            if margin < 1.0:
                w[cols] += (eta * yi) * vals
            self.t += 1

    # ---- evaluation ---------------------------------------------------------
    def accuracy(self):
        preds = np.where(self.X_test.dot(self.w) >= 0.0, 1.0, -1.0)
        return float(np.mean(preds == self.y_test))
