"""Decentralised consensus ADMM — shared core of the BDSVM and FDR-SVM protocols.

`methods/bdsvm.py` and `methods/fdr_svm.py` run the *same* global-consensus
ADMM; only their local subproblem differs (a budget set vs. a Wasserstein-
inflated regulariser). So the decentralisation is written once, here, and each
subclass supplies just `_solve_local`.

Server, per round:
    w_k <- argmin  f_k(w) + (rho/2)||w - (z - u_k)||^2      every client k
    z   <- sum_k (n_k/n) (w_k + u_k)                        the server's job
    u_k <- u_k + w_k - z

P2P, per cycle at node i:
    z_i <- sum over N(i) u {i} of (n_j / sum n) (w_j + u_j)  neighbourhood only
    u_i <- u_i + w_i - z_i
    w_i <- argmin  f_i(w) + (rho/2)||w - (z_i - u_i)||^2

The single global consensus variable z becomes one local estimate z_i per node,
formed from whichever neighbours it heard from. This is Jacobi consensus-ADMM;
on a connected graph the z_i agree in the limit, so the fixed point is the one
the server version converges to. The local subproblem is untouched -- neither
the budget set nor the ambiguity radius ever needed a server, both being
functions of local data only.

What travels is (w_i + u_i) together with the shard size n_i: exactly the
quantity the server was summing, so per-round communication volume matches the
baseline.

Reported loss and accuracy use the node's own primal iterate w_i rather than the
auxiliary z_i, so every protocol in this package is scored on the same object
and the consensus error ||w_i - mean(w)|| means the same thing throughout.
An isolated node degenerates gracefully: z_i = w_i + u_i drives u_i to zero and
the method falls back to purely local training.
"""

import time

import numpy as np

from src.network_layer.peersim_python.cdsim import CDProtocol
from src.network_layer.peersim_python.core import CommonState

from p2p._prox import prox_pegasos


class ConsensusADMMProtocol(CDProtocol):
    LINKABLE_PID = 0  # protocol id of the IdleProtocol holding neighbours

    def __init__(self):
        # State is empty on the prototype; a DataInitializer fills each node's
        # shard after the network is cloned (PeerSim NodeInitializer pattern).
        self.data_ready = False
        self.gossip_k = 1
        self.n_local_steps = 100
        self.rho = 1.0
        self.inbox: list = []      # received (w+u, n) pairs — the async mailbox
        self.metrics: list = []
        self.comm_bytes = 0
        self.start = None

    def clone(self):
        # Fresh, empty protocol per node — never deep-copy the (large) shard.
        c = type(self)()
        c.gossip_k = self.gossip_k
        c.n_local_steps = self.n_local_steps
        c.rho = self.rho
        self._clone_extra(c)
        return c

    def _clone_extra(self, c):
        """Subclass hook: copy any extra hyperparameters onto the clone."""

    # ---- set-up (called once per node by DataInitializer) -------------------
    def set_data(self, X_csr, y, X_test, y_test, lambda_reg, t0_fraction):
        # Same signature as SDCAProtocol.set_data so the existing
        # DataInitializer drives this protocol unchanged. ADMM has no primal
        # averaging, so t0_fraction is accepted and ignored.
        self.X = X_csr.tocsr()
        self.y = np.asarray(y, dtype=np.float32)
        self.X_test = X_test
        self.y_test = np.asarray(y_test, dtype=np.float32)
        self.n, self.d = self.X.shape
        self.lambda_reg = lambda_reg

        self.w = np.zeros(self.d, dtype=np.float32)   # local primal iterate
        self.z = np.zeros(self.d, dtype=np.float32)   # local consensus estimate
        self.u = np.zeros(self.d, dtype=np.float32)   # scaled dual variable
        # Per-node RNG drawn from the one shared PeerSim RNG, so a whole run is
        # reproducible from the single seed passed to CommonState.
        self.rng = np.random.default_rng(CommonState.r.randrange(2 ** 31))
        self.start = time.time()
        self._setup_extra()
        self.data_ready = True

    def _setup_extra(self):
        """Subclass hook: allocate anything derived from the shard."""

    # ---- one cycle ----------------------------------------------------------
    def nextCycle(self, node, pid):
        if not self.data_ready:
            return
        self._merge_inbox()           # 1. neighbourhood consensus + dual update
        self._local_epoch()           # 2. local proximal solve
        self._gossip_push(node, pid)  # 3. send to random neighbour(s)
        self._record()                # 4. log per-round metrics

    # ---- (1) consensus step — replaces the server's aggregate ---------------
    def _merge_inbox(self):
        # Runs even on an empty mailbox: with no neighbours z_i = w_i + u_i,
        # which drives u_i to zero and reduces the node to local training.
        total = float(self.n)
        accum = (self.w + self.u).astype(np.float64) * self.n
        for peer_wu, peer_n in self.inbox:
            accum += peer_wu.astype(np.float64) * peer_n
            total += peer_n
        if total > 0:
            self.z = (accum / total).astype(np.float32)
        self.u = self.u + self.w - self.z      # dual update
        self.inbox = []

    # ---- (2) local proximal solve ------------------------------------------
    def _local_epoch(self):
        if self.n == 0:
            return
        self.w = self._solve_local(self.z - self.u)

    def _solve_local(self, v):
        """The algorithm-specific local subproblem. Subclasses must override."""
        raise NotImplementedError

    def _prox(self, X, y, v, A):
        return prox_pegasos(X, y, v, A, self.rho, self.n_local_steps, self.rng)

    # ---- (3) gossip push through the Linkable interface ---------------------
    def _gossip_push(self, node, pid):
        link = node.getProtocol(self.LINKABLE_PID)
        deg = link.degree()
        if deg == 0:
            return

        # Sample with replacement, matching SDCAProtocol exactly: a peer can
        # be drawn twice in one cycle and receive the same message twice. Kept
        # identical to Sreekar's layer so every protocol in the study gossips
        # the same way.
        for _ in range(min(self.gossip_k, deg)):
            peer = link.getNeighbor(CommonState.r.randint(0, deg - 1))
            peer.getProtocol(pid).inbox.append(
                ((self.w + self.u).astype(np.float32), self.n))
            self.comm_bytes += self.d * 4  # one float32 vector sent

    # ---- (4) per-round metrics ----------------------------------------------
    def _record(self):
        w = self.w
        if self.n > 0:
            scores = self.X.dot(w)
            hinge = float(np.mean(np.maximum(0.0, 1.0 - self.y * scores)))
        else:
            hinge = float("nan")
        reg = float((self.lambda_reg / 2.0) * np.dot(w, w))
        # Primal-only method: no dual variables, hence no duality gap. The keys
        # are still present so observers written against SDCAProtocol work.
        self.metrics.append({
            "round":       len(self.metrics) + 1,
            "primal":      hinge + reg,
            "dual":        float("nan"),
            "duality_gap": float("nan"),
            "hinge_loss":  hinge,
            "wall_time":   time.time() - self.start,
            "comm_bytes":  self.comm_bytes,
            # ADMM's own convergence signal: how far this node's iterate sits
            # from the consensus it last computed.
            "primal_residual": float(np.linalg.norm(self.w - self.z)),
        })

    # ---- evaluation ---------------------------------------------------------
    def accuracy(self):
        if self.X_test.shape[0] == 0:
            return float("nan")
        preds = np.sign(self.X_test.dot(self.w))
        preds[preds == 0] = 1.0
        return float(np.mean(preds == self.y_test))
