"""Gossip-CoCoA and Gossip-CoCoA+ as PeerSim CDProtocols.

Decentralised counterparts of `cocoa/cocoa.py` and `cocoa/cocoa_plus.py`,
written to the same shape as `src/network_layer/peersim_python/sdca_protocol.py`
so that one set of controls drives every protocol in this package.

Why CoCoA is the easiest of the six to decentralise
---------------------------------------------------
The dual variables alpha were already private to each worker and never left it;
only the primal increment Delta w is communicated, and the server combined those
increments by *adding* them. Addition is associative and commutative, so a
gossip network can do it without a coordinator and without caring about message
order.

That also means these two protocols do NOT use age-weighted averaging. Nodes
exchange increments, not models, so there is no age to track and no
rich-get-richer effect where a node with a large accumulated age drowns out its
neighbours' contributions. (SDCAProtocol's age-weighted merge has that failure
mode: `self.age = total_age` grows multiplicatively, so after enough cycles the
weights stop reflecting how much data a model has actually seen.)

Server -> P2P, the one substitution
-----------------------------------
    server   w <- w + scaling * sum over ALL K workers of Delta w_k
    p2p      w_i <- w_i + own scaled increment, then
             w_i <- w_i + sum over RECEIVED increments

`scaling` is where the two variants differ, exactly as in the server code:

    CoCoA   (averaging)  scaling = beta / K        -> beta / (deg + 1)
    CoCoA+  (adding)     scaling = gamma           -> gamma  (unchanged)

The neighbourhood plus self is the local stand-in for "all K workers", so K
becomes deg+1. CoCoA+'s local subproblem parameter sigma = K * gamma likewise
becomes (deg + 1) * gamma.

n_global
--------
CoCoA's primal/dual link is w = (1/(lambda*n)) * sum over ALL n samples of
alpha_i y_i x_i, so the increments are only on a common scale if every node uses
the *global* sample count in that denominator. n_global is therefore a constant
of the problem (like the feature count d), set on the prototype before cloning.
It falls back to the local shard size if left unset, which reduces each node to
solving its own local dual — valid, but no longer CoCoA.
"""

import time

import numpy as np

from src.network_layer.peersim_python.cdsim import CDProtocol
from src.network_layer.peersim_python.core import CommonState


class CoCoAProtocol(CDProtocol):
    """CoCoA — averaging aggregation, scaling = beta / (deg + 1)."""

    LINKABLE_PID = 0  # protocol id of the IdleProtocol holding neighbours
    PLUS = False      # False -> CoCoA, True -> CoCoA+

    def __init__(self):
        # State is empty on the prototype; a DataInitializer fills each node's
        # shard after the network is cloned (PeerSim NodeInitializer pattern).
        self.data_ready = False
        self.gossip_k = 1
        self.n_local_steps = 10    # H, local SDCA steps per cycle
        self.beta = 1.0            # CoCoA averaging parameter
        self.sigma_prime = 1.0     # gamma, CoCoA+ additive parameter
        self.n_global = None       # see module docstring
        self.inbox: list = []      # received Delta w vectors — the async mailbox
        self.metrics: list = []
        self.comm_bytes = 0
        self.start = None
        self.deg = 0               # refreshed each cycle; drives the scaling

    def clone(self):
        # Fresh, empty protocol per node — never deep-copy the (large) shard.
        c = type(self)()
        c.gossip_k = self.gossip_k
        c.n_local_steps = self.n_local_steps
        c.beta = self.beta
        c.sigma_prime = self.sigma_prime
        c.n_global = self.n_global
        return c

    # ---- set-up (called once per node by DataInitializer) -------------------
    def set_data(self, X_csr, y, X_test, y_test, lambda_reg, t0_fraction):
        # Same signature as SDCAProtocol.set_data so the existing
        # DataInitializer drives this protocol unchanged. CoCoA has no primal
        # averaging, so t0_fraction is accepted and ignored.
        self.X = X_csr.tocsr()
        self.y = np.asarray(y, dtype=np.float32)
        self.X_test = X_test
        self.y_test = np.asarray(y_test, dtype=np.float32)
        self.n, self.d = self.X.shape
        self.lambda_reg = lambda_reg
        if self.n_global is None:
            self.n_global = self.n

        self.alpha = np.zeros(self.n, dtype=np.float64)   # dual, stays local
        self.w = np.zeros(self.d, dtype=np.float32)       # local primal estimate
        self.delta_w = np.zeros(self.d, dtype=np.float32)  # this cycle's increment
        self.start = time.time()
        self.data_ready = True

    # ---- the CoCoA / CoCoA+ split ------------------------------------------
    def _scaling(self):
        """beta/K in the server code; the neighbourhood is the local K."""
        return self.beta / max(self.deg + 1, 1)

    def _sigma(self):
        return 1.0  # unused when PLUS is False

    # ---- one cycle ----------------------------------------------------------
    def nextCycle(self, node, pid):
        if not self.data_ready:
            return
        # Degree drives the scaling, so read it before the local solve.
        self.deg = node.getProtocol(self.LINKABLE_PID).degree()
        self._merge_inbox()           # 1. asynchronous receive + aggregate
        self._local_epoch()           # 2. local SDCA training
        self._gossip_push(node, pid)  # 3. send to random neighbour(s)
        self._record()                # 4. log per-round metrics

    # ---- (1) additive merge — no ages, order-independent --------------------
    def _merge_inbox(self):
        if not self.inbox:
            return
        accum = np.zeros(self.d, dtype=np.float64)
        for peer_dw in self.inbox:
            accum += peer_dw
        self.w = (self.w + accum).astype(np.float32)
        self.inbox = []

    # ---- (2) local SDCA epoch ----------------------------------------------
    def _local_epoch(self):
        delta_alpha, delta_w = self._local_sdca()
        sc = self._scaling()
        # _local_sdca advanced alpha in place to alpha_new; the server commits
        # only the scaled step, i.e. alpha_old + sc*delta_alpha, which equals
        # alpha_new + (sc - 1)*delta_alpha.
        self.alpha += (sc - 1.0) * delta_alpha
        self.delta_w = (delta_w * sc).astype(np.float32)
        self.w = (self.w + self.delta_w).astype(np.float32)

    def _local_sdca(self):
        """Port of `cocoa/cocoa.py::_local_sdca` for CSR shards.

        Identical update rules to the reference: projected dual gradient, alpha
        clipped to [0, 1], curvature qii = ||x||^2 (times sigma when plus), and
        the primal kept in step with the dual when plus is False. Indexing the
        CSR arrays directly keeps a step at O(nnz of one row) instead of
        materialising a dense d-vector per step.
        """
        n_local = self.n
        delta_w = np.zeros(self.d, dtype=np.float64)
        if n_local == 0:
            return np.zeros(0, dtype=np.float64), delta_w

        X, y, alpha = self.X, self.y, self.alpha
        indptr, indices, data = X.indptr, X.indices, X.data
        alpha_old = alpha.copy()
        w = self.w.astype(np.float64, copy=True)  # mutated only when not plus
        lam_n = self.lambda_reg * self.n_global
        plus, sigma = self.PLUS, self._sigma()

        for _ in range(self.n_local_steps):
            idx = int(CommonState.r.randrange(n_local))
            s, e = indptr[idx], indptr[idx + 1]
            cols, vals = indices[s:e], data[s:e]
            if len(cols) == 0:
                continue
            yi = float(y[idx])

            if plus:
                score = (float(np.dot(w[cols], vals))
                         + sigma * float(np.dot(delta_w[cols], vals)))
            else:
                score = float(np.dot(w[cols], vals))
            grad = (yi * score - 1.0) * lam_n

            a_i = alpha[idx]
            if a_i <= 0.0:
                proj_grad = min(grad, 0.0)
            elif a_i >= 1.0:
                proj_grad = max(grad, 0.0)
            else:
                proj_grad = grad
            if proj_grad == 0.0:
                continue

            x_norm_sq = float(np.dot(vals, vals))
            qii = x_norm_sq * sigma if plus else x_norm_sq
            new_alpha = min(max(a_i - grad / qii, 0.0), 1.0) if qii != 0.0 else 1.0

            coef = yi * (new_alpha - a_i) / lam_n
            if not plus:
                w[cols] += coef * vals
            delta_w[cols] += coef * vals
            alpha[idx] = new_alpha

        return alpha - alpha_old, delta_w

    # ---- (3) gossip push through the Linkable interface ---------------------
    def _gossip_push(self, node, pid):
        link = node.getProtocol(self.LINKABLE_PID)
        deg = link.degree()
        if deg == 0:
            return
        # Distinct neighbours, i.e. sampling WITHOUT replacement. SDCAProtocol
        # draws with replacement, so the same peer can receive the identical
        # message twice in one cycle -- wasted bandwidth under any aggregation
        # rule, and a double-counted increment under an additive one. Measured:
        # this does NOT by itself fix CoCoA+ on sparse graphs (see the gamma
        # note in CoCoAPlusProtocol); it is simply the correct thing to send.
        # It does differ from SDCAProtocol, so Sreekar should be told.
        n_push = min(self.gossip_k, deg)
        order = list(range(deg))
        for i in range(n_push):
            j = i + CommonState.r.randrange(deg - i)   # partial Fisher-Yates
            order[i], order[j] = order[j], order[i]
            peer = link.getNeighbor(order[i])
            peer.getProtocol(pid).inbox.append(self.delta_w.copy())
            self.comm_bytes += self.d * 4  # one float32 increment sent

    # ---- (4) per-round metrics ----------------------------------------------
    def _record(self):
        """Primal, dual and gap over this node's own shard.

        This is a per-node *local* estimate, not the global duality gap: a node
        only holds its own alphas. It still falls monotonically as the run
        converges, which is what the gap plot is for. SDCAProtocol reports the
        same local quantity, so the two are directly comparable.
        """
        w = self.w
        if self.n == 0:
            hinge = primal = dual = gap = float("nan")
        else:
            lam = self.lambda_reg
            scores = self.X.dot(w)
            hinge = float(np.mean(np.maximum(0.0, 1.0 - self.y * scores)))
            primal = hinge + float((lam / 2.0) * np.dot(w, w))
            # CoCoA keeps alpha in [0, 1] unsigned, so the sign rides on y here.
            w_alpha = self.X.T.dot(self.alpha * self.y) / (lam * self.n_global)
            dual = float(np.mean(self.alpha)) - \
                float((lam / 2.0) * np.dot(w_alpha, w_alpha))
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
        if self.X_test.shape[0] == 0:
            return float("nan")
        preds = np.sign(self.X_test.dot(self.w))
        preds[preds == 0] = 1.0
        return float(np.mean(preds == self.y_test))


class CoCoAPlusProtocol(CoCoAProtocol):
    """CoCoA+ — additive aggregation, scaling = gamma, sigma = (deg+1) * gamma.

    The only differences from CoCoA, mirroring `cocoa/cocoa_plus.py`:
      * increments are added at full strength instead of averaged down by 1/K,
      * the local solver freezes w at the start of the cycle and accounts for
        its own within-cycle progress through sigma, which is what makes the
        full-strength addition safe.

    Choosing gamma under gossip
    ---------------------------
    The server's safety condition sigma' >= gamma*K assumes all K workers start
    the round from the *same* w, so their increments are disjoint contributions.
    Gossip breaks that: a neighbour's increment was computed against its own,
    different w, so adding several at full strength re-counts progress they
    already share. Measured on covtype, ring (deg=2), 10 nodes, 30 cycles:

        gamma=1.00  k=1  acc 0.6178   |  gamma=1.00  k=2  acc 0.5022  (chance)
        gamma=0.50  k=1  acc 0.5929   |  gamma=0.50  k=2  acc 0.6113
        gamma=0.25  k=1  acc 0.5910   |  gamma=0.25  k=2  acc 0.5647

    gamma=1 is the best setting available, but only while a node aggregates
    about two contributions per cycle (its own plus one). As gossip_k grows,
    gamma has to come down roughly in step, at which point CoCoA+ converges
    toward plain CoCoA -- staleness erodes the additive variant's advantage.
    Left at 1.0 to match `cocoa/cocoa_plus.py`; lower it whenever gossip_k > 1.
    """

    NAME = "cocoa_plus"
    PLUS = True

    def _scaling(self):
        return self.sigma_prime

    def _sigma(self):
        return max(self.deg + 1, 1) * self.sigma_prime
