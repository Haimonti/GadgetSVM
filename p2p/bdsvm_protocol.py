"""P2P BDSVM — decentralised counterpart of `methods/bdsvm.py`.

Faithful to Navia-Vazquez, Diaz-Morales & Fernandez-Diaz, ACM TIST 13(6) 2022
(DOI 10.1145/3539734). See methods/bdsvm.py for the algorithm; this is the
gossip version of its aggregation step.

Why this one decentralises cleanly
----------------------------------
The aggregator's job (Eq 17) is a plain SUM of per-worker contributions:

    (sum_m C_m + K''_p) beta'' = sum_m d_m

Sums are associative and commutative, so a gossip network can compute them
without a coordinator -- the same property that makes CoCoA easy and FedAvg
hard. And because Eq (17) is formally identical to the centralized Eq (12), the
fixed point does not depend on how the data was split: non-ID robustness is
inherited from the algorithm, not engineered into the protocol.

What each node tracks: a per-origin contribution table
------------------------------------------------------
Each node keeps the freshest (C_m, d_m) it has seen from every origin it has
heard of, keyed by node id, and solves with their sum:

    table_i : origin id -> (C_m, d_m)
    solve   (sum over table_i + K''_p) beta_i = sum over table_i of d_m

Entries are tagged by origin, so re-receiving one overwrites rather than adds:
nothing is ever double counted no matter how many gossip paths a contribution
travels. Once every node's table covers every origin, the left-hand side is
*exactly* sum_m C_m and the node solves the same system the aggregator would --
the identity Eq (17) == Eq (12) is recovered exactly, not approached
asymptotically.

Two schemes were tried first and both failed, for reasons worth recording:

  * Plain neighbourhood averaging is not mass conserving. A node that averages
    its estimate with a neighbour's while the sender keeps its own inflates the
    total, so the estimate is biased and never settles.
  * Push-sum (Kempe, Dobra & Gehrke) conserves mass and is what the original
    Java code uses, but it only converges asymptotically and its weights carry
    heavy variance at low fan-out -- measured min(w) = 0.018 against a mean of
    1.0 after 30 cycles, so the M/w rescaling amplified noise by ~500x. It also
    has to chase a moving target here, since C_i is recomputed from beta_i every
    cycle. Accuracy was worse than doing nothing.

The cost of the table is memory, not bandwidth: a node stores M matrices but
still sends only a bounded number per message (`gossip_entries`, default 2 --
its own entry plus one other, to keep the epidemic spreading).

`gossip_entries` buys fidelity with bandwidth. Measured on covtype, ring, 10
nodes, 30 cycles, against a server BDSVM scoring 0.7446:

    entries=1    acc 0.6902     24.7 MB
    entries=2    acc 0.7185     48.8 MB
    entries=4    acc 0.7171     95.5 MB
    entries=10   acc 0.7311    219.6 MB   (whole table every message)

Even carrying the whole table leaves a gap to the server figure, because entries
arrive having been computed against whatever beta their origin held at the time.
That staleness is intrinsic to asynchronous gossip and does not shrink with
bandwidth; it shrinks with cycles, as the betas across the network converge.

Communication cost
------------------
A message is a (P+1)x(P+1) matrix plus a (P+1) vector -- about 40 KB at P=100,
orders of magnitude more than the weight vectors the other protocols send. The
paper's own argument is that this is repaid by converging in far fewer epochs;
the comm_bytes curve is where that trade-off becomes visible, so it is worth
plotting against the others on bytes rather than cycles.
"""

import time

import numpy as np

from src.network_layer.peersim_python.cdsim import CDProtocol
from src.network_layer.peersim_python.core import CommonState

from methods.bdsvm import (_rbf, _make_preimages, _median_gamma,
                           _worker_contribution)


class BDSVMProtocol(CDProtocol):
    LINKABLE_PID = 0  # protocol id of the IdleProtocol holding neighbours

    def __init__(self):
        # State is empty on the prototype; a DataInitializer fills each node's
        # shard after the network is cloned (PeerSim NodeInitializer pattern).
        self.data_ready = False
        self.gossip_k = 1
        self.P = 100          # budget: number of pre-image vectors
        self.C = 10.0         # SVM penalty, Eq (9); see methods/bdsvm.py
        self.lam = 0.5        # mixing weight, Algorithm 2 step 8
        self.gamma = None     # RBF width; defaults to 1/n_features
        self.arch_seed = 0      # shared seed S generating the pre-images
        self.n_nodes = 1        # M, the network size
        self.gossip_entries = 2  # table entries carried per message
        self.preimage = "uniform"  # see methods/bdsvm.py::_make_preimages
        self.inbox: list = []
        self.metrics: list = []
        self.comm_bytes = 0
        self.start = None

    def clone(self):
        c = BDSVMProtocol()
        for a in ("gossip_k", "P", "C", "lam", "gamma", "arch_seed", "n_nodes",
                  "gossip_entries", "preimage"):
            setattr(c, a, getattr(self, a))
        return c

    # ---- set-up (called once per node by DataInitializer) -------------------
    def set_data(self, X_csr, y, X_test, y_test, lambda_reg, t0_fraction):
        # Same signature as SDCAProtocol.set_data so the existing
        # DataInitializer drives this protocol unchanged. BDSVM regularises
        # through C and the budget P, so lambda_reg/t0_fraction are unused.
        self.X = X_csr.tocsr()
        self.y = np.asarray(y, dtype=np.float64)
        self.X_test = X_test.tocsr()
        self.y_test = np.asarray(y_test, dtype=np.float64)
        self.n, self.d = self.X.shape


        # Architecture: every node regenerates the SAME P pre-images from the
        # shared seed, so the model architecture is common without any node
        # sending it (Algorithm 3, step 2).
        p = _make_preimages(self.P, self.d, self.arch_seed,
                            kind=self.preimage)
        if self.gamma is None:
            # Median heuristic on the local shard. Nodes see slightly different
            # values, which is a real divergence from the server version where
            # one gamma is fixed globally -- pass --gamma explicitly for a run
            # that has to match the baseline exactly.
            self.gamma = _median_gamma(self.X, p) if self.n > 0 else 1.0
        self.p = p
        self.Kpp = np.zeros((self.P + 1, self.P + 1))
        self.Kpp[:self.P, :self.P] = _rbf(p, p, self.gamma)

        # Fixed per-node kernel matrix K''_m = [K_m | 1]  (Algorithm 3, step 3)
        if self.n > 0:
            Km = _rbf(self.X, p, self.gamma)
            self.Km = np.hstack([Km, np.ones((Km.shape[0], 1))])
        else:
            self.Km = np.zeros((0, self.P + 1))

        self.beta = np.zeros(self.P + 1)
        # origin id -> (C_m, d_m). Own entry is inserted on the first cycle,
        # once the node id is known from the Node handed to nextCycle.
        self.table = {}
        self.my_id = None
        self.start = time.time()
        self.data_ready = True

    # ---- one cycle ----------------------------------------------------------
    def nextCycle(self, node, pid):
        if not self.data_ready:
            return
        if self.my_id is None:
            self.my_id = int(node.getID())
        self._merge_inbox()           # 1. fold in entries from neighbours
        self._local_epoch()           # 2. solve, update beta, re-inject own C,d
        self._gossip_push(node, pid)  # 3. send to random neighbour(s)
        self._record()                # 4. log per-round metrics

    # ---- (1) fold received entries in, keyed by origin ----------------------
    def _merge_inbox(self):
        if not self.inbox:
            return
        for entries in self.inbox:
            for origin, C_m, d_m in entries:
                if origin != self.my_id:      # never let a stale copy of our
                    self.table[origin] = (C_m, d_m)   # own entry come back
        self.inbox = []

    # ---- (2) solve Eq (17) locally, then refresh this node's contribution ---
    def _local_epoch(self):
        C_sum = np.zeros((self.P + 1, self.P + 1))
        d_sum = np.zeros(self.P + 1)
        for C_m, d_m in self.table.values():
            C_sum += C_m
            d_sum += d_m
        A = C_sum + self.Kpp
        A[np.diag_indices_from(A)] += 1e-8 * max(np.trace(A), 1.0) / A.shape[0]
        try:
            beta_new = np.linalg.solve(A, d_sum)
        except np.linalg.LinAlgError:
            beta_new = np.linalg.lstsq(A, d_sum, rcond=None)[0]
        self.beta = self.lam * self.beta + (1.0 - self.lam) * beta_new

        if self.n == 0:
            return
        # Dynamic average consensus: inject only the CHANGE in this node's own
        # contribution, so the estimate tracks a moving input instead of
        # double-counting it every cycle.
        # Refresh our own entry against the new weights.
        self.table[self.my_id] = _worker_contribution(
            self.Km, self.y, self.beta, self.C)

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
            entries = self._entries_to_send()
            peer.getProtocol(pid).inbox.append(entries)
            for _, C_m, d_m in entries:
                self.comm_bytes += (C_m.size + d_m.size) * 8   # float64

    def _entries_to_send(self):
        """Our own entry, plus up to gossip_entries-1 others chosen at random.

        Carrying a neighbour's entry onward is what lets a contribution reach
        nodes that are not adjacent to its origin; without it the table would
        only ever cover the immediate neighbourhood.
        """
        out = []
        own = self.table.get(self.my_id)
        if own is not None:
            out.append((self.my_id, own[0], own[1]))
        others = [k for k in self.table if k != self.my_id]
        n_extra = max(self.gossip_entries - 1, 0)
        for _ in range(min(n_extra, len(others))):
            k = others.pop(CommonState.r.randrange(len(others)))
            C_m, d_m = self.table[k]
            out.append((k, C_m, d_m))
        return out

    # ---- (4) per-round metrics ----------------------------------------------
    def _record(self):
        if self.n > 0:
            e = self.y - self.Km @ self.beta
            hinge = float(np.mean(np.maximum(0.0, 1.0 - self.y * (self.Km @ self.beta))))
            primal = float(np.mean(e ** 2))
        else:
            hinge = primal = float("nan")
        self.metrics.append({
            "round":       len(self.metrics) + 1,
            "primal":      primal,          # weighted least-squares residual
            "dual":        float("nan"),
            "duality_gap": float("nan"),    # IRWLS is primal-only
            "hinge_loss":  hinge,
            "wall_time":   time.time() - self.start,
            "comm_bytes":  self.comm_bytes,
        })

    # ---- evaluation ---------------------------------------------------------
    def accuracy(self):
        if self.X_test.shape[0] == 0:
            return float("nan")
        K_te = _rbf(self.X_test, self.p, self.gamma)
        scores = np.hstack([K_te, np.ones((K_te.shape[0], 1))]) @ self.beta
        preds = np.where(scores >= 0, 1.0, -1.0)
        return float(np.mean(preds == self.y_test))
