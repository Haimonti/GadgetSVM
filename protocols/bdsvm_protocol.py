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

What each node tracks
---------------------
Nodes cannot form a global sum directly, so each holds a running estimate of the
*mean* contribution and rescales by the network size M (a constant of the
problem, like the feature count):

    S_i  ~  (1/M) sum_m C_m          T_i  ~  (1/M) sum_m d_m
    solve  (M*S_i + K''_p) beta_i = M*T_i

The estimates are maintained by dynamic average consensus: when a node's own
contribution changes from C_old to C_new it injects the delta into its estimate,
then gossip-averages with neighbours. That tracks a moving input, which is
needed here because C_i is recomputed from beta_i every cycle.

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

from methods.bdsvm import _rbf, _make_preimages, _worker_contribution


class BDSVMProtocol(CDProtocol):
    LINKABLE_PID = 0  # protocol id of the IdleProtocol holding neighbours

    def __init__(self):
        # State is empty on the prototype; a DataInitializer fills each node's
        # shard after the network is cloned (PeerSim NodeInitializer pattern).
        self.data_ready = False
        self.gossip_k = 1
        self.P = 100          # budget: number of pre-image vectors
        self.C = 1.0          # SVM penalty, Eq (9)
        self.lam = 0.5        # mixing weight, Algorithm 2 step 8
        self.gamma = None     # RBF width; defaults to 1/n_features
        self.arch_seed = 0    # shared seed S generating the pre-images
        self.n_nodes = 1      # M, needed to rescale the mean back to a sum
        self.inbox: list = []
        self.metrics: list = []
        self.comm_bytes = 0
        self.start = None

    def clone(self):
        c = BDSVMProtocol()
        for a in ("gossip_k", "P", "C", "lam", "gamma", "arch_seed", "n_nodes"):
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
        if self.gamma is None:
            self.gamma = 1.0 / self.d

        # Architecture: every node regenerates the SAME P pre-images from the
        # shared seed, so the model architecture is common without any node
        # sending it (Algorithm 3, step 2).
        p = _make_preimages(self.P, self.d, self.arch_seed)
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
        self.C_own = np.zeros((self.P + 1, self.P + 1))
        self.d_own = np.zeros(self.P + 1)
        # Consensus estimates of the per-node MEAN contribution.
        self.S = np.zeros((self.P + 1, self.P + 1))
        self.T = np.zeros(self.P + 1)
        self.start = time.time()
        self.data_ready = True

    # ---- one cycle ----------------------------------------------------------
    def nextCycle(self, node, pid):
        if not self.data_ready:
            return
        self._merge_inbox()           # 1. average estimates with neighbours
        self._local_epoch()           # 2. solve, update beta, re-inject own C,d
        self._gossip_push(node, pid)  # 3. send to random neighbour(s)
        self._record()                # 4. log per-round metrics

    # ---- (1) consensus averaging of the two estimates -----------------------
    def _merge_inbox(self):
        if not self.inbox:
            return
        S, T = self.S.copy(), self.T.copy()
        for peer_S, peer_T in self.inbox:
            S += peer_S
            T += peer_T
        k = len(self.inbox) + 1
        self.S, self.T = S / k, T / k
        self.inbox = []

    # ---- (2) solve Eq (17) locally, then refresh this node's contribution ---
    def _local_epoch(self):
        M = max(self.n_nodes, 1)
        A = M * self.S + self.Kpp
        A[np.diag_indices_from(A)] += 1e-8 * max(np.trace(A), 1.0) / A.shape[0]
        try:
            beta_new = np.linalg.solve(A, M * self.T)
        except np.linalg.LinAlgError:
            beta_new = np.linalg.lstsq(A, M * self.T, rcond=None)[0]
        self.beta = self.lam * self.beta + (1.0 - self.lam) * beta_new

        if self.n == 0:
            return
        # Dynamic average consensus: inject only the CHANGE in this node's own
        # contribution, so the estimate tracks a moving input instead of
        # double-counting it every cycle.
        C_new, d_new = _worker_contribution(self.Km, self.y, self.beta, self.C)
        M = max(self.n_nodes, 1)
        self.S += (C_new - self.C_own) / M
        self.T += (d_new - self.d_own) / M
        self.C_own, self.d_own = C_new, d_new

    # ---- (3) gossip push through the Linkable interface ---------------------
    def _gossip_push(self, node, pid):
        link = node.getProtocol(self.LINKABLE_PID)
        deg = link.degree()
        if deg == 0:
            return
        # Distinct neighbours — see the note in protocols/fedavg_protocol.py.
        n_push = min(self.gossip_k, deg)
        order = list(range(deg))
        for i in range(n_push):
            j = i + CommonState.r.randrange(deg - i)   # partial Fisher-Yates
            order[i], order[j] = order[j], order[i]
            peer = link.getNeighbor(order[i])
            peer.getProtocol(pid).inbox.append((self.S.copy(), self.T.copy()))
            self.comm_bytes += (self.S.size + self.T.size) * 8   # float64

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
