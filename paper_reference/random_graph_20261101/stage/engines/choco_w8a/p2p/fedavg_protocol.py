"""Gossip-FedAvg-SVM as a PeerSim CDProtocol.

Decentralised counterpart of `methods/fedavg_svm.py`. Deliberately written to
mirror `src/network_layer/peersim_python/sdca_protocol.py` step for step, so
both protocols read the same way and can be driven by the same controls.

Each node owns a data shard and runs, per cycle:
  1. drain its inbox and merge received models (age-weighted, per-node,
     order-independent -> asynchronous aggregation),
  2. n_local_steps of Pegasos sub-gradient descent over its shard (identical
     recursion to `methods/fedavg_svm.py::_pegasos_steps`),
  3. push its current primal weight to a random neighbour's inbox (gossip),
  4. record per-round metrics.

What changes relative to the server version is only step 3+1: FedAvg's server
replaced w by the sample-weighted average over *all K clients*; here each node
averages over *the neighbours it happened to hear from*. The local Pegasos step
is untouched -- it never needed a server in the first place. That single
substitution is what turns FedAvg into decentralised parallel SGD.

Pegasos has no dual variables, so there is no duality gap to report; the
`duality_gap` field is present but NaN so that observers written against
SDCAProtocol keep working.
"""

import time

import numpy as np

from src.network_layer.peersim_python.cdsim import CDProtocol
from src.network_layer.peersim_python.core import CommonState


class FedAvgProtocol(CDProtocol):
    LINKABLE_PID = 0  # protocol id of the IdleProtocol holding neighbours

    def __init__(self):
        # State is empty on the prototype; a DataInitializer fills each node's
        # shard after the network is cloned (PeerSim NodeInitializer pattern).
        self.data_ready = False
        self.gossip_k = 1
        self.n_local_steps = 100
        # Pegasos' learning-rate counter. False reproduces
        # methods/fedavg_svm.py, where t restarts at 1 on every round, so every
        # round opens with eta = 1/lambda (10000 at lambda=1e-4). True is
        # textbook Pegasos, where t runs across the whole simulation.
        # Measured on covtype, 10 nodes, 30 rounds, server aggregation:
        #     lambda=1e-4  t restarts 0.6774 (||w||=428) | t global 0.7449 (||w||=60)
        #     lambda=1e-2  t restarts 0.6515 (||w||=5.1) | t global 0.6865 (||w||=3.5)
        # Default stays False so a P2P run is directly comparable to the
        # existing server baseline; flip both together, never just one.
        self.t_global = False
        self.inbox: list = []       # received (w, age) pairs — the async mailbox
        self.metrics: list = []
        self.comm_bytes = 0
        self.start = None

    def clone(self):
        # Fresh, empty protocol per node — never deep-copy the (large) shard.
        c = FedAvgProtocol()
        c.gossip_k = self.gossip_k
        c.n_local_steps = self.n_local_steps
        c.t_global = self.t_global
        return c

    # ---- set-up (called once per node by DataInitializer) -------------------
    def set_data(self, X_csr, y, X_test, y_test, lambda_reg, t0_fraction):
        # Same signature as SDCAProtocol.set_data so the existing
        # DataInitializer can drive this protocol unchanged. Pegasos has no
        # primal-averaging burn-in, so t0_fraction is accepted and ignored.
        self.X = X_csr.tocsr()
        self.y = np.asarray(y, dtype=np.float32)
        self.X_test = X_test
        self.y_test = np.asarray(y_test, dtype=np.float32)
        self.n, self.d = self.X.shape
        self.lambda_reg = lambda_reg

        self.w = np.zeros(self.d, dtype=np.float32)
        self.age = self.n          # model age t starts at local sample count
        self.steps_done = 0        # Pegasos iteration counter (see t_global)
        # Per-node RNG drawn from the one shared PeerSim RNG, so a whole run is
        # reproducible from the single seed passed to CommonState.
        self.rng = np.random.default_rng(CommonState.r.randrange(2 ** 31))
        self.start = time.time()
        self.data_ready = True

    # ---- one cycle ----------------------------------------------------------
    def nextCycle(self, node, pid):
        if not self.data_ready:
            return
        self._merge_inbox()           # 1. asynchronous receive + aggregate
        self._local_epoch()           # 2. local Pegasos training
        self._gossip_push(node, pid)  # 3. send to random neighbour(s)
        self._record()                # 4. log per-round metrics

    # ---- (1) age-weighted merge — same rule as SDCAProtocol -----------------
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

    # ---- (2) local Pegasos steps (sparse) -----------------------------------
    def _local_epoch(self):
        """n_local_steps of Pegasos, matching methods/fedavg_svm.py exactly.

            eta_t = 1 / (lambda * t)
            w    <- (1 - eta*lambda) * w        (+ eta*y_i*x_i if margin < 1)

        where t restarts at 1 each cycle (t_global=False, matching the server
        baseline) or runs across the whole simulation (t_global=True).

        The server version rescales the whole d-vector on every step, which is
        O(d) per step and unaffordable at d=47k inside a simulation. Here w is
        kept factored as w = s * u: the shrink becomes a scalar update on s and
        only the hinge term touches memory, so a step costs O(nnz of one row).
        Mathematically identical -- verified against _pegasos_steps to 3e-08,
        i.e. float32 round-off.
        """
        if self.n == 0:
            return
        X, y = self.X, self.y
        indptr, indices, data = X.indptr, X.indices, X.data
        lam = self.lambda_reg

        u = self.w.astype(np.float64, copy=True)
        s = 1.0
        base = self.steps_done if self.t_global else 0
        for j in range(1, self.n_local_steps + 1):
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

        self.steps_done += self.n_local_steps
        self.w = (s * u).astype(np.float32)

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
            peer.getProtocol(pid).inbox.append((self.w.copy(), self.age))
            self.comm_bytes += self.d * 4  # one float32 weight vector sent

    # ---- (4) per-round metrics ----------------------------------------------
    def _record(self):
        w_np = self.w
        if self.n > 0:
            scores = self.X.dot(w_np)
            hinge = float(np.mean(np.maximum(0.0, 1.0 - self.y * scores)))
        else:
            hinge = float("nan")
        reg = float((self.lambda_reg / 2.0) * np.dot(w_np, w_np))
        self.metrics.append({
            "round":       len(self.metrics) + 1,
            "primal":      hinge + reg,
            # Pegasos is primal-only: no alpha, hence no dual and no gap.
            "dual":        float("nan"),
            "duality_gap": float("nan"),
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
