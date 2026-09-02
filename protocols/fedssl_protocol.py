"""P2P FedSSL-AMC — decentralised counterpart of `methods/fedssl_amc.py`.

The only two-phase method in the set, so the only protocol whose nextCycle needs
a state machine rather than one repeated step.

Server version
    Phase 1  every client fits a local truncated SVD; the server averages the
             right-singular-vector matrices V_k (weighted by n_k) and
             re-orthogonalises -> one shared encoder.
    Phase 2  every client encodes Z_k = X_k V^T, fits a local LinearSVC, and the
             server FedAvgs the classifier weights.

P2P version — both aggregations become gossip, the phase boundary a cycle count.
    Phase 1 (cycle < encoder_cycles)
             each node age-weight-averages the V matrices it receives, then
             re-orthogonalises its own copy by QR. This is decentralised
             federated power iteration: repeated averaging plus
             re-orthogonalisation drives every node's encoder toward the same
             dominant subspace with no server. The payload is a d_enc x d
             matrix, by far the most expensive message in this package, which
             shows up plainly on the comm_bytes curve.
    Phase 2  the encoder freezes, each node encodes its shard once and fits its
             local SVM, and from then on only the d_enc-dimensional classifier
             is gossiped (age-weighted average, as in SDCAProtocol).

Freezing the encoder at the boundary matters: if nodes kept re-averaging V while
also training on Z, they would be learning against a moving basis.

Known limitation, measured, not a porting bug
---------------------------------------------
Under label_skew every node holds a single class, so no node can fit a local
SVM and the method cannot learn. Server version scores 0.5012, this one 0.4963,
both chance. Nodes that cannot train broadcast age 0 so their zero vectors do
not drag their neighbours down, but when *every* node is single-class there is
nothing to recover from. Decide whether to report the limitation or drop the
combination from the grid before running it.
"""

import time

import numpy as np

from src.network_layer.peersim_python.cdsim import CDProtocol
from src.network_layer.peersim_python.core import CommonState


class FedSSLAMCProtocol(CDProtocol):
    LINKABLE_PID = 0  # protocol id of the IdleProtocol holding neighbours

    def __init__(self):
        # State is empty on the prototype; a DataInitializer fills each node's
        # shard after the network is cloned (PeerSim NodeInitializer pattern).
        self.data_ready = False
        self.gossip_k = 1
        self.d_enc = 64
        self.svm_C = 1.0
        self.encoder_cycles = 10   # phase boundary
        self.inbox: list = []      # received (payload, age) — the async mailbox
        self.metrics: list = []
        self.comm_bytes = 0
        self.start = None

    def clone(self):
        c = FedSSLAMCProtocol()
        c.gossip_k = self.gossip_k
        c.d_enc = self.d_enc
        c.svm_C = self.svm_C
        c.encoder_cycles = self.encoder_cycles
        return c

    # ---- set-up (called once per node by DataInitializer) -------------------
    def set_data(self, X_csr, y, X_test, y_test, lambda_reg, t0_fraction):
        from sklearn.decomposition import TruncatedSVD

        # Same signature as SDCAProtocol.set_data so the existing
        # DataInitializer drives this protocol unchanged. lambda_reg is kept for
        # the metrics row; the classifier is regularised through svm_C instead.
        self.X = X_csr.tocsr()
        self.y = np.asarray(y, dtype=np.float32)
        self.X_test = X_test.tocsr()
        self.y_test = np.asarray(y_test, dtype=np.float32)
        self.n, self.d = self.X.shape
        self.lambda_reg = lambda_reg

        # d_enc must be a *global* constant — it depends only on d, which every
        # node knows. Deriving it from the local shard size would give nodes
        # differently-shaped encoders, which cannot be averaged at all.
        self.d_enc = max(int(min(self.d_enc, self.d - 1)), 1)
        self.phase = 1
        self.cycle = 0
        self.Z = None
        self.Z_test = None
        self.trainable = False

        # Local SSL "pre-training": a truncated SVD of this node's shard. A node
        # too small for the full rank contributes a zero-padded encoder.
        V = np.zeros((self.d_enc, self.d), dtype=np.float32)
        n_comp = min(self.d_enc, self.n - 1, self.d - 1)
        if n_comp > 0:
            svd = TruncatedSVD(n_components=n_comp, random_state=0)
            svd.fit(self.X)
            V[:n_comp] = svd.components_.astype(np.float32)
        self.V = V
        self.w = np.zeros(self.d_enc, dtype=np.float32)  # lives in encoded space
        self.age = float(max(self.n, 1))
        self.start = time.time()
        self.data_ready = True

    # ---- one cycle ----------------------------------------------------------
    def nextCycle(self, node, pid):
        if not self.data_ready:
            return
        self._merge_inbox()           # 1. asynchronous receive + aggregate
        self._local_epoch()           # 2. re-orthogonalise, or switch phase
        self._gossip_push(node, pid)  # 3. send to random neighbour(s)
        self._record()                # 4. log per-round metrics

    # ---- (1) age-weighted merge, on whichever object this phase gossips -----
    def _merge_inbox(self):
        if not self.inbox:
            return
        key = "V" if self.phase == 1 else "w"
        msgs = [(p[key], a) for p, a in self.inbox if key in p]
        self.inbox = []
        if not msgs:
            return
        cur = self.V if key == "V" else self.w
        total = float(self.age)
        accum = cur.astype(np.float64) * self.age
        for peer_val, peer_age in msgs:
            accum += peer_val.astype(np.float64) * peer_age
            total += peer_age
        if total <= 0:       # every contributor had zero weight
            return
        merged = (accum / total).astype(np.float32)
        if key == "V":
            self.V = merged
        else:
            self.w = merged
        self.age = total

    # ---- (2) phase 1 re-orthogonalises; the boundary freezes the encoder ----
    def _local_epoch(self):
        self.cycle += 1
        if self.phase != 1:
            return
        # QR on V^T restores the orthonormality that averaging destroys — the
        # re-orthogonalisation step of federated power iteration.
        Q, _ = np.linalg.qr(self.V.T)
        self.V = np.ascontiguousarray(Q.T[: self.d_enc], dtype=np.float32)
        if self.cycle >= self.encoder_cycles:
            self._enter_phase_2()

    def _enter_phase_2(self):
        """Freeze the encoder, encode the shard once, fit the local SVM."""
        from sklearn.svm import LinearSVC

        self.phase = 2
        self.Z = np.asarray(self.X.dot(self.V.T), dtype=np.float64)
        self.Z_test = np.asarray(self.X_test.dot(self.V.T), dtype=np.float64)

        w = np.zeros(self.d_enc, dtype=np.float32)
        self.trainable = self.n > 0 and len(np.unique(self.y)) > 1
        if self.trainable:
            clf = LinearSVC(C=self.svm_C, max_iter=2000)
            clf.fit(self.Z, self.y)
            w = clf.coef_.ravel().astype(np.float32)
        self.w = w
        # Age 0 for a node that could not fit a classifier: under a skewed split
        # a single-class node has nothing to contribute, and age-weighting then
        # lets it receive its neighbours' model instead of averaging their model
        # with its own zeros.
        self.age = float(self.n) if self.trainable else 0.0

    # ---- (3) gossip push through the Linkable interface ---------------------
    def _gossip_push(self, node, pid):
        link = node.getProtocol(self.LINKABLE_PID)
        deg = link.degree()
        if deg == 0:
            return
        if self.phase == 1:
            payload = {"V": self.V.copy()}
            n_bytes = self.V.size * 4
        else:
            payload = {"w": self.w.copy()}
            n_bytes = self.w.size * 4
        # Distinct neighbours — see the note in protocols/fedavg_protocol.py.
        n_push = min(self.gossip_k, deg)
        order = list(range(deg))
        for i in range(n_push):
            j = i + CommonState.r.randrange(deg - i)   # partial Fisher-Yates
            order[i], order[j] = order[j], order[i]
            peer = link.getNeighbor(order[i])
            peer.getProtocol(pid).inbox.append((payload, self.age))
            self.comm_bytes += n_bytes

    # ---- (4) per-round metrics ----------------------------------------------
    def _record(self):
        if self.phase == 1 or self.Z is None:
            # No classifier exists yet. Log the encoder's communication cost and
            # leave the loss undefined rather than reporting a meaningless zero.
            hinge = primal = float("nan")
        else:
            scores = self.Z @ self.w
            hinge = float(np.mean(np.maximum(0.0, 1.0 - self.y * scores)))
            primal = hinge + float((self.lambda_reg / 2.0) * np.dot(self.w, self.w))
        self.metrics.append({
            "round":       len(self.metrics) + 1,
            "primal":      primal,
            "dual":        float("nan"),
            "duality_gap": float("nan"),   # no dual variables in either phase
            "hinge_loss":  hinge,
            "wall_time":   time.time() - self.start,
            "comm_bytes":  self.comm_bytes,
            "phase":       self.phase,
        })

    # ---- evaluation ---------------------------------------------------------
    def accuracy(self):
        if self.phase != 2 or self.Z_test is None or self.Z_test.shape[0] == 0:
            return float("nan")
        preds = np.sign(self.Z_test @ self.w)
        preds[preds == 0] = 1.0
        return float(np.mean(preds == self.y_test))
