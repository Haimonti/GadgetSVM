"""Retired SVD/averaged-classifier approximation (old engine layout).

This is not FedSSL-AMC: the paper trains an I/Q CNN using contrastive triplets
and retains personalized SVMs. Current code: paper_reference.ssl.
"""

import time

import numpy as np

from src.peersim_python.cdsim import CDProtocol
from src.peersim_python.core import CommonState


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
        c = type(self)()
        c.gossip_k = self.gossip_k
        c.d_enc = self.d_enc
        c.svm_C = self.svm_C
        c.encoder_cycles = self.encoder_cycles
        return c

    # ---- set-up (called once per node by DataInitializer) -------------------
    def set_data(self, X_csr, y, X_test, y_test, lambda_reg, t0_fraction=None):
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
        self.w = np.zeros(self.d, dtype=np.float32)
        self.bias = 0.0
        self.age = float(max(self.n, 1))
        self.start = time.time()
        self.data_ready = True

    def configure_network(self, node_id, n_global, n_workers, local_steps, step_scale):
        self.node_id = node_id

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
        messages = self.inbox
        msgs = [(p[key], a) for p, a in messages if key in p]
        self.inbox = []
        if not msgs:
            return
        cur = self.V if key == "V" else self.w
        total = float(self.age)
        accum = cur.astype(np.float64) * self.age
        bias_accum = self.bias * self.age
        for peer_val, peer_age in msgs:
            accum += peer_val.astype(np.float64) * peer_age
            total += peer_age
        if key == "w":
            bias_accum += sum(p.get("bias", 0.0) * a for p, a in messages if "w" in p)
        if total <= 0:       # every contributor had zero weight
            return
        merged = (accum / total).astype(np.float32)
        if key == "V":
            self.V = merged
        else:
            self.w = merged
            self.bias = bias_accum / total
        # Fixed local sample weights avoid exponential age overflow in long runs.
        self.age = float(self.n) if key == "V" or self.trainable else 0.0

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

        w = np.zeros(self.d, dtype=np.float32)
        self.trainable = self.n > 0 and len(np.unique(self.y)) > 1
        if self.trainable:
            clf = LinearSVC(C=self.svm_C, max_iter=2000, random_state=0)
            clf.fit(self.Z, self.y)
            # Each peer freezes a different basis. Lift to the original feature
            # space before gossip so classifier coordinates mean the same thing.
            w = np.asarray(self.V.T @ clf.coef_.ravel(), dtype=np.float32)
            self.bias = float(clf.intercept_[0])
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
            payload = {"w": self.w.copy(), "bias": self.bias}
            n_bytes = self.w.size * 4 + 8

        # Sample with replacement, matching SDCAProtocol exactly: a peer can
        # be drawn twice in one cycle and receive the same message twice. Kept
        # identical to Sreekar's layer so every protocol in the study gossips
        # the same way.
        for _ in range(min(self.gossip_k, deg)):
            peer = link.getNeighbor(CommonState.r.randint(0, deg - 1))
            peer.getProtocol(pid).inbox.append((payload, self.age))
            self.comm_bytes += n_bytes

    # ---- (4) per-round metrics ----------------------------------------------
    def _record(self):
        if self.phase == 1 or self.Z is None:
            # No classifier exists yet. Log the encoder's communication cost and
            # leave the loss undefined rather than reporting a meaningless zero.
            hinge = primal = float("nan")
        else:
            scores = self.X.dot(self.w) + self.bias
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
        if self.phase != 2 or self.X_test.shape[0] == 0:
            return float("nan")
        preds = np.sign(self.X_test.dot(self.w) + self.bias)
        preds[preds == 0] = 1.0
        return float(np.mean(preds == self.y_test))
