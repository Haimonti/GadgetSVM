"""Tabular FedSSL adaptation: local SVD, shared encoder, local SVM, gossip.

This ports the repository's SVD-based FedSSL approximation, not a reproduction
of neural SSL-AMC. Immutable per-origin tables make both aggregations
idempotent. All peers freeze the same encoder before averaging classifiers;
averaging classifiers expressed in different local bases is invalid.
"""
import time
import warnings

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.exceptions import ConvergenceWarning
from sklearn.svm import LinearSVC

from src.peersim_python.aggregator import Aggregator
from src.peersim_python.gossip_protocol import GossipProtocol


class FedSSLTableAggregator(Aggregator):
    def aggregate(self, current, peers):
        tables = {key: dict(current[key]) for key in ('encoders', 'classifiers')}
        for message in peers:
            for key in tables:
                tables[key].update(message[key])
        return tables


class FedSSLGossipProtocol(GossipProtocol):
    def __init__(self, gossip_k=3, d_enc=64, encoder_rounds=10, seed=42):
        super().__init__(gossip_k, FedSSLTableAggregator())
        self.d_enc = int(d_enc)
        self.encoder_rounds = int(encoder_rounds)
        self.seed = int(seed)
        self.data_ready = False
        self.metrics = []
        self.node_id = None
        self.stop_cycle = None
        self.stop_reason = None
        self.phase = 1
        self.rounds = 0
        self.encoders = {}
        self.classifiers = {}
        self.local_solver_converged = True

    def clone(self):
        return type(self)(self.gossip_k, self.d_enc, self.encoder_rounds, self.seed)

    def set_data(self, X_csr, y, X_test, y_test, lambda_reg):
        self.X = X_csr.tocsr()
        self.y = np.asarray(y, dtype=np.float64)
        self.X_test = X_test.tocsr()
        self.y_test = np.asarray(y_test, dtype=np.float64)
        self.n, self.d = self.X.shape
        self.lambda_reg = float(lambda_reg)
        self.d_enc = max(1, min(self.d_enc, self.d))
        self.w = np.zeros(self.d_enc)
        self.start = time.time()
        self.data_ready = True

    def configure_network(self, node_id, n_global, n_workers,
                          local_steps=None, step_scale=None):
        self.node_id, self.n_global, self.n_workers = node_id, n_global, n_workers
        V = np.zeros((self.d_enc, self.d), dtype=np.float32)
        rank = min(self.d_enc, self.n - 1, self.d - 1)
        if rank > 0:
            svd = TruncatedSVD(n_components=rank, random_state=self.seed)
            svd.fit(self.X)
            V[:rank] = svd.components_
        self.encoders = {node_id: (V, self.n)}

    def warm_start(self):
        pass

    def ready(self):
        return self.data_ready and self.node_id is not None

    def current_state(self):
        return {'encoders': self.encoders, 'classifiers': self.classifiers}

    def set_state(self, merged):
        self.encoders = merged['encoders']
        self.classifiers = merged['classifiers']

    def outgoing_payload(self):
        # Entries are immutable; snapshot the dictionaries before delivery.
        return {key: dict(value) for key, value in self.current_state().items()}

    def payload_nbytes(self):
        # Array data + int64 origin and int64 shard/aggregation weight.
        return sum(array.nbytes + 16 for table in
                   (self.encoders, self.classifiers) for array, _ in table.values())

    def local_update(self):
        self.rounds += 1
        if (self.phase == 1 and self.rounds >= self.encoder_rounds
                and len(self.encoders) == self.n_workers):
            # Align SVD coordinate systems before averaging. Sorted origins
            # and a common reference give byte-identical encoders at all peers.
            ids = sorted(self.encoders)
            reference = self.encoders[ids[0]][0].astype(np.float64)
            accum = np.zeros_like(reference)
            for origin in ids:
                basis, count = self.encoders[origin]
                basis = basis.astype(np.float64)
                u, _, vt = np.linalg.svd(reference @ basis.T, full_matrices=False)
                accum += count * ((u @ vt) @ basis)
            q, _ = np.linalg.qr((accum / self.n_global).T)
            self.V = np.ascontiguousarray(q.T[:self.d_enc])
            self.phase = 2
            w = np.zeros(self.d_enc)
            trainable = self.n > 0 and np.unique(self.y).size == 2
            if trainable:
                # Match mean hinge + lambda/2 ||w||^2 in encoded space.
                clf = LinearSVC(C=1 / (self.n * self.lambda_reg), loss='hinge',
                                 dual=True, fit_intercept=False, tol=1e-4,
                                 max_iter=1000000, random_state=self.seed)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always', ConvergenceWarning)
                    clf.fit(self.X @ self.V.T, self.y)
                self.local_solver_converged = not any(
                    issubclass(item.category, ConvergenceWarning) for item in caught)
                self.local_solver_iterations = int(clf.n_iter_)
                w = clf.coef_[0].copy()
            self.classifiers = dict(self.classifiers)
            self.classifiers[self.node_id] = (w, self.n if trainable else 0)
        if self.phase == 2:
            total = sum(count for _, count in self.classifiers.values())
            if total:
                self.w = sum(w * count for _, (w, count) in
                             sorted(self.classifiers.items())) / total

    @property
    def convergence_ready(self):
        return self.phase == 2 and len(self.classifiers) == self.n_workers

    def accuracy(self):
        if self.phase != 2:
            return float('nan')
        scores = self.X_test @ (self.V.T @ self.w)
        return float(np.mean(np.where(scores >= 0, 1., -1.) == self.y_test))
