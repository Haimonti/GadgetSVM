"""P2P-BDSVM on the refactored gossip layer, so it and SDCA share a harness.

Both learners now sit on `GossipProtocol` and use the *same*
`VersionedContributionAggregator`: identical transport, identical merge
semantics, identical communication accounting. Anything that differs between
the two runs is the algorithm, not the plumbing.

The two methods turn out to need the same aggregation rule for the same reason.
SDCA's coordinate step is valid only while `w = X.T @ alpha / (lambda * N)`, so
its per-origin primal contribution must be re-summed rather than accumulated.
BDSVM's federated solve

    ( sum_k C_k + K_p ) beta = sum_k d_k

is likewise a sum over origins, and re-receiving an origin's entry must
overwrite rather than add. Versioning gives both the same idempotence.

An entry is `(version, payload, scalar)`. SDCA puts `(c_k, alpha_sum)` there;
BDSVM's payload is the pair of sufficient statistics `(C_k, d_k)` of its
weighted least-squares block, with the shard size in the scalar slot — the
aggregator coerces that third field with `float()`, so it has to stay a scalar.
Only the version is compared, so the aggregator itself is reused unchanged.
"""

import time

import numpy as np

from src.peersim_python.gossip_protocol import GossipProtocol
from src.peersim_python.aggregator import VersionedContributionAggregator

from methods.bdsvm import (_rbf, _make_preimages, _median_gamma,
                           _worker_contribution)


class BDSVMGossipProtocol(GossipProtocol):
    """Budget Distributed SVM (ACM TIST 13(6), 2022) as a gossip learner."""

    def __init__(self, gossip_k=1, P=100, C=30.0, rho=0.5,
                 gamma=None, gamma_mult=1.0, preimage="uniform", arch_seed=0,
                 ey_floor=1e-12):
        super().__init__(gossip_k=gossip_k,
                         aggregator=VersionedContributionAggregator())
        self.P = P                  # budget: number of pre-image vectors
        self.C = C                  # penalty in the weighting rule, Eq (9)
        self.rho = rho              # mixing weight, Algorithm 1 step 10
        self.gamma = gamma          # RBF width; None -> median heuristic
        self.gamma_mult = gamma_mult  # scales the heuristic; the heuristic is a
                                      # scale guess, and tune_bdsvm.py found 0.5x
                                      # is what keeps real-sim from oscillating
        self.preimage = preimage
        self.arch_seed = arch_seed
        self.ey_floor = ey_floor    # bound on Eq (9)'s weight; see _worker_contribution
        self.data_ready = False
        self.metrics: list = []
        self.version = 0
        self.contributions = {}     # origin -> (version, (C_k, d_k), n_k)
        self.node_id = None
        self.n_global = None
        self.start = None
        # Algorithm 2 step 9: ||beta - beta_prev|| / ||beta_prev||. The paper
        # stops on this; the evaluator reads it to latch this node's early stop.
        self.rel_change = float("inf")

    def clone(self):
        return BDSVMGossipProtocol(
            gossip_k=self.gossip_k, P=self.P, C=self.C, rho=self.rho,
            gamma=self.gamma, gamma_mult=self.gamma_mult,
            preimage=self.preimage, arch_seed=self.arch_seed,
            ey_floor=self.ey_floor)

    # ---- set-up ------------------------------------------------------------
    def set_data(self, X_csr, y, X_test, y_test, lambda_reg):
        # Matches SDCAProtocol.set_data so one DataInitializer drives both.
        # lambda_reg is accepted for that reason; BDSVM regularises through the
        # pre-image kernel and C instead, so it is recorded but not used.
        self.X = X_csr.tocsr()
        self.y = np.asarray(y, dtype=np.float64)
        self.X_test = X_test.tocsr()
        self.y_test = np.asarray(y_test, dtype=np.float64)
        self.n, self.d = self.X.shape
        self.lambda_reg = float(lambda_reg)
        self.beta = np.zeros(self.P + 1, dtype=np.float64)
        self.start = time.time()
        self.data_ready = True

    def configure_network(self, node_id, n_global, n_workers,
                          local_steps=None, step_scale=None):
        """Called by DataInitializer once the network is wired.

        local_steps and step_scale are SDCA's knobs; BDSVM solves its block in
        closed form each round and has no local step count, so they are accepted
        for signature compatibility and ignored.
        """
        self.node_id = int(node_id)
        self.n_global = int(n_global)

        # Architecture: every node regenerates the SAME P pre-images from the
        # shared seed, so it is common knowledge without being transmitted.
        p = _make_preimages(self.P, self.d, self.arch_seed, kind=self.preimage)
        self.p = p
        if self.gamma is None:
            self.gamma = (self.gamma_mult * _median_gamma(self.X, p)
                          if self.n > 0 else 1.0)
        self.Kpp = np.zeros((self.P + 1, self.P + 1))
        self.Kpp[:self.P, :self.P] = _rbf(p, p, self.gamma)
        if self.n > 0:
            Km = _rbf(self.X, p, self.gamma)
            self.Km = np.hstack([Km, np.ones((Km.shape[0], 1))])
        else:
            self.Km = np.zeros((0, self.P + 1))
        self._publish()

    def warm_start(self):
        """No-op. SDCA warm-starts its duals with a greedy pass; BDSVM's first
        round already solves the full weighted least-squares system, so there is
        nothing cheaper to precede it with."""
        return None

    # ---- GossipProtocol hooks ---------------------------------------------
    def ready(self):
        return self.data_ready and self.n_global is not None

    def current_state(self):
        return self.contributions

    def set_state(self, merged):
        self.contributions = merged

    def outgoing_payload(self):
        return self.contributions

    def payload_nbytes(self):
        # One (P+1)x(P+1) matrix plus one (P+1) vector per entry, float64.
        per = ((self.P + 1) ** 2 + (self.P + 1)) * 8
        return per * max(len(self.contributions), 1)

    def local_update(self):
        """Solve Eq (17) over the contributions known, then republish."""
        if self.n == 0:
            return
        C_sum = np.zeros((self.P + 1, self.P + 1))
        d_sum = np.zeros(self.P + 1)
        for _version, (C_k, d_k), _n_k in self.contributions.values():
            C_sum += C_k
            d_sum += d_k
        A = C_sum + self.Kpp
        # cond(A) runs ~1e8 on covtype and Eq (9)'s hard a_i = 0 threshold turns
        # tiny numerical differences into different active sets across epochs.
        A[np.diag_indices_from(A)] += 1e-8 * max(np.trace(A), 1.0) / A.shape[0]
        try:
            beta_new = np.linalg.solve(A, d_sum)
        except np.linalg.LinAlgError:
            beta_new = np.linalg.lstsq(A, d_sum, rcond=None)[0]
        beta_prev = self.beta
        self.beta = self.rho * beta_prev + (1.0 - self.rho) * beta_new
        denom = float(np.linalg.norm(beta_prev))
        self.rel_change = (float(np.linalg.norm(self.beta - beta_prev)) / denom
                           if denom > 0.0 else float("inf"))
        self._publish()

    def _publish(self):
        """Recompute this node's own (C_k, d_k) and bump its version."""
        if self.n == 0:
            return
        C_k, d_k = _worker_contribution(self.Km, self.y, self.beta, self.C,
                                        ey_floor=self.ey_floor)
        self.version += 1
        self.contributions = dict(self.contributions)
        self.contributions[self.node_id] = (self.version, (C_k, d_k), float(self.n))

    def record(self):
        """No-op — `BDSVMEvaluator` measures instead, every EVAL_EVERY cycles.

        Measuring here meant an RBF over the full test set on every node on
        every cycle, which costs more than the solve it is measuring. SDCA
        leaves this alone for the same reason, so both learners are now sampled
        by a Control on the same schedule and the curves line up cycle for cycle.
        """
        return None

    # ---- evaluation --------------------------------------------------------
    # The test-set kernel k(x_test, p_j) depends only on the pre-images and
    # gamma, both fixed once configure_network has run; beta is the only thing
    # that changes between evaluations. So the kernel is built once and kept:
    # rcv1's shared 677,399-row test set at P = 1000 is 5.4 GB per node, 54 GB
    # across ten — which fits on a 128 GB host and turns each evaluation from a
    # ~49 GFLOP sparse-times-dense product into a single matrix-vector product.
    # Above the budget the block is streamed in chunks instead; the arithmetic
    # is identical either way.
    CACHE_BUDGET_BYTES = 8 * 1024 ** 3
    CHUNK_ENTRIES = 20_000_000

    def _test_kernel(self):
        if getattr(self, "_K_te", None) is None:
            self._K_te = _rbf(self.X_test, self.p, self.gamma)
        return self._K_te

    def accuracy(self):
        n = self.X_test.shape[0]
        if n == 0:
            return float("nan")
        if n * self.P * 8 <= self.CACHE_BUDGET_BYTES:
            scores = self._test_kernel() @ self.beta[:self.P] + self.beta[self.P]
            return float(np.mean(np.where(scores >= 0, 1.0, -1.0) == self.y_test))
        step = max(1_000, self.CHUNK_ENTRIES // max(self.P, 1))
        correct = 0
        for s in range(0, n, step):
            e = min(s + step, n)
            K = _rbf(self.X_test[s:e], self.p, self.gamma)
            scores = K @ self.beta[:self.P] + self.beta[self.P]
            correct += int(np.sum(np.where(scores >= 0, 1.0, -1.0)
                                  == self.y_test[s:e]))
        return correct / n
