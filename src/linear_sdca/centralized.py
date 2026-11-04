"""Centralized linear SDCA — all training data on one machine (no network).

The reference the distributed models are measured against: the same regularised
hinge-loss problem (lambda, no bias), the same closed-form SDCA coordinate step
as `SDCAProtocol._run_local_steps` (with step scale 1 — there is nothing to
damp), on the union of every worker's shard.

    P(w) = lambda/2 ||w||^2 + (1/n) sum_i max(0, 1 - y_i x_i^T w)
    D(a) = (1/n) sum_i a_i - lambda/2 ||w(a)||^2,   w(a) = X^T (y*a) / (lambda n)

One *round* is `NUM_WORKERS * SDCA_LOCAL_STEPS` coordinate updates — the total
work all distributed nodes do in one cycle — so iteration axes line up. It runs
until the gap drops below `CENTRAL_GAP_TOL` (default 1e-6), a plateau, or `ROUNDS`.
The reference optimum P* used for every linear model's suboptimality P(w) - P*
is computed separately and exactly with liblinear (`reference_optimum`), so it
never depends on how far this loop got.

Exposes the same per-node result interface the picker reads from a PeerSim
protocol (`metrics`, `accuracy()`, `comm_bytes`, `stop_cycle`), as one node.
"""

import time

import numpy as np
import scipy.sparse as sp


class CentralizedSDCA:
    """Single-machine SDCA over the stacked shards."""

    def __init__(self, config, worker_data):
        self.cfg = config
        self.X = sp.vstack([w["X_csr"] for w in worker_data]).tocsr()
        self.y = np.concatenate([np.asarray(w["y"], dtype=np.float64) for w in worker_data])
        self.X_test = worker_data[0]["X_test"].tocsr()
        self.y_test = np.asarray(worker_data[0]["y_test"], dtype=np.float64)
        self.n, self.d = self.X.shape
        self.lam = float(config["LAMBDA"])
        self.alpha = np.zeros(self.n)
        self.w = np.zeros(self.d)
        self.rng = np.random.default_rng(config["SEED"])
        self.metrics = []
        self.comm_bytes = 0
        self.stop_cycle = None

    def _steps(self, count):
        X, y, alpha, w = self.X, self.y, self.alpha, self.w
        indptr, indices, data = X.indptr, X.indices, X.data
        lam_n = self.lam * self.n
        for i in self.rng.integers(0, self.n, size=count):
            s, e = indptr[i], indptr[i + 1]
            cols, vals = indices[s:e], data[s:e]
            norm_sq = float(np.dot(vals, vals))
            if norm_sq < 1e-12:
                continue
            yi = float(y[i])
            margin = yi * float(np.dot(w[cols], vals))
            unsigned_new = min(1.0, max(0.0, alpha[i] * yi + (1.0 - margin) * lam_n / norm_sq))
            delta = yi * unsigned_new - alpha[i]
            alpha[i] += delta
            w[cols] += (delta / lam_n) * vals

    def _objectives(self):
        hinge = float(np.mean(np.maximum(0.0, 1.0 - self.y * self.X.dot(self.w))))
        reg = 0.5 * self.lam * float(np.dot(self.w, self.w))
        primal = hinge + reg
        dual = float(np.dot(self.alpha, self.y)) / self.n - reg
        return primal, dual, hinge

    def accuracy(self):
        preds = np.where(self.X_test.dot(self.w) >= 0.0, 1.0, -1.0)
        return float(np.mean(preds == self.y_test))

    def run(self, cycles=None):
        """SDCA rounds until gap < CENTRAL_GAP_TOL, a plateau, or the budget.

        The reference optimum P* does NOT depend on this loop converging: it is
        computed once with liblinear's exact dual coordinate descent (the same
        hinge-loss dual, solved in C to tolerance 1e-10), see `reference_optimum`.
        """
        cfg = self.cfg
        cycles = cycles if cycles is not None else cfg["ROUNDS"]
        self.budget = cycles
        per_round = cfg["NUM_WORKERS"] * int(cfg.get("SDCA_LOCAL_STEPS") or self.n)
        tol = cfg.get("CENTRAL_GAP_TOL", 1e-6)
        every = cfg.get("EVAL_EVERY", 1)
        patience = cfg.get("PLATEAU_PATIENCE", 20)
        rel_eps = cfg.get("PLATEAU_REL_EPS", 1e-3)
        best, stall = float("inf"), 0
        self.p_star = self.reference_optimum()
        start = time.time()
        for r in range(1, cycles + 1):
            self._steps(per_round)
            last = r == cycles
            if r % every and not last:
                continue
            primal, dual, hinge = self._objectives()
            gap = primal - dual
            if gap < best * (1.0 - rel_eps):
                best, stall = gap, 0
            else:
                stall += 1
            done = gap < tol or (cfg.get("STOP_ON_PLATEAU", True) and stall >= patience)
            self.metrics.append({
                "round": r, "wall_time": time.time() - start,
                "accuracy": self.accuracy(), "hinge_loss": hinge,
                "primal": primal, "dual": dual, "duality_gap": gap,
                "comm_bytes": 0, "stopped": done or last,
            })
            if done:
                break
        self.stop_cycle = self.metrics[-1]["round"]
        return [self]

    def reference_optimum(self):
        """P* of  lambda/2 ||w||^2 + (1/n) sum hinge  via liblinear (sklearn LinearSVC,
        hinge loss, dual coordinate descent, no intercept, C = 1/(lambda n)): the same
        problem, solved exactly in C. Falls back to None if liblinear does not converge."""
        import warnings
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.svm import LinearSVC
        C = 1.0 / (self.lam * self.n)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            try:
                svc = LinearSVC(loss="hinge", dual=True, C=C, fit_intercept=False,
                                tol=1e-10, max_iter=1_000_000).fit(self.X, self.y)
            except ConvergenceWarning:
                return None
        w = svc.coef_.ravel()
        hinge = float(np.mean(np.maximum(0.0, 1.0 - self.y * self.X.dot(w))))
        return hinge + 0.5 * self.lam * float(np.dot(w, w))

    def extra_json(self):
        return {"p_star": self.p_star, "p_star_solver": "liblinear (LinearSVC hinge, dual CD, tol 1e-10)"}
