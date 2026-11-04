"""Controls for the centralized SDCA baseline: data assignment + evaluation."""

import time

import numpy as np
import scipy.sparse as sp

from src.peersim_python.cdsim import CDState
from src.peersim_python.core import Control, Network
from src.peersim_python.logger import logger


class CentralDataInitializer(Control):
    """Stack every worker's shard onto the single node."""

    def __init__(self, pid, worker_data, cfg):
        self.pid = pid
        self.worker_data = worker_data
        self.cfg = cfg

    def execute(self):
        wd = self.worker_data
        X = sp.vstack([w["X_csr"] for w in wd]).tocsr()
        y = np.concatenate([np.asarray(w["y"], dtype=np.float64) for w in wd])
        # the work all distributed nodes do in one cycle: NUM_WORKERS x SDCA_LOCAL_STEPS,
        # or, when SDCA_LOCAL_STEPS is unset (one local pass per node), one pass over all data
        steps = self.cfg.get("SDCA_LOCAL_STEPS")
        per_round = self.cfg["NUM_WORKERS"] * int(steps) if steps else X.shape[0]
        Network.get(0).getProtocol(self.pid).set_data(
            X, y, wd[0]["X_test"], wd[0]["y_test"], self.cfg["LAMBDA"], per_round,
            self.cfg["SEED"])
        logger.info("init", f"Central node loaded — {X.shape[0]} samples (all {len(wd)} shards), "
                            f"{per_round} steps per cycle")
        return False


class CentralEvaluator(Control):
    """Duality gap of the single node every `eval_every` rounds; stops at gap < tol
    or on a plateau (round r = cycle + 1, evaluated when r % eval_every == 0)."""

    def __init__(self, pid, tol, eval_every=1, total_cycles=None, stop_on_plateau=True,
                 plateau_patience=20, plateau_rel_eps=1e-3):
        self.pid = pid
        self.tol = float(tol)
        self.eval_every = max(1, int(eval_every))
        self.total_cycles = total_cycles
        self.stop_on_plateau = stop_on_plateau
        self.plateau_patience = plateau_patience
        self.plateau_rel_eps = float(plateau_rel_eps)
        self._best, self._stall = float("inf"), 0

    def execute(self):
        p = Network.get(0).getProtocol(self.pid)
        r = CDState.getCycle() + 1
        last = self.total_cycles is not None and r >= self.total_cycles
        if r % self.eval_every and not last:
            return False
        primal, dual, hinge = p.objectives()
        gap = primal - dual
        if gap < self._best * (1.0 - self.plateau_rel_eps):
            self._best, self._stall = gap, 0
        else:
            self._stall += 1
        done = gap < self.tol or (self.stop_on_plateau and self._stall >= self.plateau_patience)
        p.metrics.append({
            "round": r, "wall_time": time.time() - p.start,
            "accuracy": p.accuracy(), "hinge_loss": hinge,
            "primal": primal, "dual": dual, "duality_gap": gap,
            "comm_bytes": 0, "stopped": done or last,
        })
        logger.info("observer", f"round={r} gap={gap:.3e} primal={primal:.6e}")
        if done or last:
            p.stopped, p.stop_cycle = True, r
        return done


def reference_optimum(X, y, lam):
    """P* of  lambda/2 ||w||^2 + (1/n) sum hinge  via liblinear (sklearn LinearSVC,
    hinge loss, dual coordinate descent, no intercept, C = 1/(lambda n)): the same
    problem, solved exactly in C. None if liblinear does not converge."""
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.svm import LinearSVC
    C = 1.0 / (lam * X.shape[0])
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        try:
            svc = LinearSVC(loss="hinge", dual=True, C=C, fit_intercept=False,
                            tol=1e-10, max_iter=1_000_000).fit(X, y)
        except ConvergenceWarning:
            return None
    w = svc.coef_.ravel()
    hinge = float(np.mean(np.maximum(0.0, 1.0 - y * X.dot(w))))
    return hinge + 0.5 * lam * float(np.dot(w, w))
