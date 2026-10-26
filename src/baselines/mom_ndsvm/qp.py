"""Exact solver for MoM-NDSVM's per-node dual QP, eq. (35).

    lambda_j(t+1) = argmin_{0 <= lambda <= JC}  1/2 lambda^T Q lambda - p^T lambda

``Q`` is the fixed (N x N) PSD matrix of (35) and ``p`` its iteration-dependent
linear term. The paper defines ``lambda_j(t+1)`` as the exact maximiser, so the
solver runs dual coordinate descent to a certified KKT tolerance rather than a
fixed number of steps.

* **Closed-form coordinate step.** ``Q_nn >= 1/(2 eta |B_j|) > 0``, so each
  coordinate minimiser is ``clip(lambda_n - g_n / Q_nn, 0, JC)``.
* **Persistent ``Q lambda`` cache.** ``Q`` never changes across ADMM iterations;
  only ``p`` does. Keeping ``Q lambda`` between solves makes the warm-started
  gradient ``Q lambda - p`` an O(N) update.
* **Shrinking.** Each pass visits only coordinates whose KKT violation exceeds
  the tolerance, largest first.
* **Certification.** On apparent convergence ``Q lambda`` is recomputed with one
  dense product and the KKT conditions are re-checked, so incremental round-off
  can never be reported as an optimum.

``lambda`` need not be unique when ``Q`` is singular (e.g. duplicate rows), but the
primal iterate built from it is, so callers compare primal quantities only.
"""

import numpy as np


def kkt_violation(lam, grad, upper):
    """Projected-gradient KKT violation per coordinate for ``0 <= lam <= upper``."""
    at_lower = lam <= 0.0
    at_upper = lam >= upper
    viol = np.abs(grad)
    viol[at_lower] = np.maximum(-grad[at_lower], 0.0)
    viol[at_upper] = np.maximum(grad[at_upper], 0.0)
    return viol


class BoxQPSolver:
    """Warm-started exact solver for one node's (35); keeps ``lambda`` and ``Q lambda``."""

    def __init__(self, Q, upper, tol=1e-8, max_passes=10000):
        self.Q = Q
        self.diag = np.ascontiguousarray(np.diag(Q)).copy()
        if np.any(self.diag <= 0.0):
            raise ValueError("(35) needs a strictly positive diagonal")
        self.upper = float(upper)
        self.tol = float(tol) * max(1.0, self.upper)
        self.max_passes = int(max_passes)
        n = Q.shape[0]
        self.lam = np.zeros(n)          # lambda_j(0): arbitrary per Prop. 3
        self.Qlam = np.zeros(n)
        self.last_passes = 0
        self.last_violation = 0.0
        self.converged = True

    def solve(self, p):
        """Solve (35) for linear term ``p``, warm-starting from the previous
        ``lambda``. Returns ``lambda`` (the solver's own array; copy to keep)."""
        Q, diag, upper, tol = self.Q, self.diag, self.upper, self.tol
        lam, Qlam = self.lam, self.Qlam
        passes = 0
        self.converged = False
        while True:
            viol = kkt_violation(lam, Qlam - p, upper)
            active = np.flatnonzero(viol > tol)
            if active.size == 0:
                Qlam[:] = Q @ lam                          # certify, drop drift
                viol = kkt_violation(lam, Qlam - p, upper)
                if viol.max(initial=0.0) <= tol:
                    self.converged = True
                    break
                continue
            if passes >= self.max_passes:
                break
            passes += 1
            for n in active[np.argsort(-viol[active])]:
                new = lam[n] - (Qlam[n] - p[n]) / diag[n]
                if new < 0.0:
                    new = 0.0
                elif new > upper:
                    new = upper
                delta = new - lam[n]
                if delta != 0.0:
                    lam[n] = new
                    Qlam += delta * Q[n]
        self.last_passes = passes
        self.last_violation = float(viol.max(initial=0.0))
        return lam
