"""Sparse proximal Pegasos helper used by historical quadratic approximations.

This is not the FDR paper local solver. Its quadratic recursion is retained
for legacy code; the exact robust constraints are in paper_reference.fdr.
The step counter restarts on every call, so finite local solves can retain
sampling noise. rho=0 with v=None gives plain Pegasos.
"""
import numpy as np


def prox_pegasos(X, y, v, A, rho, n_steps, rng):
    """Minimise (A/2)||w||^2 + hinge, pulled toward v, starting from w0 = v.

    Args:
        X, y:     the node's shard (CSR) and its labels in {-1, +1}
        v:        the proximal centre, z_i - u_i in ADMM
        A:        curvature; also sets eta_t = 1/(A*t)
        rho:      ADMM penalty; 0 disables the pull toward v
        n_steps:  Pegasos steps (t restarts at 1 each call, as in methods/)
        rng:      a numpy Generator

    Returns float32 w.
    """
    n = len(y)
    if n == 0:
        return np.asarray(v, dtype=np.float32).copy()

    indptr, indices, data = X.indptr, X.indices, X.data
    v64 = np.asarray(v, dtype=np.float64)
    u = v64.copy()          # w starts at v, so u = v and s = 1, c = 0
    s, c = 1.0, 0.0
    use_v = rho != 0.0

    for t in range(1, n_steps + 1):
        eta = 1.0 / (A * t)
        i = int(rng.integers(n))
        st, e = indptr[i], indptr[i + 1]
        cols, vals = indices[st:e], data[st:e]
        if len(cols) == 0:
            continue

        w_i = s * u[cols]
        if use_v:
            w_i = w_i + c * v64[cols]
        margin = float(y[i]) * float(np.dot(w_i, vals))

        a = 1.0 - eta * A
        s *= a
        if use_v:
            c = c * a + eta * rho
        if abs(s) < 1e-12:      # exact: at t=1, eta*A == 1 annihilates s*u
            u[:] = 0.0
            s = 1.0

        if margin < 1.0:
            u[cols] += (eta * float(y[i]) / s) * vals

    w = s * u
    if use_v:
        w += c * v64
    return w.astype(np.float32)
