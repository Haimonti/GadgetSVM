"""Compiled equivalent of _prox.prox_pegasos; identical sampled row sequence.

No fastmath: preserve the float64 recursion and float32 returned model.
Randomness is drawn by the original NumPy Generator outside the compiled loop.
"""
import numpy as np
from numba import njit


@njit(cache=True)
def _solve(indptr, indices, data, y, v, A, rho, draws):
    u = v.copy()
    s, c = 1., 0.
    for j in range(len(draws)):
        t = j + 1
        eta = 1. / (A*t)
        i = draws[j]
        st, end = indptr[i], indptr[i+1]
        if st == end:
            continue
        score = 0.
        for q in range(st, end):
            col = indices[q]
            wi = s*u[col]
            if rho != 0.:
                wi += c*v[col]
            score += wi*data[q]
        margin = float(y[i])*score
        a = 1.-eta*A
        s *= a
        if rho != 0.:
            c = c*a+eta*rho
        if abs(s) < 1e-12:
            u[:] = 0.
            s = 1.
        if margin < 1.:
            amount = eta*float(y[i])/s
            for q in range(st, end):
                u[indices[q]] += amount*data[q]
    return (s*u+c*v).astype(np.float32)


def prox_pegasos_fast(X, y, v, A, rho, n_steps, rng):
    if len(y) == 0:
        return np.asarray(v, dtype=np.float32).copy()
    draws = rng.integers(len(y), size=n_steps)
    return _solve(X.indptr, X.indices, X.data, y,
                  np.asarray(v, dtype=np.float64), A, rho, draws)
