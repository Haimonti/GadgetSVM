"""Same proximal Pegasos recursion in a row-Gram representation.

For a small dense shard with n << d, maintain X@u rather than rebuilding a
long feature dot product for every sampled row. This is algebraically the same
update as _prox.py, with the same row draws and float64 arithmetic.
"""
import numpy as np
from numba import njit


@njit(cache=True)
def _coefficients(G, xv, y, A, rho, draws):
    n = len(y)
    coeff = np.zeros(n, dtype=np.float64)
    scores = xv.copy()
    base, s, c = 1., 1., 0.
    for j in range(len(draws)):
        i = draws[j]
        eta = 1./(A*(j+1))
        margin = float(y[i])*(s*scores[i]+c*xv[i])
        a = 1.-eta*A
        s *= a
        c = c*a+eta*rho
        if abs(s) < 1e-12:
            coeff[:] = 0.
            scores[:] = 0.
            base, s = 0., 1.
        if margin < 1.:
            delta = eta*float(y[i])/s
            coeff[i] += delta
            for k in range(n):
                scores[k] += delta*G[i, k]
    return s*coeff, s*base+c


def prox_pegasos_gram(X, G, y, v, A, rho, n_steps, rng):
    v64 = np.asarray(v, dtype=np.float64)
    draws = rng.integers(len(y), size=n_steps)
    coeff, base = _coefficients(G, X.dot(v64), y, A, rho, draws)
    return np.asarray(X.T.dot(coeff)+base*v64, dtype=np.float32)
