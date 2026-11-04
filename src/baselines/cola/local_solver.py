"""COLA's LinearSVM local subproblem solver — a line-by-line port of the official code.

Sources (github.com/epfml/cola):
  * `cola/cocoasolvers.py::LinearSVM`  — subproblem standardisation + recovery
  * `fast_cd/solver.py::SVMCoordSolver` — warm-started wrapper
  * `fast_cd/svm.pyx::sparse_cd`        — the randomised dual coordinate descent

Problem (COLA's form, data split by samples; A = X^T is d x n, x signed duals):

    min_x  C/2 ||A x||^2 - <y, x>     s.t.  y_i x_i in [0, 1]

which is our dual exactly when C = 1/(lambda * n_global): then w = C * A x equals
our w = X^T alpha / (lambda n), and the objective is -n times our D(alpha).

Node k's CoCoA subproblem at the averaged estimate v of A x is

    min_dx  <C v, A_k dx> + sigma C / 2 ||A_k dx||^2 - <y_k, x_k + dx>

`standardize` rewrites it as `sparse_cd`'s problem on alpha = x_k + dx,

    min_alpha  C'/2 ||A_k alpha||^2 - <y_k + b, alpha>,
    C' = sigma C,   b = A_k^T (sigma C A_k x_k - C v)

and `sparse_cd` runs `max_iter * n_k` random coordinate steps (in passes of n_k,
with COLA's xorshift RNG reseeded from a RandomState on every call), stopping
early only when its local gap drops below `tol`.
"""

import numpy as np

RAND_R_MAX = 0x7FFFFFFF
_U32 = 0xFFFFFFFF


def _our_rand_r(state):
    """svm.pyx `our_rand_r`: 32-bit xorshift (13, 17, 5), then mod (RAND_R_MAX+1)."""
    s = state[0]
    s ^= (s << 13) & _U32
    s ^= s >> 17
    s ^= (s << 5) & _U32
    state[0] = s
    return s % (RAND_R_MAX + 1)


def sparse_cd(alpha, y, b, X, max_iter, C, tol, rng):
    """Port of `fast_cd/svm.pyx::sparse_cd`. `X` is the node's shard (n_k x d CSR),
    i.e. the CSC storage of A_k that the Cython code reads. Mutates and returns
    alpha; also returns (gap, n_iter) like the original."""
    indptr, indices, data = X.indptr, X.indices, X.data
    n_cols = len(y)
    n_iter_total = int(max_iter * n_cols)
    n_iter = n_iter_total
    gap = tol + 1.0
    state = [rng.randint(0, RAND_R_MAX) & _U32]

    norm_cols = np.array([float(np.dot(data[indptr[j]:indptr[j + 1]],
                                       data[indptr[j]:indptr[j + 1]])) for j in range(n_cols)])
    V = np.asarray(X.T.dot(alpha)).ravel()                       # V = A @ alpha

    while n_iter > 0:
        alpha_max = 0.0
        d_alpha_max = 0.0
        for _ in range(min(n_cols, n_iter)):
            jj = _our_rand_r(state) % n_cols
            if norm_cols[jj] == 0.0:
                continue
            s, e = indptr[jj], indptr[jj + 1]
            cols, vals = indices[s:e], data[s:e]
            alpha_jj = alpha[jj]
            tmp = float(np.dot(vals, V[cols]))
            delta = (y[jj] + b[jj] - C * tmp) / (C * norm_cols[jj])
            alpha[jj] = y[jj] * max(0.0, min(1.0, y[jj] * (alpha_jj + delta)))
            delta = alpha[jj] - alpha_jj
            if delta != 0.0:
                V[cols] += delta * vals
            d_alpha_jj = abs(alpha[jj] - alpha_jj)
            if d_alpha_jj > d_alpha_max:
                d_alpha_max = d_alpha_jj
            if abs(alpha[jj]) > alpha_max:
                alpha_max = abs(alpha[jj])
        if alpha_max == 0.0 or d_alpha_max / alpha_max < tol or n_iter <= n_cols:
            tmp_all = np.asarray(X.dot(V)).ravel()               # A[:, j] . V for every j
            gap = float(np.sum(tmp_all * alpha + np.maximum(0.0, 1.0 - y * (tmp_all - b))
                               - (y + b) * alpha)) / n_cols
            if gap < tol:
                break
        n_iter -= min(n_cols, n_iter)
    return alpha, gap, (n_iter_total - n_iter) + 1


class ColaLinearSVMSolver:
    """`LinearSVM` + `SVMCoordSolver` for one node (warm-started across rounds)."""

    def __init__(self, X, y, C, sigma, local_iters, theta, random_state):
        self.X = X.tocsr()
        self.y = np.asarray(y, dtype=np.float64)
        self.C = float(C)
        self.sigma = float(sigma)
        self.local_iters = float(local_iters)
        self.theta = float(theta)
        # check_random_state(random_state) is built once per solver in SVMCoordSolver
        self.rng = np.random.RandomState(random_state)
        self.dual_coef = np.zeros(self.X.shape[0])
        self.last_gap = None
        self.last_iters = None

    def solve(self, v, Akxk, xk):
        """`LinearSVM.solve`: returns (delta_x, delta_Akxk).

        Faithful detail: `LinearSVM.solve` assigns `solver.coef_ = xk`, but
        `SVMCoordSolver.fit` warm-starts from `dual_coef_` (its previous solution),
        not `coef_`. We reproduce that; with COLA's gamma = 1 the two coincide."""
        w = self.C * v                                           # grad_f(v)
        b = np.asarray(self.X.dot(self.sigma * self.C * Akxk - w)).ravel()
        alpha, gap, n_iter = sparse_cd(self.dual_coef, self.y, b, self.X, self.local_iters,
                                       self.sigma * self.C, self.theta, self.rng)
        self.dual_coef = alpha
        self.last_gap, self.last_iters = gap, n_iter
        delta_x = alpha - xk
        delta_v = np.asarray(self.X.T.dot(delta_x)).ravel()
        return delta_x, delta_v
