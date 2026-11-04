"""Kernel helpers for the non-linear models — RBF and polynomial only.

DSRR itself prescribes no kernel (it is a *linear* reduction). The project scope
(10/26 meeting) is kernel SVMs with exactly two kernels, chosen by
CONFIG["KERNEL"]:

    rbf   K(a, b) = exp(-gamma ||a - b||^2)                    gamma = RBF_GAMMA
    poly  K(a, b) = (g <a, b> + coef0)^degree                  g = POLY_GAMMA (None = 1/d),
                                                               coef0 = POLY_COEF0, degree = POLY_DEGREE

`make_kernel(cfg)` returns the Gram function K(A, B) for the configured kernel;
every non-linear model (our kernel SDCA, Forero) calls it, so they always use the
same kernel. Thin wrappers over scikit-learn's vectorised kernels.
"""

import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel

KERNELS = ("rbf", "poly")

# Largest Gram block kernel_scores builds at once: 2**25 float64 = 256 MB. Bounds
# memory on large datasets (w8a / ijcnn1) without changing any score.
MAX_KERNEL_ENTRIES = 2 ** 25


def _arr(M):
    """Dense float64 array, or a sparse matrix as-is (raw data in the recovery step)."""
    return M if sp.issparse(M) else np.asarray(M, dtype=np.float64)


def rbf_gram(A, B, gamma):
    """Gram matrix K[i, j] = exp(-gamma ||A_i - B_j||^2), shape (len(A), len(B)).

    Handles the empty-set case (no support vectors yet) by returning a correctly
    shaped empty array so callers never special-case it.
    """
    A, B = _arr(A), _arr(B)
    if A.shape[0] == 0 or B.shape[0] == 0:
        return np.zeros((A.shape[0], B.shape[0]), dtype=np.float64)
    return rbf_kernel(A, B, gamma=gamma)


def kernel_scores(sv_x, sv_alpha, points, kernel, max_entries=MAX_KERNEL_ENTRIES):
    """Unscaled kernel score ``sum_j alpha_j K(sv_j, point)`` for each point.

    ``kernel`` is a Gram function K(A, B) (from `make_kernel`); a bare number is
    taken as an RBF gamma, for older callers.

    Returns a vector of length ``len(points)``. The 1/(lambda*n) scaling that turns
    this into the SDCA score ``f`` is applied by the caller, since the decision
    sign is unaffected by a positive scale.

    Memory-bounded: the (n_sv x n_points) Gram block is never larger than
    ``max_entries``; bigger requests are computed over row chunks of ``points``
    (each point's score is still the same sum over every support vector). A
    request that fits is computed in one shot, exactly as before.
    """
    n_pts = points.shape[0]
    n_sv = len(sv_alpha)
    if n_sv == 0 or n_pts == 0:
        return np.zeros(n_pts, dtype=np.float64)
    gram = kernel if callable(kernel) else (lambda A, B: rbf_gram(A, B, kernel))
    alpha = np.asarray(sv_alpha, dtype=np.float64)
    if n_sv * n_pts <= max_entries:
        return gram(sv_x, points).T @ alpha
    rows = max(1, max_entries // n_sv)
    out = np.empty(n_pts, dtype=np.float64)
    for s in range(0, n_pts, rows):
        out[s:s + rows] = gram(sv_x, points[s:s + rows]).T @ alpha
    return out


def poly_gram(A, B, degree=3, gamma=None, coef0=1.0):
    """Gram matrix (gamma <A_i, B_j> + coef0)^degree; gamma None = 1/n_features."""
    A, B = _arr(A), _arr(B)
    if A.shape[0] == 0 or B.shape[0] == 0:
        return np.zeros((A.shape[0], B.shape[0]), dtype=np.float64)
    return polynomial_kernel(A, B, degree=degree, gamma=gamma, coef0=coef0)


def make_kernel(cfg):
    """K(A, B) for CONFIG["KERNEL"] in {"rbf", "poly"}."""
    name = cfg.get("KERNEL", "rbf")
    if name == "rbf":
        gamma = cfg["RBF_GAMMA"]
        return lambda A, B: rbf_gram(A, B, gamma)
    if name == "poly":
        deg, g, c0 = cfg.get("POLY_DEGREE", 3), cfg.get("POLY_GAMMA"), cfg.get("POLY_COEF0", 1.0)
        return lambda A, B: poly_gram(A, B, deg, g, c0)
    raise ValueError(f"Unknown KERNEL '{name}'. Choose: {' | '.join(KERNELS)}")
