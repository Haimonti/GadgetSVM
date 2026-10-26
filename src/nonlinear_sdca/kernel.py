"""RBF kernel helpers for the non-linear tenant.

DSRR itself prescribes no kernel (it is a *linear* reduction), so per the project
decision the kernel applied in the projected space is the RBF / Gaussian kernel:

    K(a, b) = exp(-gamma * ||a - b||^2)

Kept in its own module so a different kernel (polynomial, Laplacian, ...) is a
drop-in replacement without touching the protocol. Thin wrappers over
scikit-learn's vectorised `rbf_kernel`.
"""

import numpy as np
from sklearn.metrics.pairwise import rbf_kernel


def rbf_gram(A, B, gamma):
    """Gram matrix K[i, j] = exp(-gamma ||A_i - B_j||^2), shape (len(A), len(B)).

    Handles the empty-set case (no support vectors yet) by returning a correctly
    shaped empty array so callers never special-case it.
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if A.shape[0] == 0 or B.shape[0] == 0:
        return np.zeros((A.shape[0], B.shape[0]), dtype=np.float64)
    return rbf_kernel(A, B, gamma=gamma)


def kernel_scores(sv_x, sv_alpha, points, gamma):
    """Unscaled kernel score ``sum_j alpha_j K(sv_j, point)`` for each point.

    Returns a vector of length ``len(points)``. The 1/(lambda*n) scaling that turns
    this into the SDCA score ``f`` is applied by the caller, since the decision
    sign is unaffected by a positive scale.
    """
    if len(sv_alpha) == 0 or len(points) == 0:
        return np.zeros(len(points), dtype=np.float64)
    K = rbf_gram(sv_x, points, gamma)      # (n_sv, n_points)
    return K.T @ np.asarray(sv_alpha, dtype=np.float64)
