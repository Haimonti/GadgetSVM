"""DSRR randomized reduction — the shared projection A (x_hat = A x).

The communication + recovery layer from Yang et al. (ICML 2015). Each node
projects its ``d``-dimensional data into ``m`` dimensions with a random matrix
``A in R^{m x d}``. Two properties matter for the gossip setting:

1. **Shared, not transmitted.** ``A`` is generated purely from
   ``(n_components, n_features, random_state)`` — never from the data values — so
   every node that passes the SAME ``seed``, ``m`` and feature count ``d`` builds
   the *identical* ``A`` locally. Nothing about ``A`` is ever sent over the wire;
   only the projected vectors ``x_hat`` travel.

2. **JL-preserving.** ``A`` is sub-Gaussian with variance ``1/m`` (Gaussian) or a
   sparse sign matrix (hashing), so by the Johnson-Lindenstrauss lemma distances
   are preserved up to ``(1 +/- eps)``. The RBF kernel on the projected data is
   therefore close to the kernel on the original data — the reduced-space model is
   *recoverable*.

Backed by scikit-learn's `random_projection` (already a dependency via
`src/scripts/data_sharding.py`). ``"gaussian"`` -> ``A_ij ~ N(0, 1/m)``; ``"sparse"`` ->
the memory-efficient Achlioptas/hashing matrix the paper uses for very large ``d``.
"""

import numpy as np
from sklearn.random_projection import (
    GaussianRandomProjection,
    SparseRandomProjection,
)


def make_projector(seed, m, dist="sparse"):
    """Build an (unfitted) deterministic random projector.

    ``random_state=seed`` makes the drawn matrix a pure function of
    ``(m, d, seed)``, so every node gets the same ``A`` once fitted on data with the
    same feature count. ``dense_output=True`` so the projected shards are plain
    dense arrays the kernel code can feed straight to the RBF.
    """
    if dist == "gaussian":
        return GaussianRandomProjection(n_components=m, random_state=seed)
    if dist == "sparse":
        return SparseRandomProjection(
            n_components=m, random_state=seed, dense_output=True
        )
    raise ValueError(f"Unknown PROJECTION_DIST '{dist}'. Choose: gaussian | sparse")


def project_shard(X, X_test, seed, m, dist="sparse"):
    """Fit the shared projector on ``X`` and project both train and test.

    Fitting only reads ``X``'s *shape* (feature count) to size ``A``; the matrix
    entries come from ``seed`` alone, so the fit is identical on every node that
    holds the same number of features. Returns dense float64 arrays
    ``(X_hat, X_hat_test, projector)``.
    """
    proj = make_projector(seed, m, dist)
    proj.fit(X)  # sizes A from (m, n_features) + seed — not from the data values
    X_hat = np.asarray(proj.transform(X), dtype=np.float64)
    X_hat_test = np.asarray(proj.transform(X_test), dtype=np.float64)
    return X_hat, X_hat_test, proj
