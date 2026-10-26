"""Single-machine RBF kernel SDCA over a DSRR projection — correctness oracle.

The meeting action item ("work out a concrete kernelized-SDCA example on a single
machine first, then derive the gossip version"). This runs the SAME model the
distributed tenant runs — project with the shared DSRR matrix, then kernel SDCA in
the reduced space — but centrally, on the full training set with no gossip. The
distributed run (`src/scripts/main.py -m nonlinear`) should approach this accuracy / duality gap once its
accumulate-not-average support-vector exchange reaches consensus.

    python src/nonlinear_sdca/reference_kernel_sdca.py adult
    python src/nonlinear_sdca/reference_kernel_sdca.py gis --epochs 20
"""

import sys
import time
import argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import CONFIG
from src.scripts.dataset_keywords import select_dataset
from src.scripts.data_sharding import load_shards
from src.nonlinear_sdca.projection import project_shard
from src.nonlinear_sdca.kernel import rbf_gram


def _duality_gap(K, alpha, y, lam_n, lam, n, tau):
    f = (K @ alpha) / lam_n                              # score of every point
    hinge = float(np.mean(np.maximum(0.0, (1.0 - tau) - y * f)))
    norm_h2 = float(np.dot(alpha, f) / lam_n)           # ||f||_H^2
    reg = (lam / 2.0) * norm_h2
    primal = hinge + reg
    dual = float(np.dot(alpha, y)) / n - reg            # alpha.y = sum of unsigned
    return primal - dual, primal, hinge


def run(keyword, epochs):
    select_dataset(keyword)
    cfg = dict(CONFIG)
    cfg["NUM_WORKERS"] = 1                               # one shard = the full train set
    shard = load_shards(cfg)[0]

    seed, m, dist = cfg["SEED"], cfg["PROJECTION_DIM"], cfg["PROJECTION_DIST"]
    gamma, lam, tau = cfg["RBF_GAMMA"], cfg["LAMBDA"], cfg.get("DSRR_TAU", 0.0)

    t0 = time.time()
    Xhat, Xhat_test, _ = project_shard(
        shard["X_csr"], shard["X_test"], seed, m, dist
    )
    y = np.asarray(shard["y"], dtype=np.float64)
    y_test = np.asarray(shard["y_test"], dtype=np.float64)
    n = Xhat.shape[0]
    K = rbf_gram(Xhat, Xhat, gamma)                     # n x n local Gram
    print(f"[ref] {keyword}: n={n}, projected d->m={m} ({dist}), "
          f"gamma={gamma}, lambda={lam}, tau={tau}  ({time.time()-t0:.1f}s)")

    alpha = np.zeros(n, dtype=np.float64)
    f = np.zeros(n, dtype=np.float64)                   # f = (K @ alpha)/lam_n
    lam_n = lam * n
    rng = np.random.RandomState(seed)
    for ep in range(1, epochs + 1):
        for i in rng.permutation(n):
            kii = float(K[i, i])
            if kii < 1e-12:
                continue
            yi = float(y[i])
            unsigned_old = alpha[i] * yi
            unsigned_new = min(1.0, max(0.0, unsigned_old
                                        + ((1.0 - tau) - yi * f[i]) * lam_n / kii))
            delta = yi * unsigned_new - alpha[i]
            if delta == 0.0:
                continue
            alpha[i] += delta
            f += (delta / lam_n) * K[:, i]
        gap, primal, hinge = _duality_gap(K, alpha, y, lam_n, lam, n, tau)
        n_sv = int(np.count_nonzero(alpha))
        print(f"[ref] epoch {ep:2d}  primal={primal:.6f}  hinge={hinge:.6f}  "
              f"gap={gap:.3e}  support_vectors={n_sv}/{n}")

    # test accuracy from the support-vector expansion
    scores = rbf_gram(Xhat, Xhat_test, gamma).T @ alpha
    acc = float(np.mean(np.where(scores >= 0.0, 1.0, -1.0) == y_test))
    print(f"[ref] {keyword}: single-machine kernel-SDCA test accuracy = {acc:.4f}")
    return acc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", help="dataset keyword (rcv cov gis rsim w8a ijcnn adult webspam)")
    ap.add_argument("--epochs", type=int, default=15)
    a = ap.parse_args()
    run(a.dataset, a.epochs)
