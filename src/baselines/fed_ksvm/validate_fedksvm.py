"""Validator for the Fed-KSVM baseline. Prints PASS/FAIL per check, exits non-zero on failure.

    python src/baselines/fed_ksvm/validate_fedksvm.py

F1  random Fourier features: (a~ a~^T) approximates the RBF Gram, error shrinking with D
F2  Theorem 1: block boosting (L = 4) reaches the same optimum of the local primal (8)
    as the single-block exact SDCA solve (L = 1)
F3  Fed-KSVM with beta = 1 (CoCoA averaging) converges to the centralised linear SVM on
    the same random features (sklearn LinearSVC, hinge, no intercept, same C)
F4  w(t) == sum_m u_m at every round (primal-dual bookkeeping), with and without
    secure aggregation (masks cancel up to float rounding)
"""

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.baselines.fed_ksvm.rff import RandomFourierFeatures
from src.baselines.fed_ksvm.block_boosting import BlockBoosting
from src.nonlinear_sdca.kernel import rbf_gram

RESULTS = []


def report(name, ok, msg):
    RESULTS.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {msg}")


def _data(n=200, p=5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = np.where(np.sum(X[:, :2] ** 2, axis=1) > 2.0, 1.0, -1.0)
    return X, y


def primal8(F, y, C, w_bar, w):
    return C * np.maximum(0.0, 1.0 - y * (F @ (w_bar + w))).sum() + 0.5 * w @ w


def check_f1():
    X, _ = _data(120)
    K = rbf_gram(X, X, 0.5)
    errs = []
    for D in (100, 1000, 10000):
        F = RandomFourierFeatures(X.shape[1], D, 0.5, seed=1).transform(X)
        errs.append(float(np.abs(F @ F.T - K).max()))
    report("F1 RFF approximates RBF", errs[2] < errs[0] and errs[2] < 0.05,
           f"max |a~a~^T - K| for D=100/1000/10000: {', '.join(f'{e:.3f}' for e in errs)}")


def check_f2():
    X, y = _data(150)
    F = RandomFourierFeatures(X.shape[1], 64, 0.5, seed=2).transform(X)
    C = 0.5
    w_bar = np.random.default_rng(3).normal(scale=0.1, size=64)
    exact = BlockBoosting(F, y, C, L=1, Q=1, H=400, rng=np.random.default_rng(4))
    bb = BlockBoosting(F, y, C, L=4, Q=400, H=5, rng=np.random.default_rng(5))
    w0 = np.zeros(64)
    p_exact = primal8(F, y, C, w_bar, exact.solve(w_bar, w0))
    p_bb = primal8(F, y, C, w_bar, bb.solve(w_bar, w0))
    rel = abs(p_bb - p_exact) / abs(p_exact)
    report("F2 block boosting == exact local solve (Theorem 1)", rel < 1e-3,
           f"primal (8): exact {p_exact:.6f}, block-boosted {p_bb:.6f}, rel diff {rel:.1e}")


def _build(secure, X, y, M=4, D=64):
    from src.config import CONFIG
    from src.baselines.fed_ksvm.algorithm import FedKSVMRun
    shards = []
    for m, idx in enumerate(np.array_split(np.arange(len(y)), M)):
        shards.append(dict(X_csr=sp.csr_matrix(X[idx]), y=y[idx], X_test=sp.csr_matrix(X[:20]),
                           y_test=y[:20], n_local=len(idx)))
    cfg = dict(CONFIG, SEED=7, LAMBDA=0.01, RBF_GAMMA=0.5, KERNEL="rbf", FEDKSVM_D=D,
               FEDKSVM_L=1, FEDKSVM_Q=1, FEDKSVM_H=50, FEDKSVM_BETA=1.0,
               FEDKSVM_SECURE_AGG=secure, EVAL_EVERY=1, STOP_ON_PLATEAU=False)
    return FedKSVMRun(cfg, shards).build(), cfg


def check_f3_f4():
    from sklearn.svm import LinearSVC
    X, y = _data(200)
    for secure in (False, True):
        run, cfg = _build(secure, X, y)
        run.run(300)
        drift = max(r["w_minus_sum_u"] for r in run.devices[0].metrics)
        report(f"F4 w == sum_m u_m (secure_agg={secure})", drift < 1e-8,
               f"max ||w - sum u_m|| over rounds = {drift:.2e}")
        if secure:
            continue
        F = run.F_all
        svc = LinearSVC(loss="hinge", C=run.C, fit_intercept=False, tol=1e-10,
                        max_iter=200000, dual=True).fit(F, run.y_all)
        w_ref = svc.coef_.ravel()
        obj = lambda w: float(np.maximum(0.0, 1.0 - run.y_all * (F @ w)).mean()
                              + 0.5 * cfg["LAMBDA"] * w @ w)
        rel = abs(obj(run.w) - obj(w_ref)) / obj(w_ref)
        report("F3 Fed-KSVM (beta=1) -> centralised SVM on the same features", rel < 1e-3,
               f"objective Fed-KSVM {obj(run.w):.6f} vs LinearSVC {obj(w_ref):.6f} (rel {rel:.1e})")


CHECKS = {"F1": check_f1, "F2": check_f2, "F3": check_f3_f4}


def main():
    for name in [a.upper() for a in sys.argv[1:]] or list(CHECKS):
        CHECKS[name]()
    n_fail = RESULTS.count(False)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
