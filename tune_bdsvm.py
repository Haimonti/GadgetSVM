"""Select BDSVM's hyperparameters for one dataset on a validation split.

The covtype settings in `docs_hyperparameters.md` came from a grid on
validation AUC; this is that procedure made repeatable, so gisette, real-sim
and rcv1 get their own C, P and gamma instead of inheriting covtype's. Carried
over unchanged, covtype's C = 30 / P = 1000 produced accuracy that *fell* over
training on gisette and a 5000-cycle limit cycle on real-sim — convergence to
a poor optimum and no convergence at all, respectively.

What is searched and why:

  C           the penalty in the weighting rule a_i = 2C/(e_i y_i). Sets how
              much the data term outweighs the (1/2) beta' K_p beta regulariser;
              too small and IRWLS settles below the epoch-1 iterate.
  P           the pre-image budget. On gisette each shard holds 480 rows, so
              P = 1000 makes every worker's least-squares block underdetermined
              and the ridge dominates. Smaller P is also quadratically cheaper
              to gossip.
  gamma       as a multiple of the median heuristic, since that heuristic is a
              scale guess rather than a tuned value.

The split: the same shards `load_shards(CONFIG)` hands the P2P run — same seed,
same test hold-out — are concatenated and 20% is carved off as validation. The
test set is never touched here. Selection is by validation AUC at the epoch
the eta rule stops, and each row also records the peak AUC and where it
occurred, so a configuration that peaks early and decays is visible rather
than hidden behind a final number.

Configurations are independent, so they run as a process pool with BLAS held
to one thread per worker (threadpoolctl), the way run_grid.py does.

    python tune_bdsvm.py gis
    python tune_bdsvm.py rsim --jobs 20
    python tune_bdsvm.py rcv --out results/tune
"""

import argparse
import csv
import itertools
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import CONFIG, CODE_DIR
from src.data_sharding import load_shards
from methods.bdsvm import (_rbf, _make_preimages, _median_gamma,
                           _worker_contribution)
from run_compare import DATASET_KEYWORDS

GRID = {
    "C":          [1.0, 10.0, 30.0, 100.0, 300.0, 1000.0],
    "P":          [100, 300, 1000],
    "gamma_mult": [0.5, 1.0, 2.0],
}
# Pre-image kinds to try. Where the right kind is settled (see _make_preimages)
# it is one value; for the new low-dimensional sets it is not obvious — w8a and
# a9a are sparse binary in 300 / 123 dims, ijcnn1 is dense in [-1, 1]^22 where
# "uniform" on [0,1] covers half the cube, webspam is normalised counts — so all
# three kinds go into the grid and the data decides.
PREIMAGE = {"covtype": ["uniform"], "gisette": ["uniform"],
            "real-sim": ["sparse"], "rcv1": ["sparse"],
            "w8a": ["uniform", "unit", "sparse"],
            "a9a": ["uniform", "unit", "sparse"],
            "ijcnn1": ["uniform", "unit", "sparse"],
            "webspam": ["uniform", "unit", "sparse"]}
RHO, ETA, MAX_EPOCHS, N_CLIENTS, VAL_FRAC = 0.5, 5e-3, 100, 10, 0.2


def split(config, max_train=None):
    """Validation split off the P2P run's own training shards; test untouched.

    `max_train` caps the rows used for the search. One IRWLS epoch is
    O(n P^2): on webspam's 280k training rows at P = 1000 that is ~2e11 flops
    per epoch per configuration, which makes the grid a day's work. A 50k
    subsample ranks configurations the same way at 1/6 the cost — the same
    cap run_grid.py used on covtype.
    """
    shards = load_shards(config)
    X = sp.vstack([s["X_csr"] for s in shards]).tocsr()
    y = np.concatenate([s["y"] for s in shards]).astype(np.float64)
    rng = np.random.RandomState(config["SEED"] + 1)
    if max_train is not None and X.shape[0] > max_train:
        keep = rng.choice(X.shape[0], size=max_train, replace=False)
        X, y = X[keep], y[keep]
    perm = rng.permutation(X.shape[0])
    n_val = int(VAL_FRAC * X.shape[0])
    va, tr = perm[:n_val], perm[n_val:]
    return X[tr], y[tr], X[va], y[va]


def one(args):
    """Train one configuration; return its row. Runs in a worker process."""
    (dataset, C, P, gmult, preimage, seed,
     X_tr, y_tr, X_va, y_va) = args
    from threadpoolctl import threadpool_limits
    from sklearn.metrics import roc_auc_score
    t0 = time.time()
    with threadpool_limits(limits=1):
        p = _make_preimages(P, X_tr.shape[1], seed, kind=preimage)
        gamma = gmult * _median_gamma(X_tr, p)
        Kpp = np.zeros((P + 1, P + 1))
        Kpp[:P, :P] = _rbf(p, p, gamma)
        clients = np.array_split(np.arange(X_tr.shape[0]), N_CLIENTS)
        workers = []
        for ci in clients:
            Km = _rbf(X_tr[ci], p, gamma)
            workers.append((np.hstack([Km, np.ones((Km.shape[0], 1))]), y_tr[ci]))
        K_va = _rbf(X_va, p, gamma)

        beta = np.zeros(P + 1)
        aucs, accs = [], []
        for epoch in range(1, MAX_EPOCHS + 1):
            C_sum = np.zeros((P + 1, P + 1))
            d_sum = np.zeros(P + 1)
            for Km, ym in workers:
                C_m, d_m = _worker_contribution(Km, ym, beta, C)
                C_sum += C_m
                d_sum += d_m
            A = C_sum + Kpp
            A[np.diag_indices_from(A)] += 1e-8 * max(np.trace(A), 1.0) / A.shape[0]
            try:
                beta_new = np.linalg.solve(A, d_sum)
            except np.linalg.LinAlgError:
                beta_new = np.linalg.lstsq(A, d_sum, rcond=None)[0]
            beta_prev = beta
            beta = RHO * beta_prev + (1.0 - RHO) * beta_new

            s = K_va @ beta[:P] + beta[P]
            aucs.append(float(roc_auc_score(y_va, s)))
            accs.append(float(np.mean(np.where(s >= 0, 1.0, -1.0) == y_va)))

            denom = np.linalg.norm(beta_prev)
            if denom > 0 and np.linalg.norm(beta - beta_prev) / denom < ETA:
                break

    aucs, accs = np.array(aucs), np.array(accs)
    tail = aucs[-10:]
    return dict(
        dataset=dataset, C=C, P=P, gamma_mult=gmult, gamma=gamma,
        preimage=preimage, epochs=len(aucs),
        val_auc=aucs[-1], val_auc_peak=aucs.max(), peak_epoch=int(aucs.argmax()) + 1,
        val_acc=accs[-1], val_acc_peak=accs.max(),
        decay=aucs.max() - aucs[-1],          # how far it fell from its own peak
        tail_std=float(tail.std()),           # oscillation over the last epochs
        seconds=time.time() - t0,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=sorted(DATASET_KEYWORDS))
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-train", type=int, default=None,
                    help="cap on training rows used for the search (see split)")
    args = ap.parse_args()

    CONFIG["DATASET"] = DATASET_KEYWORDS[args.dataset]
    dataset = CONFIG["DATASET"]
    out_dir = Path(args.out) if args.out else CODE_DIR / "results" / "tune"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"bdsvm_{dataset}.csv"

    X_tr, y_tr, X_va, y_va = split(CONFIG, args.max_train)
    print(f"{dataset}: {X_tr.shape[0]} train / {X_va.shape[0]} val, "
          f"{X_tr.shape[1]} features, preimage in {PREIMAGE[dataset]}")

    jobs = [(dataset, C, P, g, kind, 0, X_tr, y_tr, X_va, y_va)
            for C, P, g, kind in itertools.product(GRID["C"], GRID["P"],
                                                   GRID["gamma_mult"],
                                                   PREIMAGE[dataset])]
    print(f"{len(jobs)} configurations")

    rows = []
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(one, j): j for j in jobs}
        for f in as_completed(futs):
            r = f.result()
            rows.append(r)
            print(f"  C={r['C']:<7g} P={r['P']:<5} g×{r['gamma_mult']:<4g} {r['preimage']:<8} "
                  f"epochs={r['epochs']:<4} val_auc={r['val_auc']:.4f} "
                  f"(peak {r['val_auc_peak']:.4f}@{r['peak_epoch']}, "
                  f"decay {r['decay']:.4f}, tail_std {r['tail_std']:.4f}) "
                  f"val_acc={r['val_acc']:.4f}  {r['seconds']:.0f}s", flush=True)

    rows.sort(key=lambda r: -r["val_auc"])
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\nTop 8 by validation AUC → {out}")
    print(f"{'C':>7} {'P':>5} {'g×':>4} {'preimage':>8} {'epochs':>6} {'val_auc':>8} {'peak':>8} "
          f"{'decay':>7} {'tail_std':>8} {'val_acc':>8}")
    for r in rows[:8]:
        print(f"{r['C']:>7g} {r['P']:>5} {r['gamma_mult']:>4g} {r['preimage']:>8} {r['epochs']:>6} "
              f"{r['val_auc']:>8.4f} {r['val_auc_peak']:>8.4f} {r['decay']:>7.4f} "
              f"{r['tail_std']:>8.4f} {r['val_acc']:>8.4f}")


if __name__ == "__main__":
    main()
