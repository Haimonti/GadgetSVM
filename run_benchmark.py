"""Federated SVM Benchmark Runner.

Sweeps over datasets × partitioning schemes × num_clients × methods,
logs results to results.csv, and prints a summary table.

Usage:
    python run_benchmark.py [--data-dir ./data] [--download] [--fast]
                            [--datasets rcv1 covtype]
                            [--methods centralized fedavg_svm bdsvm fdr_svm fedssl_amc]
                            [--out results.csv]

--fast: smaller subset for quick smoke-testing (rcv1 train ≤ 5k, covtype ≤ 10k,
        10 FL rounds instead of 50, 10 clients only).
"""
import argparse
import csv
import time
from itertools import product
from pathlib import Path

import numpy as np

import benchmark as data_lib
from methods import METHODS


# ── Experiment grid ─────────────────────────────────────────────────────────

SCHEMES = [
    ("iid",         {}),
    ("dirichlet",   {"alpha": 1.0}),
    ("dirichlet",   {"alpha": 0.3}),
    ("dirichlet",   {"alpha": 0.1}),
    ("label_skew",  {}),
]

FULL_DATASETS   = ["rcv1", "covtype"]
FULL_CLIENTS    = [10, 50]
FULL_ROUNDS     = 50
FULL_MAX_TRAIN  = {"rcv1": None, "covtype": 50_000}   # covtype train: subsample to 50k
FULL_MAX_TEST   = 20_000  # cap test eval — rcv1_test is 677k rows, evaluating all is slow

FAST_DATASETS   = ["rcv1"]
FAST_CLIENTS    = [10]
FAST_ROUNDS     = 10
FAST_MAX_TRAIN  = 5_000
FAST_MAX_TEST   = 2_000

METHOD_KWARGS = {
    "centralized": {},
    "fedavg_svm":  {"n_local_steps": 100, "lambda_reg": 0.01},
    "bdsvm":       {"P": 100, "C": 1.0, "lam": 0.5, "eta": 5e-3},
    "fdr_svm":     {"n_local_steps": 100, "lambda_reg": 0.01, "rho": 1.0, "eps_scale": 1.0},
    "fedssl_amc":  {"d_enc": 64, "svm_C": 1.0},
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def scheme_label(scheme, kwargs):
    if scheme == "dirichlet":
        return f"dirichlet_a{kwargs['alpha']}"
    return scheme


def subsample(X, y, n, seed=0):
    if n is None or len(y) <= n:
        return X, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y), n, replace=False)
    return X[idx], y[idx]


_NO_ROUNDS = {"fedssl_amc"}  # two-phase methods that don't use iterative rounds

def run_method(name, module, X_tr, y_tr, X_te, y_te, client_idx, n_rounds, seed):
    kwargs = dict(METHOD_KWARGS[name])
    if name == "centralized":
        return module.run(X_tr, y_tr, X_te, y_te, seed=seed, **kwargs)
    if name not in _NO_ROUNDS:
        kwargs["n_rounds"] = n_rounds
    kwargs["seed"] = seed
    return module.run(X_tr, y_tr, X_te, y_te, client_idx, **kwargs)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--download", action="store_true",
                    help="Download datasets before running")
    ap.add_argument("--fast", action="store_true",
                    help="Quick smoke-test: small subsets, fewer rounds")
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--methods",  nargs="+", default=None)
    ap.add_argument("--out", default="results.csv")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data_dir = args.data_dir

    if args.download:
        print("=== Downloading datasets ===")
        data_lib.download_all(data_dir)

    fast       = args.fast
    datasets   = args.datasets or (FAST_DATASETS if fast else FULL_DATASETS)
    all_methods = args.methods  or list(METHODS.keys())
    n_clients_list = FAST_CLIENTS if fast else FULL_CLIENTS
    n_rounds   = FAST_ROUNDS if fast else FULL_ROUNDS
    max_test   = FAST_MAX_TEST  if fast else FULL_MAX_TEST

    out_path = Path(args.out)
    write_header = not out_path.exists()
    csv_file = open(out_path, "a", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=[
        "dataset", "scheme", "num_clients", "method", "test_acc", "elapsed_s", "seed",
    ])
    if write_header:
        writer.writeheader()

    print(f"\n{'dataset':>10} {'scheme':>20} {'K':>4} {'method':>14}  "
          f"{'acc':>7}  {'time':>7}")
    print("-" * 70)

    for ds in datasets:
        try:
            print(f"[loading] {ds} train...", flush=True)
            X_tr, y_tr = data_lib.load_dataset(ds, split="train", data_dir=data_dir)
            print(f"[loading] {ds} test...", flush=True)
            X_te, y_te = data_lib.load_dataset(ds, split="test",  data_dir=data_dir)
        except FileNotFoundError as e:
            print(f"[skip] {ds}: {e}")
            continue

        if fast:
            max_train = FAST_MAX_TRAIN
        elif isinstance(FULL_MAX_TRAIN, dict):
            max_train = FULL_MAX_TRAIN.get(ds)
        else:
            max_train = FULL_MAX_TRAIN
        X_tr, y_tr = subsample(X_tr, y_tr, max_train, args.seed)
        X_te, y_te = subsample(X_te, y_te, max_test,  args.seed)

        for (scheme, skw), K, mname in product(SCHEMES, n_clients_list, all_methods):
            slabel = scheme_label(scheme, skw)
            client_idx = data_lib.partition(y_tr, K, scheme, **skw, seed=args.seed)

            module = METHODS[mname]
            t0 = time.perf_counter()
            try:
                result = run_method(mname, module,
                                    X_tr, y_tr, X_te, y_te,
                                    client_idx, n_rounds, args.seed)
                acc = result["test_acc"]
                status = f"{acc:.4f}"
            except Exception as exc:
                acc = float("nan")
                status = f"ERROR: {exc}"

            elapsed = time.perf_counter() - t0

            row = dict(dataset=ds, scheme=slabel, num_clients=K,
                       method=mname, test_acc=acc, elapsed_s=round(elapsed, 2),
                       seed=args.seed)
            writer.writerow(row)
            csv_file.flush()

            print(f"{ds:>10} {slabel:>20} {K:>4} {mname:>14}  "
                  f"{status:>7}  {elapsed:>6.1f}s")

    csv_file.close()
    print(f"\nResults saved to {out_path}")

    # Print pivot summary
    try:
        _print_summary(out_path)
    except Exception:
        pass


def _print_summary(csv_path):
    import csv as _csv
    rows = []
    with open(csv_path) as f:
        rows = list(_csv.DictReader(f))
    if not rows:
        return

    from collections import defaultdict
    # Group by (dataset, scheme, K, method) → last acc
    data = {}
    for r in rows:
        key = (r["dataset"], r["scheme"], r["num_clients"], r["method"])
        try:
            data[key] = float(r["test_acc"])
        except ValueError:
            pass

    methods = sorted({k[3] for k in data})
    configs = sorted({k[:3] for k in data})

    col_w = 12
    header = f"{'dataset':>10} {'scheme':>20} {'K':>4}  " + \
             "  ".join(f"{m:>{col_w}}" for m in methods)
    print("\n" + "=" * len(header))
    print("SUMMARY (test accuracy)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for ds, scheme, K in configs:
        accs = [f"{data.get((ds, scheme, K, m), float('nan')):>{col_w}.4f}"
                for m in methods]
        print(f"{ds:>10} {scheme:>20} {K:>4}  " + "  ".join(accs))


if __name__ == "__main__":
    main()
