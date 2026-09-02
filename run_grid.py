"""Parallel grid runner for the P2P experiments.

Every configuration is independent, so they are run as separate processes in a
pool rather than one after another. Each worker gets OMP_NUM_THREADS=1: the
matrices here are small enough that BLAS threading only makes the workers fight
each other, so the parallelism belongs at the configuration level.

    python run_grid.py                     # all of it, jobs = cpu_count - 2
    python run_grid.py --jobs 20
    python run_grid.py --only covtype      # covtype | rcv1
    python run_grid.py --dry-run           # list what would run

Resumable: a configuration that already has a result file is skipped, so an
interrupted run picks up where it stopped. Each configuration writes its own CSV
(no shared-file races); `--merge` combines them into results/grid_<set>.csv.
"""
import argparse
import csv
import glob
import itertools
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

OUT = Path("results/raw")
MERGED = Path("results")

METHODS = ["fedavg_svm", "cocoa", "cocoa_plus", "bdsvm", "fdr_svm", "fedssl_amc"]
TOPOLOGIES = ["ring", "random_kout", "full"]
SCHEMES = ["iid", "dirichlet_1.0", "dirichlet_0.3", "dirichlet_0.1", "label_skew"]
KS = [1, 2]          # gossip fan-out; k=1 leaves random_kout disconnected


def configs(which):
    """Every configuration in the grid, as (name, argv-tail) pairs."""
    out = []

    if which in ("all", "covtype"):
        for k, m, t, s in itertools.product(KS, METHODS, TOPOLOGIES, SCHEMES):
            out.append((f"covtype_k{k}_{m}_{t}_{s}", [
                "--method", m, "--topology", t, "--scheme", s,
                "--nodes", "10", "--cycles", "100", "--gossip-k", str(k),
                "--lambda-reg", "0.01", "--seed", "42",
                "--data", "data/covtype.libsvm.binary.scale",
                "--max-samples", "50000", "--tag", "covtype",
            ]))

    if which in ("all", "rcv1"):
        for k, m, s in itertools.product(KS, METHODS, ["iid", "dirichlet_0.3", "label_skew"]):
            extra = []
            if m == "bdsvm":
                # P=100 with the default gamma is degenerate at d=47k; the
                # budget has to scale with the intrinsic dimension.
                extra = ["--preimage", "sparse", "--P", "1000"]
            out.append((f"rcv1_k{k}_{m}_{s}", [
                "--method", m, "--topology", "random_kout", "--scheme", s,
                "--nodes", "10", "--cycles", "30", "--gossip-k", str(k),
                "--lambda-reg", "1e-4", "--seed", "42",
                "--data", "data/rcv1_train.binary",
                "--max-samples", "20000", "--tag", "rcv1",
            ] + extra))

    return out


def run_one(job):
    name, tail = job
    csv_path = OUT / f"{name}.csv"
    if csv_path.exists():
        return name, "skip", 0.0

    env = dict(os.environ)
    # One BLAS thread per worker: these matrices are small, and letting each
    # process spawn its own thread pool just oversubscribes the machine.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[var] = "1"

    t0 = time.perf_counter()
    tmp = OUT / f".{name}.partial"
    proc = subprocess.run(
        [sys.executable, "-m", "p2p.run_peersim", *tail, "--csv", str(tmp)],
        env=env, capture_output=True, text=True,
    )
    dt = time.perf_counter() - t0
    if proc.returncode != 0 or not tmp.exists():
        (OUT / f"{name}.err").write_text((proc.stdout or "") + (proc.stderr or ""))
        return name, "FAIL", dt
    tmp.replace(csv_path)          # only appears once complete
    return name, "ok", dt


def merge():
    MERGED.mkdir(parents=True, exist_ok=True)
    for tag in ("covtype", "rcv1"):
        files = sorted(glob.glob(str(OUT / f"{tag}_*.csv")))
        if not files:
            continue
        rows, header = [], None
        for f in files:
            with open(f) as fh:
                r = csv.reader(fh)
                h = next(r, None)
                if h is None:
                    continue
                header = header or h
                rows.extend(list(r))
        dest = MERGED / f"grid_{tag}.csv"
        with open(dest, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(header)
            w.writerows(rows)
        print(f"  {dest}  ({len(rows)} rows from {len(files)} configurations)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--only", default="all", choices=["all", "covtype", "rcv1"])
    ap.add_argument("--limit", type=int, default=None,
                    help="run only the first N pending configurations")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--merge", action="store_true", help="only merge existing results")
    args = ap.parse_args()

    if args.merge:
        merge()
        return

    OUT.mkdir(parents=True, exist_ok=True)
    jobs = configs(args.only)
    todo = [j for j in jobs if not (OUT / f"{j[0]}.csv").exists()]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(jobs)} configurations, {len(jobs) - len(todo)} already done, "
          f"{len(todo)} to run on {args.jobs} workers")
    if args.dry_run:
        for n, _ in todo:
            print("  " + n)
        return
    if not todo:
        merge()
        return

    done = fail = 0
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futures = {ex.submit(run_one, j): j[0] for j in todo}
        for fut in as_completed(futures):
            name, status, dt = fut.result()
            done += 1
            if status == "FAIL":
                fail += 1
            elapsed = time.perf_counter() - t0
            rate = done / max(elapsed, 1e-9)
            eta = (len(todo) - done) / rate if rate > 0 else 0
            flag = "  <-- FAILED" if status == "FAIL" else ""
            print(f"[{done}/{len(todo)}] {name} {status} {dt:.0f}s "
                  f"eta {eta/60:.0f}min{flag}", flush=True)

    print(f"\nfinished in {(time.perf_counter()-t0)/60:.1f} min, {fail} failure(s)")
    if fail:
        print(f"see {OUT}/*.err")
    merge()


if __name__ == "__main__":
    main()
