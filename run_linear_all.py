"""Run SDCA and linear-kernel BDSVM on the identical shards for eight datasets.

Outputs are isolated in results/linear_<dataset>_<tag>; existing RBF and CoCoA
results are never overwritten. Datasets run in separate processes to use CPU.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

from run_compare import DATASET_KEYWORDS

ROOT = Path(__file__).resolve().parent
ORDER = ["cov", "gis", "rsim", "rcv", "ijcnn", "adult", "w8a", "webspam"]


def run_one(keyword, cycles, tag, blas_threads):
    dataset = DATASET_KEYWORDS[keyword]
    out = ROOT / "results" / f"linear_{dataset}_{tag}"
    sdca_path = out / "sdca_metrics.json"
    linear_path = out / "bdsvm_linear_metrics.json"
    if sdca_path.exists() and linear_path.exists():
        print(f"{dataset}: keeping completed run in {out}", flush=True)
        return dataset, out, 0
    if sdca_path.exists() or linear_path.exists():
        raise RuntimeError(f"Partial run already exists in {out}; use a new --tag")
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "run.log"
    env = os.environ.copy()
    for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[key] = str(blas_threads)
    cmd = [sys.executable, "run_compare.py", keyword, "--only", "sdca",
           "bdsvm_linear", "--out", str(out)]
    if cycles is not None:
        cmd += ["--cycles", str(cycles)]
    with log_path.open("w", encoding="utf-8") as log:
        status = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log,
                                stderr=subprocess.STDOUT).returncode
    if status == 0:
        with log_path.open("a", encoding="utf-8") as log:
            plot = subprocess.run([sys.executable, "plot_merged.py", str(out)],
                                  cwd=ROOT, env=env, stdout=log,
                                  stderr=subprocess.STDOUT)
        status = plot.returncode
    return dataset, out, status


def summarize(directories, destination):
    rows = []
    for dataset, out, status in directories:
        if status:
            rows.append({"dataset": dataset, "status": f"failed ({status})"})
            continue
        row = {"dataset": dataset, "status": "ok"}
        for method in ("sdca", "bdsvm_linear"):
            result = json.loads((out / f"{method}_metrics.json").read_text())
            stops = result["per_node_stop_cycle"]
            row[f"{method}_accuracy"] = result["average_accuracy"]
            row[f"{method}_max_stop_cycle"] = (max(stops) if all(
                stop is not None for stop in stops) else "cap")
            row[f"{method}_communication_mb"] = result["total_comm_bytes"] / 1e6
            row[f"{method}_kernel"] = (result.get("bdsvm_params", {}).get("kernel")
                                       if method == "bdsvm_linear" else "linear")
        row["accuracy_delta_linear_minus_sdca"] = (
            row["bdsvm_linear_accuracy"] - row["sdca_accuracy"])
        rows.append(row)
    fields = ["dataset", "status", "sdca_accuracy", "bdsvm_linear_accuracy",
              "accuracy_delta_linear_minus_sdca", "sdca_max_stop_cycle",
              "bdsvm_linear_max_stop_cycle", "sdca_communication_mb",
              "bdsvm_linear_communication_mb", "sdca_kernel",
              "bdsvm_linear_kernel"]
    with destination.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="*", choices=sorted(DATASET_KEYWORDS))
    parser.add_argument("--cycles", type=int, default=None)
    parser.add_argument("--jobs", type=int, default=None)
    parser.add_argument("--tag", default=date.today().isoformat())
    args = parser.parse_args()
    selected = args.datasets or ORDER
    cpus = os.cpu_count() or 1
    jobs = max(1, min(args.jobs or cpus, len(selected)))
    blas_threads = max(1, cpus // jobs)
    print(f"Running {len(selected)} datasets with {jobs} processes and "
          f"{blas_threads} BLAS threads each", flush=True)
    finished = []
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(run_one, kw, args.cycles, args.tag, blas_threads)
                   for kw in selected]
        for future in as_completed(futures):
            dataset, out, status = future.result()
            finished.append((dataset, out, status))
            print(f"{dataset}: {'ok' if status == 0 else 'FAILED'} -> {out}", flush=True)
    summary = ROOT / "results" / f"linear_summary_{args.tag}.csv"
    summarize(sorted(finished), summary)
    print(f"Summary -> {summary}", flush=True)
    completed_dirs = [str(out) for _, out, status in finished if status == 0]
    if completed_dirs:
        overview_dir = ROOT / "results" / f"plots_linear_{args.tag}"
        for metric in ("accuracy", "hinge_loss", "comm_bytes"):
            status = subprocess.run([sys.executable, "plot_overview.py",
                                     *completed_dirs, "--metric", metric,
                                     "--out", str(overview_dir)],
                                    cwd=ROOT).returncode
            if status:
                raise SystemExit(status)
        print(f"Overview plots -> {overview_dir}", flush=True)
    if any(status for _, _, status in finished):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
