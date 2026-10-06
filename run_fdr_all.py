"""Run FDR-SVM and current Lincoln SDCA on all eight datasets in parallel."""
import argparse
import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from run_compare import DATASET_KEYWORDS

ROOT = Path(__file__).resolve().parent
ORDER = ["cov", "gis", "rsim", "rcv", "ijcnn", "adult", "w8a", "webspam"]


def run_one(keyword, out_root, cycles, data_dir, blas_threads, covtype_path, gisette_path):
    dataset = DATASET_KEYWORDS[keyword]
    out = out_root / dataset
    out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["GADGETSVM_DATA_DIR"] = str(data_dir)
    if covtype_path:
        env["GADGETSVM_COVTYPE_PATH"] = str(covtype_path)
    if gisette_path:
        env["GADGETSVM_GISETTE_PATH"] = str(gisette_path)
    for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[name] = str(blas_threads)
    command = [sys.executable, "run_compare.py", keyword, "--only", "sdca",
               "fdr_svm", "--cycles", str(cycles), "--out", str(out)]
    with (out / "run.log").open("w", encoding="utf-8") as log:
        status = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                stderr=subprocess.STDOUT).returncode
    return dataset, out, status


def summarize(finished, out_root):
    fields = ["dataset", "status", "sdca_accuracy", "fdr_svm_accuracy",
              "fdr_minus_sdca", "sdca_stop_cycle", "fdr_stop_cycle",
              "sdca_comm_mb", "fdr_comm_mb"]
    with (out_root / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for dataset, path, status in sorted(finished):
            row = {"dataset": dataset, "status": "ok" if status == 0 else f"failed:{status}"}
            if status == 0:
                sdca = json.loads((path / "sdca_metrics.json").read_text())
                fdr = json.loads((path / "fdr_svm_metrics.json").read_text())
                row.update(sdca_accuracy=sdca["average_accuracy"],
                           fdr_svm_accuracy=fdr["average_accuracy"],
                           fdr_minus_sdca=fdr["average_accuracy"] - sdca["average_accuracy"],
                           sdca_stop_cycle=sdca["stopped_at"],
                           fdr_stop_cycle=fdr["stopped_at"],
                           sdca_comm_mb=sdca["total_comm_bytes"] / 1e6,
                           fdr_comm_mb=fdr["total_comm_bytes"] / 1e6)
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/fdr_comparison"))
    parser.add_argument("--cycles", type=int, default=5000)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--covtype-path", type=Path)
    parser.add_argument("--gisette-path", type=Path)
    parser.add_argument("datasets", nargs="*", choices=ORDER)
    args = parser.parse_args()
    selected = args.datasets or ORDER
    out_root = (ROOT / args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    cpus = os.cpu_count() or 1
    jobs = min(len(selected), max(1, args.jobs))
    blas_threads = max(1, cpus // jobs)
    print(f"datasets={selected} jobs={jobs} blas_threads={blas_threads}", flush=True)
    futures = []
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        for keyword in selected:
            futures.append(pool.submit(run_one, keyword, out_root, args.cycles,
                                       args.data_dir, blas_threads,
                                       args.covtype_path, args.gisette_path))
        finished = []
        for future in as_completed(futures):
            result = future.result()
            finished.append(result)
            print(f"{result[0]}: exit={result[2]}", flush=True)
    summarize(finished, out_root)
    if any(status for _, _, status in finished):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
