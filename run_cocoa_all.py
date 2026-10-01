"""Run P2P-CoCoA and P2P-CoCoA+ on every dataset, next to the existing results.

For each dataset keyword, finds the latest `results/compare_<dataset>_*`
directory that already holds the SDCA / BDSVM / FedAvg runs, writes
`cocoa_metrics.json` and `cocoa_plus_metrics.json` into that same directory
(same CONFIG, same seed, same shards as those runs), and redraws its figures.
Finishes with the cross-dataset overview pages.

    python run_cocoa_all.py                     # all 8 datasets
    python run_cocoa_all.py cov gis             # a subset
    python run_cocoa_all.py --cycles 20         # quick test (writes to a scratch dir)

Resumable: a dataset whose directory already has both CoCoA JSONs is skipped.
"""

import argparse
import subprocess
import sys
from pathlib import Path

from run_compare import DATASET_KEYWORDS

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
ORDER = ["cov", "gis", "rsim", "rcv", "ijcnn", "adult", "w8a", "webspam"]
BASELINE_FILES = ("sdca_metrics.json", "bdsvm_metrics.json", "fedavg_metrics.json")


def existing_run_dir(dataset):
    """Latest compare_<dataset>_* directory holding all three baseline runs."""
    cands = [d for d in RESULTS.glob(f"compare_{dataset}_*")
             if d.is_dir() and all((d / f).exists() for f in BASELINE_FILES)]
    return max(cands, key=lambda d: d.stat().st_mtime) if cands else None


def main():
    ap = argparse.ArgumentParser()
    # default=None, not ORDER: argparse checks a list default against
    # `choices` as one value and rejects it.
    ap.add_argument("datasets", nargs="*", default=None,
                    choices=sorted(DATASET_KEYWORDS))
    ap.add_argument("--cycles", type=int, default=None,
                    help="cap on cycles; when set, results go to a scratch "
                         "directory instead of next to the real runs")
    args = ap.parse_args()
    args.datasets = args.datasets or ORDER

    done_dirs = []
    for kw in args.datasets:
        ds = DATASET_KEYWORDS[kw]
        if args.cycles is not None:
            out = RESULTS / f"cocoa_test_{ds}"
        else:
            out = existing_run_dir(ds)
            if out is None:
                out = RESULTS / f"compare_{ds}_cocoa"
                print(f"[{kw}] no earlier SDCA/BDSVM/FedAvg run found -> {out}")
        if (args.cycles is None and (out / "cocoa_metrics.json").exists()
                and (out / "cocoa_plus_metrics.json").exists()):
            print(f"[{kw}] already done in {out}, skipping")
            done_dirs.append(out)
            continue

        print(f"\n===== [{kw}] {ds} -> {out} =====", flush=True)
        cmd = [sys.executable, "run_compare.py", kw,
               "--only", "cocoa", "cocoa_plus", "--out", str(out)]
        if args.cycles is not None:
            cmd += ["--cycles", str(args.cycles)]
        if subprocess.run(cmd, cwd=ROOT).returncode != 0:
            print(f"[{kw}] FAILED — see the output above; continuing")
            continue
        subprocess.run([sys.executable, "plot_merged.py", str(out)], cwd=ROOT)
        done_dirs.append(out)

    if done_dirs:
        for metric in ("accuracy", "hinge_loss", "comm_bytes"):
            subprocess.run([sys.executable, "plot_overview.py",
                            *map(str, done_dirs), "--metric", metric,
                            "--out", str(RESULTS / "plots_cocoa")], cwd=ROOT)
        print("\nPer-dataset figures: <run dir>/plots/merged_with_cocoa*.png")
        print(f"Overview figures:    {RESULTS / 'plots_cocoa'}/overview_*_cocoa*.png")
        for d in done_dirs:
            print(f"  {d}")


if __name__ == "__main__":
    main()
