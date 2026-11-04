"""One command for the full experiment suite.

    python src/scripts/run_all.py                                   # gisette, a9a, w8a, ijcnn1 · 5 seeds · everything
    python src/scripts/run_all.py -d adult --seeds 0 1              # one dataset, 2 seeds
    python src/scripts/run_all.py --skip-ablations                  # main comparisons only
    python src/scripts/run_all.py -a topology quantization          # only these ablations
    python src/scripts/run_all.py --set ROUNDS=20000 EVAL_EVERY=50  # overrides for every run

For every dataset it runs, in order:
  1. LINEAR comparison     — sdca, central_sdca, gadget, cola, choco   -> comparative plots
  2. NONLINEAR comparison  — nonlinear, ndsvm (Forero), fed_ksvm       -> comparative plots
  3. ABLATIONS (linear)    — every ablation in ablations.py            -> results/ablations/...
Linear and nonlinear models are plotted separately (never mixed in one figure).
A failing leg is logged and skipped; a summary is printed at the end.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.peersim_python.logger import logger
from src.scripts.model_picker import run_model, parse_overrides
from src.scripts.ablations import ABLATIONS, LINEAR_MODELS
from src.scripts.dataset_keywords import DATASET_KEYWORDS

NONLINEAR_MODELS = ["nonlinear", "ndsvm", "fed_ksvm"]
DEFAULT_DATASETS = ["gis", "adult", "w8a", "ijcnn"]      # gisette, a9a, w8a, ijcnn1


def _run_group(name, models, datasets, seeds, overrides, summary):
    import json
    for ds in datasets:
        for model in models:
            for seed in seeds:
                key = (name, ds, model, seed)
                t0 = time.time()
                try:
                    out = run_model(model, ds, overrides={**overrides, "SEED": seed})
                    info = json.loads(Path(out).read_text())
                    summary[key] = (f"{info.get('stop_reason', '?')} after {info.get('rounds_run')}"
                                    f"/{info.get('round_budget')} rounds ({time.time() - t0:.0f}s)")
                except Exception as e:                 # noqa: BLE001 — one leg must not kill the suite
                    summary[key] = f"FAILED: {e}"
                    logger.info("run_all", f"!!! {name} {model} {ds} seed {seed} FAILED ({e}) — continuing")


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Run linear + nonlinear comparisons and ablations.")
    ap.add_argument("-d", "--datasets", nargs="+", default=DEFAULT_DATASETS,
                    help=f"dataset keywords (default: {' '.join(DEFAULT_DATASETS)})")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("-a", "--ablations", nargs="+", default=["all"],
                    help=f"ablations to run: {' '.join(ABLATIONS)} or 'all' (default)")
    ap.add_argument("--set", nargs="+", default=[], dest="overrides", metavar="KEY=VAL",
                    help="CONFIG overrides applied to every run")
    ap.add_argument("--skip-linear", action="store_true")
    ap.add_argument("--skip-nonlinear", action="store_true")
    ap.add_argument("--skip-ablations", action="store_true")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    bad = [d for d in args.datasets if d.lower() not in DATASET_KEYWORDS]
    abl = list(ABLATIONS) if "all" in args.ablations else args.ablations
    bad += [a for a in abl if a not in ABLATIONS]
    if bad:
        raise SystemExit(f"unknown dataset/ablation: {bad}")
    overrides = parse_overrides(args.overrides)
    summary = {}
    t_start = time.time()

    from src.scripts import plot_analyzer
    if not args.skip_linear:
        logger.info("run_all", f"=== LINEAR comparison: {LINEAR_MODELS} on {args.datasets} ===")
        _run_group("linear", LINEAR_MODELS, args.datasets, args.seeds, overrides, summary)
        if not args.no_plots:
            plot_analyzer.plot(LINEAR_MODELS, args.datasets)
    if not args.skip_nonlinear:
        logger.info("run_all", f"=== NONLINEAR comparison: {NONLINEAR_MODELS} on {args.datasets} ===")
        _run_group("nonlinear", NONLINEAR_MODELS, args.datasets, args.seeds, overrides, summary)
        if not args.no_plots:
            plot_analyzer.plot(NONLINEAR_MODELS, args.datasets)
    if not args.skip_ablations:
        from src.scripts.ablation_runner import run_ablation
        for ds in args.datasets:
            for name in abl:
                logger.info("run_all", f"=== ABLATION {name} on {ds} ===")
                try:
                    out = run_ablation(name, ds, LINEAR_MODELS, args.seeds, overrides)
                    summary[("ablation", ds, name, "all")] = f"ok -> {out}"
                except Exception as e:                 # noqa: BLE001
                    summary[("ablation", ds, name, "all")] = f"FAILED: {e}"

    logger.info("run_all", f"=== summary ({(time.time() - t_start) / 60:.1f} min) ===")
    for (group, ds, what, seed), status in summary.items():
        logger.info("run_all", f"  {group:9s} {ds:6s} {what:13s} seed={seed!s:4s} {status}")
    not_conv = [k for k, v in summary.items() if v.startswith("budget")]
    failed = [k for k, v in summary.items() if v.startswith("FAILED")]
    if not_conv:
        logger.info("run_all", f"!!! {len(not_conv)} run(s) used the whole round budget without "
                               f"converging: {not_conv} — rerun them with --set ROUNDS=<larger>")
    if failed:
        logger.info("run_all", f"!!! {len(failed)} run(s) failed: {failed}")


if __name__ == "__main__":
    main()
