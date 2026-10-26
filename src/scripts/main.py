"""The single entry point: run any models on any datasets, then plot.

    python scripts/main.py [-m MODELS...] [-d DATASETS...] [--no-plots]

      -m / --models    one or more of: sdca gadget nonlinear ndsvm (or 'all'; default all)
      -d / --datasets  one or more dataset keywords             (or 'all'; default all)
      --no-plots       train + save JSON only, skip plotting

This file only *chooses what to run*. It trains every chosen (model, dataset)
combination via `scripts/model_picker.py` (which writes `results/json/*.json`), then
hands off once to `scripts/plot_analyzer.py`, which decides the plot kind from the
number of models: ONE model -> individual plots; TWO or more -> comparative
plots. A failing leg (download hiccup, or the kernel tenant running out of memory
on a large dataset) is logged and skipped rather than aborting the whole sweep.

    python scripts/main.py                              # all models × all datasets -> comparative
    python scripts/main.py -m sdca -d all               # one model, every dataset -> individual
    python scripts/main.py -m all -d gis                # all models, one dataset -> comparative
    python scripts/main.py -m sdca nonlinear -d gis w8a # subset × subset -> comparative
    python scripts/main.py -m gadget -d a9a --no-plots  # just train, no plots
"""

import sys
from pathlib import Path

# Ensure the repo root is importable so `src.*` and root-level `data.*` resolve.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.scripts.model_picker import MODELS, run_model
from src.peersim_python.logger import logger


def main():
    import argparse
    from src.scripts.dataset_keywords import DATASET_KEYWORDS
    from src.scripts import plot_analyzer

    ap = argparse.ArgumentParser(
        description="Run selected models on selected datasets, then plot "
                    "(individual for one model, comparative for two or more).")
    ap.add_argument("-m", "--models", "--algos", nargs="+", dest="models",
                    default=["all"], metavar="MODEL",
                    help="models to run: sdca gadget nonlinear ndsvm or 'all' (default: all)")
    ap.add_argument("-d", "--datasets", nargs="+", default=["all"], metavar="DATASET",
                    help="dataset keywords or 'all' (default: all)")
    ap.add_argument("--no-plots", action="store_true",
                    help="train + save JSON only, skip plotting")
    args = ap.parse_args()

    models = list(MODELS) if "all" in args.models else args.models
    bad = [m for m in models if m not in MODELS]
    if bad:
        raise SystemExit(f"Unknown model(s): {bad}. Choose from: {' '.join(MODELS)}")

    datasets = list(DATASET_KEYWORDS) if "all" in args.datasets else args.datasets
    bad = [d for d in datasets if d.lower() not in DATASET_KEYWORDS]
    if bad:
        raise SystemExit(f"Unknown dataset keyword(s): {bad}. "
                         f"Choose from: {' '.join(DATASET_KEYWORDS)}")

    mode = "INDIVIDUAL" if len(models) == 1 else "COMPARATIVE"
    logger.info("main", f"{mode} run: models={models}  datasets={datasets}")

    outcomes = {}
    for ds in datasets:
        for model in models:
            try:
                run_model(model, ds)
                outcomes[(model, ds)] = True
            except Exception as e:                       # noqa: BLE001 — one leg must not kill the sweep
                outcomes[(model, ds)] = False
                logger.info("main", f"!!! {model} '{ds}' FAILED ({e}) — continuing")

    logger.info("main", "=== run summary ===")
    for ds in datasets:
        for model in models:
            logger.info("main", f"  {model:10s} {ds:8s} "
                                f"{'ok' if outcomes[(model, ds)] else 'FAILED'}")

    if args.no_plots:
        logger.info("main", "Done (no plots requested).")
        return
    plot_analyzer.plot(models, datasets)


if __name__ == "__main__":
    main()
