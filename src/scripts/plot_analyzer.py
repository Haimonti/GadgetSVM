"""Read the saved metrics and plot them — individual or comparative, decided here.

This is the single plotting authority. It never re-runs training; it only reads
the canonical metrics files the worker writes to ``results/json/`` ---
``sdca_<dataset>.json``, ``gadget_<dataset>.json`` and
``nonlinear_<dataset>.json``. The *kind* of plot is conditional on how many
models were run (`plot(models, datasets)`):

  * ONE model   -> INDIVIDUAL plots: the full per-run grid for that one model, per
                   dataset, under ``results/individual_plots/``.
  * TWO or more -> COMPARATIVE plots: the models overlaid on shared axes, per
                   dataset, under ``results/comparative_plots/``.

``scripts/main.py`` calls ``plot(...)`` right after training; the ``__main__`` CLI
below lets you re-plot from existing JSON without retraining.

Layout written:

    results/individual_plots/<model>_run<N>_<dataset>_<mm-dd-yyyy>/plots/*.png
    results/comparative_plots/<mm-dd-yyyy>/<dataset>/comp_iter.png
    results/comparative_plots/<mm-dd-yyyy>/<dataset>/comp_time.png
    results/comparative_plots/<mm-dd-yyyy>/<dataset>/summary.json

Usage (standalone re-plot):
    python scripts/plot_analyzer.py                       # all models × all datasets with JSON
    python scripts/plot_analyzer.py -d adult w8a          # these datasets
    python scripts/plot_analyzer.py -m sdca -d gis        # one model -> individual plots
    python scripts/plot_analyzer.py -m sdca nonlinear     # these models -> comparative
"""

import json
import re
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import CODE_DIR
from src.scripts.dataset_keywords import DATASET_KEYWORDS
from src.evaluation.visualizer import (
    plot_loss_vs_time, plot_gap_vs_time, plot_comm_cost_vs_time,
    plot_accuracy_vs_time, plot_std_band,
)

JSON_DIR = CODE_DIR / "results" / "json"
COMP_ROOT = CODE_DIR / "results" / "comparative_plots"
INDIV_ROOT = CODE_DIR / "results" / "individual_plots"

# Models, their plot styling, and the per-model convergence series.
ALGO_LIST = ["sdca", "gadget", "nonlinear", "ndsvm"]
ALGO_STYLE = {
    "sdca":      ("SDCA", "o", "C0"),
    "gadget":    ("GADGET", "s", "C1"),
    "nonlinear": ("Non-linear (DSRR)", "^", "C2"),
    "ndsvm": ("MoM-NDSVM (Forero)", "D", "C3"),
}
# SDCA and the non-linear kernel tenant expose a duality gap; GADGET exposes its
# primal objective. This drives both the comparative convergence panel and the
# individual convergence plot.
CONV_SERIES = {
    "sdca":      ("duality_gap", "SDCA duality gap"),
    "nonlinear": ("duality_gap", "Non-linear duality gap"),
    "gadget":    ("objective",   "GADGET primal objective"),
    "ndsvm": ("objective", "MoM-NDSVM primal objective"),
}


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #
def _resolve_name(token: str) -> str:
    """Accept either a keyword ('adult') or the dataset name ('a9a')."""
    return DATASET_KEYWORDS.get(token.lower(), token)


def _json_path(algo, name) -> Path:
    return JSON_DIR / f"{algo}_{name}.json"


def _load(algo, name):
    return json.loads(_json_path(algo, name).read_text())


def _datasets_available(algos):
    """Map dataset name -> set of the *selected* algos that have a JSON, plus the
    sorted list of names comparable (>= 2 of the selected algorithms present)."""
    present = {}
    for algo in algos:
        for p in JSON_DIR.glob(f"{algo}_*.json"):
            name = p.stem[len(algo) + 1:]
            present.setdefault(name, set()).add(algo)
    comparable = sorted(n for n, a in present.items() if len(a) >= 2)
    return present, comparable


def _mean_by_round(all_metrics, y_key, x_key):
    """Align workers by cycle and average `y_key`; x is the mean `x_key` per round."""
    by_round = {}
    for m in all_metrics:
        for r in m:
            if y_key in r and x_key in r:
                slot = by_round.setdefault(r["round"], {"y": [], "x": []})
                slot["y"].append(r[y_key])
                slot["x"].append(r[x_key])
    rounds = sorted(by_round)
    x = np.array([np.mean(by_round[r]["x"]) for r in rounds])
    y = np.array([np.mean(by_round[r]["y"]) for r in rounds])
    return x, y


def _xlabel(x_key):
    return "Iteration (cycle)" if x_key == "round" else "Wall time (s)"


# --------------------------------------------------------------------------- #
# INDIVIDUAL plots (one model, one dataset) — the full per-run grid
# --------------------------------------------------------------------------- #
def _indiv_run_dir(model: str, ds: str) -> Path:
    """results/individual_plots/<model>_run<N>_<dataset>_<mm-dd-yyyy>."""
    INDIV_ROOT.mkdir(parents=True, exist_ok=True)
    nums = [int(m.group(1)) for p in INDIV_ROOT.iterdir() if p.is_dir()
            if (m := re.match(rf"{model}_run(\d+)_", p.name))]
    n = max(nums, default=0) + 1
    return INDIV_ROOT / f"{model}_run{n}_{ds}_{datetime.now().strftime('%m-%d-%Y')}"


def individual_plots(model: str, name: str) -> None:
    """The per-run grid for a single model+dataset, read from its saved JSON. The
    convergence panel is the duality gap (SDCA / non-linear) or the primal
    objective (GADGET)."""
    all_metrics = _load(model, name)["metrics"]
    conv_key = CONV_SERIES[model][0]
    plots = _indiv_run_dir(model, name) / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    for x_key, suffix in (("wall_time", "vs_time"), ("round", "vs_iterations")):
        plot_loss_vs_time(all_metrics,      plots / f"loss_{suffix}.png",      x_key=x_key)
        plot_accuracy_vs_time(all_metrics,  plots / f"accuracy_{suffix}.png",  x_key=x_key)
        plot_comm_cost_vs_time(all_metrics, plots / f"comm_cost_{suffix}.png", x_key=x_key)
        if conv_key == "duality_gap":
            plot_gap_vs_time(all_metrics, plots / f"duality_gap_{suffix}.png", x_key=x_key)
        else:  # objective: no dedicated gap plot — draw the objective as a band line
            axis = "vs time" if x_key == "wall_time" else "vs iterations"
            plot_std_band(all_metrics, plots / f"objective_{suffix}.png", "objective",
                          f"Primal Objective ({axis})", "Primal objective",
                          log_y=True, x_key=x_key)

    conv_name   = "Duality Gap" if conv_key == "duality_gap" else "Primal Objective"
    conv_ylabel = "Duality gap" if conv_key == "duality_gap" else "Primal objective"
    conv_base   = f"{conv_key}_std_band" if conv_key == "duality_gap" else "objective_std_band"
    std_band_specs = [
        (conv_key,     conv_base,            conv_name,            conv_ylabel,             True),
        ("hinge_loss", "loss_std_band",      "Hinge Loss",         "Hinge loss",            True),
        ("accuracy",   "accuracy_std_band",  "Test Accuracy",      "Test accuracy",         False),
        ("comm_bytes", "comm_cost_std_band", "Communication Cost", "Cumulative bytes sent", False),
    ]
    for x_key, suffix in (("round", "vs_iterations"), ("wall_time", "vs_time")):
        for y_key, base, nm, ylabel, log_y in std_band_specs:
            plot_std_band(all_metrics, plots / f"{base}_{suffix}.png",
                          y_key, f"{nm} — Mean ±1σ Across Workers",
                          ylabel, log_y=log_y, x_key=x_key)
    print(f"  [{name}] {model} -> individual plots → {plots}")


# --------------------------------------------------------------------------- #
# COMPARATIVE plots (>=2 models overlaid, one dataset)
# --------------------------------------------------------------------------- #
def _overlay_panel(ax, datasets, y_key, x_key, title, ylabel, log_y=False):
    for algo in ALGO_LIST:
        data = datasets.get(algo)
        if data is None:
            continue
        label, marker, color = ALGO_STYLE[algo]
        xs, ys = _mean_by_round(data["metrics"], y_key, x_key)
        if len(xs):
            ax.plot(xs, ys, marker=marker, markersize=3, color=color, label=label)
    ax.set(title=title, xlabel=_xlabel(x_key), ylabel=ylabel)
    ax.grid(True, alpha=0.3)
    if log_y:
        ax.set_yscale("log")
    ax.legend(fontsize=8)


def _convergence_panel(ax, datasets, x_key):
    for algo in ALGO_LIST:
        data = datasets.get(algo)
        if data is None:
            continue
        y_key, label = CONV_SERIES[algo]
        _, marker, color = ALGO_STYLE[algo]
        xs, ys = _mean_by_round(data["metrics"], y_key, x_key)
        if len(xs):
            ax.plot(xs, ys, marker=marker, markersize=3, color=color, label=label)
    ax.set(title="Convergence (own criterion)", xlabel=_xlabel(x_key), ylabel="gap / objective")
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    ax.legend(fontsize=8)


def _grid(datasets, name, x_key, out_path):
    """One 2x2 image: accuracy | hinge loss / comm cost | convergence, for one x-axis.
    Overlays whichever of SDCA / GADGET / Non-linear are present for this dataset."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    _overlay_panel(axes[0, 0], datasets, "accuracy", x_key, "Test Accuracy", "Test accuracy")
    _overlay_panel(axes[0, 1], datasets, "hinge_loss", x_key, "Hinge Loss", "Hinge loss", log_y=True)
    _overlay_panel(axes[1, 0], datasets, "comm_bytes", x_key, "Communication Cost", "Cumulative bytes / node")
    _convergence_panel(axes[1, 1], datasets, x_key)
    axis = "vs iterations" if x_key == "round" else "vs wall-clock time"
    algos = " vs ".join(ALGO_STYLE[a][0] for a in ALGO_LIST if a in datasets)
    fig.suptitle(f"{algos}  [dataset: {name}]  ({axis})", fontsize=14)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def _final(all_metrics, key, reduce=np.mean):
    vals = [m[-1][key] for m in all_metrics if m and key in m[-1]]
    return float(reduce(vals)) if vals else None


def comparative_plots(name, algos_present, date_dir):
    """Overlay the given models (those with a JSON for `name`) on shared axes."""
    datasets = {algo: _load(algo, name) for algo in algos_present}
    ds_dir = date_dir / name
    _grid(datasets, name, "round", ds_dir / "comp_iter.png")
    _grid(datasets, name, "wall_time", ds_dir / "comp_time.png")
    summary = {"dataset": name}
    for algo, data in datasets.items():
        conv_key = CONV_SERIES[algo][0]
        summary[algo] = {
            "final_mean_accuracy": _final(data["metrics"], "accuracy"),
            f"final_mean_{conv_key}": _final(data["metrics"], conv_key),
            "total_comm_bytes": data.get("total_comm_bytes"),
            "final_round": max((m[-1]["round"] for m in data["metrics"] if m), default=None),
        }
    (ds_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    accs = "  ".join(f"{a}={summary[a]['final_mean_accuracy']}" for a in datasets)
    print(f"  [{name}]  acc {accs}  -> {ds_dir}")


# --------------------------------------------------------------------------- #
# the one entry point: decide individual vs comparative by model count
# --------------------------------------------------------------------------- #
def plot(models, datasets) -> None:
    """Plot the results of a run. `models`/`datasets` are what was *chosen*;
    one model -> individual plots, two or more -> comparative plots. Datasets
    may be keywords or names; missing JSONs are skipped with a message."""
    if not JSON_DIR.exists():
        raise SystemExit(f"No metrics yet: {JSON_DIR} does not exist. Run scripts/main.py first.")

    names = [_resolve_name(d) for d in datasets]

    if len(models) == 1:                       # ---- INDIVIDUAL ----
        model = models[0]
        print(f"Individual plots: {model} on {names}")
        for name in names:
            if _json_path(model, name).exists():
                individual_plots(model, name)
            else:
                print(f"SKIP '{name}': no {model} JSON (did that leg run?)")
    else:                                       # ---- COMPARATIVE ----
        present, _ = _datasets_available(models)
        date_dir = COMP_ROOT / datetime.now().strftime("%m-%d-%Y")
        print(f"Comparative plots: {models} on {names}  ->  {date_dir}")
        for name in names:
            have = sorted(present.get(name, set()))
            if len(have) >= 2:
                comparative_plots(name, have, date_dir)
            else:
                print(f"SKIP '{name}': needs >=2 of {models} (have: {have or 'none'})")
    print("Done.")


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="Plot saved metrics: individual (one model) or comparative (two or more).")
    ap.add_argument("-d", "--datasets", nargs="+", default=["all"], metavar="DATASET",
                    help="dataset keywords or names (default: all with JSON present)")
    ap.add_argument("-m", "--models", "--algos", nargs="+", dest="models",
                    default=["all"], metavar="MODEL",
                    help="models to plot: sdca gadget nonlinear ndsvm or 'all' (default: all)")
    args = ap.parse_args()

    models = list(ALGO_LIST) if "all" in args.models else args.models
    bad = [m for m in models if m not in ALGO_LIST]
    if bad:
        raise SystemExit(f"Unknown model(s): {bad}. Choose from: {' '.join(ALGO_LIST)}")

    if "all" in args.datasets:
        # every dataset that has a JSON for at least one of the chosen models
        present, _ = _datasets_available(models)
        datasets = sorted(present)
        if not datasets:
            raise SystemExit(f"No JSON found for models {models} in {JSON_DIR}.")
    else:
        datasets = args.datasets

    plot(models, datasets)


if __name__ == "__main__":
    main()
