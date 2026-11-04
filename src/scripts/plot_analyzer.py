"""Read the saved metrics and plot them — individual or comparative, decided here.

This is the single plotting authority. It never re-runs training; it only reads
the canonical metrics files the worker writes to ``results/json/``
(``<model>_<dataset>_seed<k>.json``). The *kind* of plot is conditional on how many
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
    results/comparative_plots/<mm-dd-yyyy>/<dataset>/comp_bars.png
    results/comparative_plots/<mm-dd-yyyy>/<dataset>/summary.json

Every (model, dataset) may have several seeds (``<model>_<dataset>_seed<k>.json``):
comparative curves are the mean ± std ACROSS SEEDS of each seed's node-average,
and the bar chart / summary report mean ± std across seeds.

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

# Models, their plot styling, group, and the per-model convergence series.
ALGO_LIST = ["central_sdca", "sdca", "gadget", "cola", "choco", "nonlinear", "ndsvm", "fed_ksvm"]
ALGO_STYLE = {
    "central_sdca": ("Centralized SDCA", "*", "k"),
    "sdca":         ("SDCA (ours)", "o", "C0"),
    "gadget":       ("GADGET", "s", "C1"),
    "cola":         ("COLA", "v", "C4"),
    "choco":        ("CHOCO-SGD", "P", "C5"),
    "nonlinear":    ("Non-linear (DSRR)", "^", "C2"),
    "ndsvm":        ("MoM-NDSVM (Forero)", "D", "C3"),
    "fed_ksvm":     ("Fed-KSVM", "X", "C6"),
}
# Plot grouping asked for in the 10/26 meeting: P2P/gossip vs federated algorithms
# (plus the centralized reference). Legends list models grouped in this order.
ALGO_GROUP = {
    "central_sdca": "Centralized", "sdca": "P2P / gossip", "gadget": "P2P / gossip",
    "cola": "P2P / gossip", "choco": "P2P / gossip", "nonlinear": "P2P / gossip",
    "ndsvm": "P2P / gossip", "fed_ksvm": "Federated",
}
GROUP_ORDER = ["P2P / gossip", "Federated", "Centralized"]
LINEAR = {"central_sdca", "sdca", "gadget", "cola", "choco"}
# Dual methods expose a duality gap; primal-only methods (GADGET, CHOCO) their
# primal objective. Linear models additionally get a common suboptimality panel
# P(w) - P*, with P* from centralized SDCA when its JSON exists.
CONV_SERIES = {
    "central_sdca": ("duality_gap", "Centralized SDCA duality gap"),
    "sdca":         ("duality_gap", "SDCA duality gap"),
    "cola":         ("duality_gap", "COLA duality gap"),
    "nonlinear":    ("duality_gap", "Non-linear duality gap"),
    "gadget":       ("objective",   "GADGET primal objective"),
    "choco":        ("objective",   "CHOCO-SGD primal objective"),
    "ndsvm":        ("objective",   "MoM-NDSVM primal objective"),
    "fed_ksvm":     ("objective",   "Fed-KSVM primal objective"),
}


def _ordered(algos):
    """Models in legend order: by group, then by ALGO_LIST position."""
    return sorted(algos, key=lambda a: (GROUP_ORDER.index(ALGO_GROUP.get(a, "P2P / gossip")),
                                        ALGO_LIST.index(a)))


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #
def _resolve_name(token: str) -> str:
    """Accept either a keyword ('adult') or the dataset name ('a9a')."""
    return DATASET_KEYWORDS.get(token.lower(), token)


def _seed_files(algo, name):
    """Every saved run of (algo, dataset): one file per seed, plus a legacy
    seedless `<algo>_<name>.json` if present."""
    files = sorted(JSON_DIR.glob(f"{algo}_{name}_seed*.json"))
    legacy = JSON_DIR / f"{algo}_{name}.json"
    return files + ([legacy] if legacy.exists() else [])


def _has_json(algo, name) -> bool:
    return bool(_seed_files(algo, name))


def _load(algo, name):
    """All seeds of (algo, dataset) merged: `metrics` pools every seed's per-node
    lists (so curves average over nodes x seeds), `runs` keeps each seed's JSON."""
    runs = [json.loads(p.read_text()) for p in _seed_files(algo, name)]
    return {
        "metrics": [m for r in runs for m in r["metrics"]],
        "runs": runs,
        "seeds": [r.get("seed") for r in runs],
        "total_comm_bytes": float(np.mean([r["total_comm_bytes"] for r in runs])),
    }


def _datasets_available(algos):
    """Map dataset name -> set of the *selected* algos that have a JSON, plus the
    sorted list of names comparable (>= 2 of the selected algorithms present)."""
    present = {}
    for algo in algos:
        for p in JSON_DIR.glob(f"{algo}_*.json"):
            name = re.sub(r"_seed\d+$", "", p.stem[len(algo) + 1:])
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
def _per_round(metrics, y_key, x_key):
    """{round: (mean x, mean y)} over the nodes of ONE run."""
    acc = {}
    for m in metrics:
        for r in m:
            if r.get(y_key) is not None and x_key in r:
                slot = acc.setdefault(r["round"], ([], []))
                slot[0].append(r[x_key])
                slot[1].append(r[y_key])
    return {k: (float(np.mean(v[0])), float(np.mean(v[1]))) for k, v in acc.items()}


def _seed_band(data, y_key, x_key, transform=None):
    """Mean and std ACROSS SEEDS of each seed's node-mean curve, aligned by round."""
    per_seed = [_per_round(r["metrics"], y_key, x_key) for r in data["runs"]]
    rounds = sorted(set().union(*per_seed)) if per_seed else []
    xs, mu, sd = [], [], []
    for rd in rounds:
        pts = [ps[rd] for ps in per_seed if rd in ps]
        ys = np.array([p[1] for p in pts])
        if transform:
            ys = transform(ys)
        xs.append(np.mean([p[0] for p in pts]))
        mu.append(float(np.mean(ys)))
        sd.append(float(np.std(ys)))
    return np.array(xs), np.array(mu), np.array(sd)


def log_safe_band(mu, sd):
    """For a log y-axis: values <= 0 (e.g. a hinge loss of exactly 0) become gaps
    instead of plunging to the axis floor, and the lower edge of the std band is
    kept positive (at most one decade below the mean) so it never draws a spike."""
    mu = np.where(mu > 0, mu, np.nan)
    lo = mu - sd
    return mu, np.where(lo > 0, np.maximum(lo, mu / 10.0), mu / 10.0), mu + sd


def _draw(ax, algo, xs, mu, sd, label=None, log_y=False):
    name, marker, color = ALGO_STYLE[algo]
    if not len(xs):
        return
    if log_y:
        mu, lo, hi = log_safe_band(mu, sd)
    else:
        lo, hi = mu - sd, mu + sd
    ax.plot(xs, mu, marker=marker, markersize=2, linewidth=1.2, color=color, label=label or name)
    ax.fill_between(xs, lo, hi, color=color, alpha=0.18, linewidth=0)


def _finish(ax, title, x_key, ylabel, log_y):
    """Log x-axis: models stop after very different numbers of rounds (centralized
    SDCA ~20, GADGET ~250, SDCA ~1k, COLA ~10k, CHOCO 50k), so a linear axis
    squashes all but the longest run against the left edge."""
    ax.set(title=title, xlabel=_xlabel(x_key) + " (log scale)", ylabel=ylabel)
    ax.grid(True, which="both", alpha=0.25)
    ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.legend(fontsize=8)


def _overlay_panel(ax, datasets, y_key, x_key, title, ylabel, log_y=False):
    for algo in _ordered(datasets):
        _draw(ax, algo, *_seed_band(datasets[algo], y_key, x_key), log_y=log_y)
    _finish(ax, title + " (mean ± std over seeds)", x_key, ylabel, log_y)


def _convergence_panel(ax, datasets, x_key):
    for algo in _ordered(datasets):
        y_key, label = CONV_SERIES[algo]
        _draw(ax, algo, *_seed_band(datasets[algo], y_key, x_key), label=label, log_y=True)
    _finish(ax, "Convergence (own criterion)", x_key, "gap / objective", True)


def _p_star(datasets):
    """Reference optimum P*: liblinear's exact solve stored by central_sdca (or, if
    absent, centralized SDCA's best final primal)."""
    c = datasets.get("central_sdca")
    if not c:
        return None
    exact = [r["p_star"] for r in c["runs"] if r.get("p_star") is not None]   # liblinear
    return min(exact + [m[-1]["primal"] for m in c["metrics"] if m])


def _suboptimality_panel(ax, datasets, x_key, p_star):
    for algo in _ordered(a for a in datasets if a in LINEAR and a != "central_sdca"):
        key = "primal" if algo != "gadget" else "objective"
        _draw(ax, algo, *_seed_band(datasets[algo], key, x_key,
                                    transform=lambda v: np.maximum(v - p_star, 1e-16)), log_y=True)
    _finish(ax, f"Suboptimality P(w) - P*  (P* = {p_star:.6g}, centralized SDCA)", x_key,
            "P(w) - P*", True)


def _grid(datasets, name, x_key, out_path):
    """accuracy | hinge / comm | convergence, plus suboptimality when P* is known."""
    p_star = _p_star(datasets)
    linear = [a for a in datasets if a in LINEAR and a != "central_sdca"]
    rows = 3 if (p_star is not None and linear) else 2
    fig, axes = plt.subplots(rows, 2, figsize=(14, 5 * rows))
    _overlay_panel(axes[0, 0], datasets, "accuracy", x_key, "Test Accuracy", "Test accuracy")
    _overlay_panel(axes[0, 1], datasets, "hinge_loss", x_key, "Hinge Loss", "Hinge loss", log_y=True)
    _overlay_panel(axes[1, 0], datasets, "comm_bytes", x_key, "Communication Cost", "Cumulative bytes / node",
                   log_y=True)
    _convergence_panel(axes[1, 1], datasets, x_key)
    if rows == 3:
        _suboptimality_panel(axes[2, 0], datasets, x_key, p_star)
        axes[2, 1].axis("off")
    axis = "vs iterations" if x_key == "round" else "vs wall-clock time"
    groups = {}
    for a in _ordered(datasets):
        groups.setdefault(ALGO_GROUP.get(a, "P2P / gossip"), []).append(ALGO_STYLE[a][0])
    title = "   |   ".join(f"{g}: {', '.join(ms)}" for g, ms in groups.items())
    fig.suptitle(f"{title}  [dataset: {name}]  ({axis})", fontsize=13)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def _final(all_metrics, key, reduce=np.mean):
    vals = [m[-1][key] for m in all_metrics if m and m[-1].get(key) is not None]
    return float(reduce(vals)) if vals else None


def _seed_stats(data, fn):
    """mean/std across seeds of a per-run scalar fn(run_json)."""
    vals = [v for v in (fn(r) for r in data["runs"]) if v is not None]
    if not vals:
        return None
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n_seeds": len(vals)}


BAR_METRICS = [
    ("final_accuracy",   "Final test accuracy",       lambda r: _final(r["metrics"], "accuracy")),
    ("final_hinge_loss", "Final hinge loss",          lambda r: _final(r["metrics"], "hinge_loss")),
    ("total_comm_bytes", "Total bytes sent (all nodes)", lambda r: r.get("total_comm_bytes")),
    ("rounds",           "Rounds run",                lambda r: max((m[-1]["round"] for m in r["metrics"] if m), default=None)),
]


def _bars(stats, name, out_path):
    """One bar per model per metric, std bars across seeds, colored by model,
    models ordered by group (gossip | federated | centralized)."""
    algos = _ordered(stats)
    fig, axes = plt.subplots(1, len(BAR_METRICS), figsize=(4.2 * len(BAR_METRICS), 4.5))
    for ax, (key, title, _) in zip(axes, BAR_METRICS):
        present = [a for a in algos if stats[a].get(key)]
        mu = [stats[a][key]["mean"] for a in present]
        sd = [stats[a][key]["std"] for a in present]
        ax.bar(range(len(present)), mu, yerr=sd, capsize=4,
               color=[ALGO_STYLE[a][2] for a in present])
        ax.set_xticks(range(len(present)))
        ax.set_xticklabels([ALGO_STYLE[a][0] for a in present], rotation=35, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
        if key == "total_comm_bytes":
            ax.set_yscale("log")
    fig.suptitle(f"Mean ± std over seeds  [dataset: {name}]", fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def comparative_plots(name, algos_present, date_dir):
    """Overlay the given models (those with a JSON for `name`) on shared axes."""
    datasets = {algo: _load(algo, name) for algo in algos_present}
    ds_dir = date_dir / name
    _grid(datasets, name, "round", ds_dir / "comp_iter.png")
    _grid(datasets, name, "wall_time", ds_dir / "comp_time.png")
    summary = {"dataset": name, "p_star": _p_star(datasets), "models": {}}
    for algo, data in datasets.items():
        entry = {"group": ALGO_GROUP.get(algo), "seeds": data["seeds"]}
        for key, _, fn in BAR_METRICS:
            entry[key] = _seed_stats(data, fn)
        conv_key = CONV_SERIES[algo][0]
        entry[f"final_{conv_key}"] = _seed_stats(data, lambda r, k=conv_key: _final(r["metrics"], k))
        summary["models"][algo] = entry
    ds_dir.mkdir(parents=True, exist_ok=True)
    _bars(summary["models"], name, ds_dir / "comp_bars.png")
    (ds_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    accs = "  ".join(f"{a}={(summary['models'][a]['final_accuracy'] or {}).get('mean')}"
                     for a in _ordered(datasets))
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
            if _has_json(model, name):
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
                    help=f"models to plot: {' '.join(ALGO_LIST)} or 'all' (default: all)")
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
