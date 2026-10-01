"""One figure across every dataset: accuracy over gossip cycles, SDCA vs BDSVM.

The per-dataset merged figures show four metrics on one dataset; this is the
transposed view — one metric across all datasets — so the pattern in *where*
each method wins is visible on a single page. Each panel is one dataset, both
methods, mean over the 10 workers with a +/- 1 std ribbon, last measured cycle
marked, and the caption under each panel carrying the final numbers.

    python plot_overview.py results/compare_covtype_... results/compare_gisette_... --out results/plots
    python plot_overview.py --from-dir results/all8      # <dir>/<dataset>/{sdca,bdsvm}_metrics.json
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_merged import METHODS, BASELINES, FIGURES, series

ORDER = ["covtype", "gisette", "real-sim", "rcv1", "ijcnn1", "a9a", "w8a", "webspam"]


def human(b):
    for u, s in ((1e12, "TB"), (1e9, "GB"), (1e6, "MB"), (1e3, "KB")):
        if b >= u:
            return f"{b / u:,.1f} {s}"
    return f"{b:,.0f} B"


def load_runs(dirs):
    runs = {}
    for d in dirs:
        d = Path(d)
        data = {}
        for key, _l, _c in METHODS:
            p = d / f"{key}_metrics.json"
            if p.exists():
                data[key] = json.loads(p.read_text())
        if data:
            ds = next(iter(data.values()))["config"]["DATASET"]
            runs[ds] = data
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="*")
    ap.add_argument("--from-dir", default=None,
                    help="directory whose subdirectories each hold one dataset's JSONs")
    ap.add_argument("--out", default="results/plots")
    ap.add_argument("--metric", default="accuracy",
                    choices=["accuracy", "hinge_loss", "comm_bytes"])
    args = ap.parse_args()

    dirs = list(args.dirs)
    if args.from_dir:
        dirs += [p for p in Path(args.from_dir).iterdir() if p.is_dir()]
    runs = load_runs(dirs)
    names = [n for n in ORDER if n in runs] + sorted(set(runs) - set(ORDER))
    if not names:
        raise SystemExit("no runs found")

    # One overview per figure group, as plot_merged does: CoCoA and CoCoA+
    # are separate algorithms and each gets its own page against the baselines.
    for _name, keys, suffix in FIGURES:
        own = [k for k in keys if k not in BASELINES]
        if own and not any(k in runs[ds] for ds in names for k in own):
            continue
        methods = [m for m in METHODS if m[0] in keys]
        draw(names, runs, methods, suffix, args)


def draw(names, runs, methods, suffix, args):
    n = len(names)
    cols = 4 if n > 4 else n
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.4 * rows), squeeze=False)
    log_y = args.metric != "accuracy"
    ylabel = {"accuracy": "Test accuracy", "hinge_loss": "Hinge loss (global)",
              "comm_bytes": "Cumulative bytes sent"}[args.metric]

    for ax, ds in zip(axes.ravel(), names):
        data = runs[ds]
        caption = []
        for m, label, colour in methods:
            if m not in data:
                continue
            s = series(data[m]["per_node_metrics"], args.metric)
            if s is None:
                continue
            x, mean, std = s
            ax.plot(x, mean, color=colour, lw=1.7, label=label)
            ax.fill_between(x, mean - std, mean + std, color=colour, alpha=0.18,
                            linewidth=0)
            ax.plot(x[-1], mean[-1], "o", color=colour, ms=4.5)
            d = data[m]
            stops = [c for c in d["per_node_stop_cycle"] if c is not None]
            caption.append(f"{label.split('-')[1]}: {d['average_accuracy']:.4f} "
                           f"@{max(stops) if stops else '—'}, "
                           f"{human(d['total_comm_bytes'])}")
        ax.set_title(ds, fontsize=11, loc="left", fontweight="bold")
        ax.set_xlabel("Gossip cycle", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        if log_y:
            ax.set_yscale("log")
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.25, linewidth=0.6)
        # One method per line: with four methods a single line is wider than
        # the panel, and tight_layout then squeezes every axis to fit it.
        ax.text(0.5, -0.22, "\n".join(caption), transform=ax.transAxes,
                ha="center", va="top", fontsize=7.4, color="#444444")
    for ax in axes.ravel()[n:]:
        ax.axis("off")

    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False, fontsize=10,
               bbox_to_anchor=(0.99, 0.995))
    fig.suptitle(f"{' vs '.join(l for _m, l, _c in methods)} — {ylabel.lower()} on {n} datasets, "
                 "10 workers, random k-out (k=3), per-node early stopping",
                 fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.02, 1, 0.96), h_pad=3.2)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"overview_{args.metric}{suffix}.png"
    fig.savefig(path, dpi=160)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
