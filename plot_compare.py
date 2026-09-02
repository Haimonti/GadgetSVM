"""Cross-method comparison figures from the history CSVs.

plot_history.py answers "how did the nodes of one run behave"; this answers
"how do the methods compare". One line per method — the mean over nodes, with a
shaded band for the node spread, so a method whose nodes disagree is visibly
different from one whose nodes agree even at the same mean.

    python plot_compare.py results/history --out results/plots

Produces per scheme:
    compare_<scheme>_accuracy.png
    compare_<scheme>_loss.png
and a single panel putting every scheme side by side:
    compare_grid_accuracy.png / compare_grid_loss.png
"""
import argparse
import csv
import glob
import math
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Fixed order and colour per method so every figure reads the same way.
METHODS = ["bdsvm", "fdr_svm", "cocoa", "cocoa_plus", "fedavg_svm", "fedssl_amc"]
COLOR = dict(zip(METHODS, ["#17595E", "#B4762A", "#2E6B46", "#7A4E9E",
                           "#A33A2C", "#4A5360"]))
SCHEME_ORDER = ["iid", "dirichlet_1.0", "dirichlet_0.3", "dirichlet_0.1", "label_skew"]
SERIES = [("test_acc", "Test accuracy", "accuracy"),
          ("hinge_loss", "Hinge loss", "loss")]


def fnum(row, col):
    try:
        return float(row[col])
    except (KeyError, TypeError, ValueError):
        return math.nan


def load(paths):
    """{(scheme, method): {node: [rows]}} across every history file given."""
    out = defaultdict(lambda: defaultdict(list))
    for p in paths:
        with open(p) as fh:
            for r in csv.DictReader(fh):
                out[(r["scheme"], r["method"])][int(r["node"])].append(r)
    return out


def band(nodes, col):
    """Rounds, mean over nodes, and min/max envelope."""
    series = []
    for rows in nodes.values():
        rows = sorted(rows, key=lambda r: int(r["round"]))
        x = np.array([int(r["round"]) for r in rows])
        y = np.array([fnum(r, col) for r in rows])
        m = ~np.isnan(y)
        if m.any():
            series.append((x[m], y[m]))
    if not series:
        return None
    common = sorted(set.intersection(*(set(x.tolist()) for x, _ in series)))
    if not common:
        return None
    cx = np.array(common)
    ys = np.stack([np.interp(cx, x, y) for x, y in series])
    return cx, ys.mean(axis=0), ys.min(axis=0), ys.max(axis=0)


def all_positive(data, schemes, col):
    """True only if every value across every scheme is > 0.

    The grid shares one y axis, so the scale has to be decided once for all
    panels; and label_skew drives hinge loss to exactly 0, which a log axis
    would silently drop.
    """
    vals = []
    for s in schemes:
        for m in METHODS:
            b = band(data.get((s, m), {}), col)
            if b is not None:
                vals.append(b[1])
    if not vals:
        return False
    v = np.concatenate(vals)
    return bool(v.size and np.nanmin(v) > 0)


def draw(ax, data, scheme, col, logy):
    drawn = 0
    for m in METHODS:
        b = band(data.get((scheme, m), {}), col)
        if b is None:
            continue
        x, mu, lo, hi = b
        ax.fill_between(x, lo, hi, color=COLOR[m], alpha=.13, lw=0)
        ax.plot(x, mu, color=COLOR[m], lw=1.8, label=m)
        drawn += 1
    if col == "test_acc":
        ax.axhline(0.5, color="crimson", ls="--", lw=.9, alpha=.55)
    elif logy:
        ax.set_yscale("log")
    ax.grid(True, alpha=.25)
    return drawn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="history CSV or directory of them")
    ap.add_argument("--out", default="results/plots")
    args = ap.parse_args()

    files = ([args.path] if os.path.isfile(args.path)
             else sorted(glob.glob(os.path.join(args.path, "*.csv"))))
    if not files:
        raise SystemExit(f"no history CSVs under {args.path}")
    data = load(files)
    schemes = [s for s in SCHEME_ORDER if any(k[0] == s for k in data)]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    made = []

    for col, ylabel, suffix in SERIES:
        # one figure per scheme
        for s in schemes:
            fig, ax = plt.subplots(figsize=(7.6, 4.6))
            logy = suffix == "loss" and all_positive(data, [s], col)
            if not draw(ax, data, s, col, logy):
                plt.close(fig); continue
            ax.set_title(f"{ylabel} vs iteration — {s}", fontsize=12)
            ax.set_xlabel("Iteration (cycle)"); ax.set_ylabel(ylabel)
            ax.legend(fontsize=8, framealpha=.9)
            fig.tight_layout()
            d = out / f"compare_{s}_{suffix}.png"
            fig.savefig(d, dpi=150); plt.close(fig); made.append(d)

        # one panel with every scheme side by side
        fig, axes = plt.subplots(1, len(schemes), figsize=(4.4*len(schemes), 4.3),
                                 sharey=True, squeeze=False)
        logy = suffix == "loss" and all_positive(data, schemes, col)
        for ax, s in zip(axes[0], schemes):
            draw(ax, data, s, col, logy)
            ax.set_title(s, fontsize=11); ax.set_xlabel("Iteration")
        axes[0][0].set_ylabel(ylabel)
        h, l = axes[0][0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=len(l), fontsize=9,
                   frameon=False, bbox_to_anchor=(.5, -.02))
        fig.suptitle(f"{ylabel} vs iteration — all methods", fontsize=13)
        fig.tight_layout(rect=[0, .05, 1, 1])
        d = out / f"compare_grid_{suffix}.png"
        fig.savefig(d, dpi=150, bbox_inches="tight"); plt.close(fig); made.append(d)

    for d in made:
        print(f"  {d}")
    print(f"\n{len(made)} figure(s) in {out}")


if __name__ == "__main__":
    main()
