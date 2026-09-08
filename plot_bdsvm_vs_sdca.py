"""BDSVM against SDCA on one pair of axes: mean over workers, with a std band.

plot_history.py draws every worker as its own line, which shows the spread but
gets unreadable once two methods share an axis. Here each method is one mean
line with a +/- 1 std ribbon over the 10 workers, so the two are directly
comparable and the ribbon still says whether the network agreed.

    python plot_bdsvm_vs_sdca.py results/history --out results/plots
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

PAIR = [("bdsvm", "P2P-BDSVM", "#17595E"),
        ("sdca",  "P2P-SDCA",  "#B4762A")]
SCHEME_ORDER = ["iid", "dirichlet_1.0", "dirichlet_0.3", "dirichlet_0.1", "label_skew"]
SERIES = [("test_acc", "Test accuracy", "accuracy", False),
          ("hinge_loss", "Hinge loss", "loss", True)]


def fnum(row, col):
    try:
        return float(row[col])
    except (KeyError, TypeError, ValueError):
        return math.nan


def load(paths):
    out = defaultdict(lambda: defaultdict(list))
    for p in paths:
        with open(p) as fh:
            for r in csv.DictReader(fh):
                out[(r["scheme"], r["method"])][int(r["node"])].append(r)
    return out


def stats(nodes, col):
    """Rounds, mean over workers, std over workers."""
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
    return cx, ys.mean(axis=0), ys.std(axis=0), ys.shape[0]


def draw(ax, data, scheme, col, logy):
    n_workers = 0
    for key, label, colour in PAIR:
        st = stats(data.get((scheme, key), {}), col)
        if st is None:
            continue
        x, mu, sd, n_workers = st
        lo = mu - sd
        if logy:
            # On a log axis mu-sd goes non-positive whenever the spread exceeds
            # the mean, and the band then fills the whole panel. Clip it to a
            # decade below the mean so the ribbon still reads as "wide" without
            # swallowing the figure.
            lo = np.maximum(lo, mu * 0.1)
        ax.fill_between(x, lo, mu + sd, color=colour, alpha=.18, lw=0)
        ax.plot(x, mu, color=colour, lw=2.0, label=label)
    if col == "test_acc":
        ax.axhline(0.5, color="crimson", ls="--", lw=.9, alpha=.55,
                   label="chance")
    elif logy:
        ax.set_yscale("log")
    ax.grid(True, alpha=.25)
    return n_workers


def all_positive(data, schemes, col):
    """A log axis silently drops non-positive points; label_skew sends hinge
    loss to exactly 0, so only take log when every value is strictly positive."""
    vals = []
    for s in schemes:
        for key, _, _ in PAIR:
            st = stats(data.get((s, key), {}), col)
            if st is not None:
                vals.append(st[1])
    if not vals:
        return False
    v = np.concatenate(vals)
    return bool(v.size and np.nanmin(v) > 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--out", default="results/plots")
    args = ap.parse_args()

    files = ([args.path] if os.path.isfile(args.path)
             else sorted(glob.glob(os.path.join(args.path, "*.csv"))))
    data = load(files)
    schemes = [s for s in SCHEME_ORDER if any(k[0] == s for k in data)]
    if not schemes:
        raise SystemExit("no BDSVM/SDCA history found")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    made = []

    for col, ylabel, suffix, wants_log in SERIES:
        logy = wants_log and all_positive(data, schemes, col)
        fig, axes = plt.subplots(1, len(schemes),
                                 figsize=(4.6 * len(schemes), 4.4),
                                 sharey=True, squeeze=False)
        nw = 0
        for ax, s in zip(axes[0], schemes):
            nw = draw(ax, data, s, col, logy) or nw
            ax.set_title(s, fontsize=11)
            ax.set_xlabel("Iteration (cycle)")
        axes[0][0].set_ylabel(ylabel)
        h, l = axes[0][0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=len(l), fontsize=9.5,
                   frameon=False, bbox_to_anchor=(.5, -.02))
        fig.suptitle(f"{ylabel} vs iteration — mean over {nw} workers, "
                     f"$\\pm$1 std band", fontsize=13)
        fig.tight_layout(rect=[0, .06, 1, 1])
        d = out / f"bdsvm_vs_sdca_{suffix}.png"
        fig.savefig(d, dpi=150, bbox_inches="tight")
        plt.close(fig)
        made.append(d)

    for d in made:
        print(f"  {d}")


if __name__ == "__main__":
    main()
