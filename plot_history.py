"""Per-node convergence plots from the history CSVs.

Produces, for every (method, topology, scheme) in the input:

    <out>/<key>_loss.png       hinge loss vs iteration, one line per node
    <out>/<key>_accuracy.png   test accuracy vs iteration, one line per node

Both plot every node separately rather than a mean, because the spread between
nodes is the thing a P2P run has and a server run does not — a tight band means
the network agreed, a fan means it did not. The mean is drawn over the top in
black so the two readings are available at once.

    python -m p2p.run_peersim ... --history results/history/<name>.csv
    python plot_history.py results/history --out results/plots

    python plot_history.py results/history/covtype_bdsvm_ring_label_skew.csv
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

# One metric per figure; (csv column, axis label, log scale, filename suffix)
SERIES = [
    ("hinge_loss", "Hinge loss", True, "loss"),
    ("test_acc", "Test accuracy", False, "accuracy"),
]


def read(path):
    """Group rows into {(method, topology, scheme, k): {node: [rows...]}}."""
    runs = defaultdict(lambda: defaultdict(list))
    with open(path) as fh:
        for r in csv.DictReader(fh):
            key = (r.get("tag") or "", r["method"], r["topology"],
                   r["scheme"], r["gossip_k"], r.get("components", "?"))
            runs[key][int(r["node"])].append(r)
    return runs


def fnum(row, col):
    try:
        v = float(row[col])
    except (KeyError, TypeError, ValueError):
        return math.nan
    return v


def plot_run(key, nodes, out_dir):
    tag, method, topo, scheme, k, comps = key
    stem = "_".join(x for x in (tag, method, topo, scheme, f"k{k}") if x)
    made = []

    for col, ylabel, logy, suffix in SERIES:
        # Skip a metric this protocol never reports (e.g. accuracy while
        # FedSSL-AMC is still in its encoder phase, or a gap on a primal method).
        if all(math.isnan(fnum(r, col)) for rows in nodes.values() for r in rows):
            continue

        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        curves = []
        for node in sorted(nodes):
            rows = sorted(nodes[node], key=lambda r: int(r["round"]))
            x = np.array([int(r["round"]) for r in rows])
            y = np.array([fnum(r, col) for r in rows])
            m = ~np.isnan(y)
            if not m.any():
                continue
            ax.plot(x[m], y[m], lw=1.1, alpha=.75, label=f"node {node}")
            curves.append((x[m], y[m]))

        if not curves:
            plt.close(fig)
            continue

        # Mean over nodes, on the rounds every node reported.
        common = sorted(set.intersection(*(set(x.tolist()) for x, _ in curves)))
        if common:
            cx = np.array(common)
            cy = np.mean([np.interp(cx, x, y) for x, y in curves], axis=0)
            ax.plot(cx, cy, color="black", lw=2.0, zorder=5, label="mean")

        if logy:
            # A log axis silently drops non-positive points, which turns a
            # curve that legitimately reaches zero into disconnected spikes.
            # Under label_skew a single-class node fits its own shard exactly,
            # so hinge loss really does hit 0 — that is signal, not noise, and
            # it is why the node learns nothing global. Fall back to a linear
            # axis whenever any value is non-positive.
            finite = np.concatenate([y for _, y in curves])
            finite = finite[np.isfinite(finite)]
            if finite.size and finite.min() > 0:
                ax.set_yscale("log")
            else:
                ax.set_ylim(bottom=0)
        if col == "test_acc":
            ax.axhline(0.5, color="crimson", ls="--", lw=.9, alpha=.6)
            ax.text(ax.get_xlim()[1], 0.5, " chance", color="crimson",
                    fontsize=8, va="center")

        disc = "" if comps in ("1", "?") else f"  —  DISCONNECTED ({comps} components)"
        ax.set_title(f"{method} · {topo} · {scheme} · k={k}{disc}", fontsize=11)
        ax.set_xlabel("Iteration (cycle)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=.25)
        ncol = 2 if len(curves) > 8 else 1
        ax.legend(fontsize=7, ncol=ncol, framealpha=.9)
        fig.tight_layout()

        dest = out_dir / f"{stem}_{suffix}.png"
        fig.savefig(dest, dpi=150)
        plt.close(fig)
        made.append(dest)

    return made


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="history CSV, or a directory of them")
    ap.add_argument("--out", default="results/plots")
    args = ap.parse_args()

    files = ([args.path] if os.path.isfile(args.path)
             else sorted(glob.glob(os.path.join(args.path, "*.csv"))))
    if not files:
        raise SystemExit(f"no history CSVs under {args.path}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for f in files:
        for key, nodes in read(f).items():
            for dest in plot_run(key, nodes, out_dir):
                print(f"  {dest}")
                total += 1
    print(f"\n{total} plot(s) in {out_dir}")


if __name__ == "__main__":
    main()
