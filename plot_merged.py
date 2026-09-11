"""The combined P2P-SDCA vs P2P-BDSVM figure, from one comparison run.

Reads the two `*_metrics.json` files `run_compare.py` writes and draws each
metric as one panel: per method, the mean over the 10 workers with a +/- 1 std
ribbon, so the two are directly comparable and the ribbon still says whether
the network agreed.

Both methods early-stop per node, and a stopped node stops appending metrics —
so the curves have different lengths, and the shorter one ending is the result,
not missing data. Each method's last measured cycle is marked, and the caption
line under each panel reports where every node had stopped.

    python plot_merged.py results/compare_09-10-2026
    python plot_merged.py results/compare_09-10-2026 --out results/plots
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METHODS = [("sdca", "P2P-SDCA", "#B4762A"),
           ("bdsvm", "P2P-BDSVM", "#17595E")]

# key, axis label, log scale, normalise-by-first-value
#
# Consensus error is normalised because the two methods measure it in different
# parameter spaces: SDCA's is ||w_i - mean_w|| over a 54-dim weight vector,
# BDSVM's is ||beta_i - mean_beta|| over 101 kernel coefficients. The absolute
# values are not on a common scale and plotting them together would invite a
# comparison that means nothing; each curve's progress *relative to its own
# starting spread* is the comparable quantity.
PANELS = [
    ("accuracy",        "Test accuracy",                        False, False),
    ("hinge_loss",      "Hinge loss (global)",                  True,  False),
    ("consensus_error", "Consensus error (relative to first)",  True,  True),
    ("comm_bytes",      "Cumulative bytes sent",                True,  False),
]


def series(per_node, key):
    """Mean and std over workers, on the cycles every worker reported.

    Workers stop at different cycles, so the common prefix is intersected
    rather than padded: averaging a stopped node's absent value against a
    running node's present one would invent a trajectory neither node had.
    """
    rounds, vals = [], []
    for rows in per_node:
        r = np.array([m["round"] for m in rows], dtype=float)
        v = np.array([m.get(key, np.nan) for m in rows], dtype=float)
        ok = ~np.isnan(v)
        if ok.any():
            rounds.append(r[ok])
            vals.append(v[ok])
    if not rounds:
        return None
    common = sorted(set.intersection(*(set(r.tolist()) for r in rounds)))
    if not common:
        return None
    x = np.array(common)
    M = np.stack([np.interp(x, r, v) for r, v in zip(rounds, vals)])
    return x, M.mean(axis=0), M.std(axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="directory holding <method>_metrics.json")
    ap.add_argument("--out", default=None, help="output directory")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out) if args.out else run_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    data = {}
    for key, _label, _c in METHODS:
        p = run_dir / f"{key}_metrics.json"
        if p.exists():
            data[key] = json.loads(p.read_text())
    if not data:
        raise SystemExit(f"No *_metrics.json found in {run_dir}")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (key, ylabel, log_y, norm) in zip(axes.ravel(), PANELS):
        for m, label, colour in METHODS:
            if m not in data:
                continue
            s = series(data[m]["per_node_metrics"], key)
            if s is None:
                continue
            x, mean, std = s
            if norm and mean[0] != 0:
                mean, std = mean / mean[0], std / mean[0]
            ax.plot(x, mean, color=colour, lw=1.8, label=label)
            ax.fill_between(x, mean - std, mean + std, color=colour, alpha=0.18,
                            linewidth=0)
            ax.plot(x[-1], mean[-1], "o", color=colour, ms=5)
        ax.set_xlabel("Gossip cycle")
        ax.set_ylabel(ylabel)
        if log_y:
            ax.set_yscale("log")
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.legend(frameon=False)

    bits = []
    for m, label, _c in METHODS:
        if m not in data:
            continue
        d = data[m]
        stops = [c for c in d["per_node_stop_cycle"] if c is not None]
        where = (f"all 10 nodes stopped by cycle {max(stops)}"
                 if len(stops) == len(d["per_node_stop_cycle"])
                 else f"{len(stops)}/10 nodes stopped")
        bits.append(f"{label}: {where}, "
                    f"acc {d['average_accuracy']:.4f}, "
                    f"{d['total_comm_bytes'] / 1e6:.2f} MB")
    fig.suptitle("P2P-SDCA vs P2P-BDSVM — covtype, 10 workers, random k-out (k=3), "
                 "per-node early stopping", fontsize=12)
    fig.text(0.5, 0.005, "   |   ".join(bits), ha="center", fontsize=9,
             color="#444444")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))

    path = out_dir / "merged_sdca_vs_bdsvm.png"
    fig.savefig(path, dpi=160)
    print(f"wrote {path}")

    # Each panel also on its own, for slides.
    for key, ylabel, log_y, norm in PANELS:
        f1, a1 = plt.subplots(figsize=(7, 4.5))
        drew = False
        for m, label, colour in METHODS:
            if m not in data:
                continue
            s = series(data[m]["per_node_metrics"], key)
            if s is None:
                continue
            x, mean, std = s
            if norm and mean[0] != 0:
                mean, std = mean / mean[0], std / mean[0]
            a1.plot(x, mean, color=colour, lw=1.8, label=label)
            a1.fill_between(x, mean - std, mean + std, color=colour, alpha=0.18,
                            linewidth=0)
            a1.plot(x[-1], mean[-1], "o", color=colour, ms=5)
            drew = True
        if not drew:
            plt.close(f1)
            continue
        a1.set_xlabel("Gossip cycle")
        a1.set_ylabel(ylabel)
        if log_y:
            a1.set_yscale("log")
        a1.grid(alpha=0.25, linewidth=0.6)
        a1.legend(frameon=False)
        a1.set_title(f"{ylabel} — mean ±1σ across workers", fontsize=11)
        f1.tight_layout()
        p1 = out_dir / f"merged_{key}.png"
        f1.savefig(p1, dpi=160)
        plt.close(f1)
        print(f"wrote {p1}")


if __name__ == "__main__":
    main()
