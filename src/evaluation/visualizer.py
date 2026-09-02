"""Plotting utilities for convergence and comparison charts.

Every plot can be drawn against either wall-clock time (`x_key="wall_time"`, the
default) or the training iteration/round (`x_key="round"`). The PeerSim driver
renders both so the two views can be compared side by side — the iteration view
is the fairer convergence comparison since wall time is a single-process
simulator artifact (nodes run sequentially, so their timestamps are staggered).
"""

import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


def _plot_series(metrics_per_worker, out_path, y_key, title, ylabel,
                 marker="o", log_y=False, x_key="wall_time",
                 x_label="Wall time (s)"):
    """Draw one metric for every worker on a single graph (shared helper)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, m in enumerate(metrics_per_worker):
        if not m:
            continue
        ax.plot([r[x_key] for r in m], [r[y_key] for r in m],
                marker=marker, markersize=3, label=f"Worker {i}")
    ax.set(title=title, xlabel=x_label, ylabel=ylabel)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    if log_y:
        ax.set_yscale("log")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def _axis(x_key):
    """(x_key, x_label, title_suffix) for the two supported x-axes."""
    if x_key == "round":
        return "round", "Iteration (cycle)", "Iterations"
    return "wall_time", "Wall time (s)", "Time"


def plot_loss_vs_time(metrics_per_worker: list, out_path: Path,
                      x_key: str = "wall_time") -> None:
    """Hinge loss for all workers on one graph (vs time or iterations)."""
    xk, xl, suf = _axis(x_key)
    _plot_series(metrics_per_worker, out_path, "hinge_loss",
                 f"Hinge Loss vs {suf}", "Hinge loss",
                 marker="o", log_y=True, x_key=xk, x_label=xl)


def plot_gap_vs_time(metrics_per_worker: list, out_path: Path,
                     x_key: str = "wall_time") -> None:
    """Duality gap P(w)-D(alpha) for all workers (vs time or iterations)."""
    xk, xl, suf = _axis(x_key)
    _plot_series(metrics_per_worker, out_path, "duality_gap",
                 f"Duality Gap P(w)-D(α) vs {suf}", "Duality gap",
                 marker="s", log_y=True, x_key=xk, x_label=xl)


def plot_accuracy_vs_time(metrics_per_worker: list, out_path: Path,
                          x_key: str = "wall_time") -> None:
    """Test accuracy for all workers on one graph (vs time or iterations)."""
    xk, xl, suf = _axis(x_key)
    _plot_series(metrics_per_worker, out_path, "accuracy",
                 f"Test Accuracy vs {suf}", "Test accuracy",
                 marker="o", log_y=False, x_key=xk, x_label=xl)


def plot_comm_cost_vs_time(metrics_per_worker: list, out_path: Path,
                           x_key: str = "wall_time") -> None:
    """Cumulative communication bytes for all workers (vs time or iterations)."""
    xk, xl, suf = _axis(x_key)
    _plot_series(metrics_per_worker, out_path, "comm_bytes",
                 f"Communication Cost vs {suf}", "Cumulative bytes sent",
                 marker="^", log_y=False, x_key=xk, x_label=xl)

def _aggregate_by_round(metrics_per_worker, y_key, x_key):
    """Align workers by cycle and return (x, mean, std) arrays across workers.

    Workers may stop at different cycles (convergence), so each round is
    averaged over only the workers that actually recorded it. `x` is the round
    number, or the mean wall-clock time at that round when `x_key="wall_time"`.
    """
    by_round = {}
    for m in metrics_per_worker:
        for r in m:
            slot = by_round.setdefault(r["round"], {"y": [], "x": []})
            slot["y"].append(r[y_key])
            slot["x"].append(r[x_key])
    rounds = sorted(by_round)
    x    = np.array([np.mean(by_round[r]["x"]) for r in rounds])
    mean = np.array([np.mean(by_round[r]["y"]) for r in rounds])
    std  = np.array([np.std(by_round[r]["y"]) for r in rounds])
    return x, mean, std


def plot_std_band(metrics_per_worker: list, out_path: Path, y_key: str,
                  title: str, ylabel: str, log_y: bool = False,
                  x_key: str = "round", show_workers: bool = True) -> None:
    """Mean convergence trajectory with a ±1 std band across all workers.

    Shows how tightly the individual workers track the swarm average: the solid
    line is the per-cycle mean, the shaded band is ±1 standard deviation, and
    (optionally) the faint lines behind it are the individual workers.
    """
    xk, xl, _ = _axis(x_key)
    x, mean, std = _aggregate_by_round(metrics_per_worker, y_key, xk)

    fig, ax = plt.subplots(figsize=(8, 5))
    if show_workers:
        for m in metrics_per_worker:
            if not m:
                continue
            ax.plot([r[xk] for r in m], [r[y_key] for r in m],
                    color="0.8", linewidth=0.8, zorder=1)
    ax.fill_between(x, mean - std, mean + std, alpha=0.25, color="C0",
                    label="±1 std (worker spread)", zorder=2)
    ax.plot(x, mean, color="C0", linewidth=2.0, marker="o", markersize=3,
            label="Mean across workers", zorder=3)
    ax.set(title=title, xlabel=xl, ylabel=ylabel)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    if log_y:
        ax.set_yscale("log")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
