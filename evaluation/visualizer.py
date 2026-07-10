"""Plotting utilities for convergence and comparison charts."""

import matplotlib.pyplot as plt
from pathlib import Path


def plot_loss_vs_time(metrics_per_worker: list, out_path: Path) -> None:
    """Hinge loss vs wall time for all workers on one graph."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, m in enumerate(metrics_per_worker):
        if not m:
            continue
        ax.plot([r["wall_time"] for r in m], [r["hinge_loss"] for r in m],
                marker="o", label=f"Worker {i}")
    ax.set(title="Hinge Loss vs Time", xlabel="Wall time (s)", ylabel="Hinge loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

def plot_gap_vs_time(metrics_per_worker: list, out_path: Path) -> None:
    """Duality gap vs wall time for all workers on one graph."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, m in enumerate(metrics_per_worker):
        if not m:
            continue
        ax.plot([r["wall_time"] for r in m], [r["duality_gap"] for r in m],
                marker="s", label=f"Worker {i}")
    ax.set(title="Duality Gap P(w)−D(α) vs Time", xlabel="Wall time (s)", ylabel="Duality gap")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_comm_cost_vs_time(metrics_per_worker: list, out_path: Path) -> None:
    """Cumulative communication bytes vs wall time for all workers on one graph."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, m in enumerate(metrics_per_worker):
        if not m:
            continue
        ax.plot([r["wall_time"] for r in m], [r["comm_bytes"] for r in m],
                marker="^", label=f"Worker {i}")
    ax.set(title="Communication Cost vs Time", xlabel="Wall time (s)", ylabel="Cumulative bytes sent")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
