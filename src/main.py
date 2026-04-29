import os
import time
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

from p2pfl.settings import Settings
Settings.general.DISABLE_RAY = True

from p2pfl.management.logger import logger

from config import CONFIG, CODE_DIR, DATA_DIR
from data_loader import load_data, split_workers
from network_topology import connect_topology, setup_nodes


def run_training(nodes: list) -> None:
    """Wire topology, trigger learning from node 0, and wait for completion."""
    connect_topology(nodes, CONFIG["TOPOLOGY"], CONFIG["BASE_PORT"])
    time.sleep(4)

    nodes[0].set_start_learning(
        rounds = CONFIG["ROUNDS"],
        epochs = CONFIG["EPOCHS"],
    )
    logger.info("main", f"Training started — {CONFIG['ROUNDS']} rounds, topology={CONFIG['TOPOLOGY']}")

    while True:
        time.sleep(1)
        if nodes[0].state.round is None:
            break

    logger.info("main", "Training complete")


def save_plots(lightning_modules: list, results_dir: Path) -> None:
    """Plot per-worker convergence metrics and save to results_dir/plots/."""
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle(
        f"P2P-SDCA  |  topology={CONFIG['TOPOLOGY']}  "
        f"workers={CONFIG['NUM_WORKERS']}  λ={CONFIG['LAMBDA']}  "
        f"rounds={CONFIG['ROUNDS']}"
    )

    for i, lm in enumerate(lightning_modules):
        m = lm._metrics
        if not m:
            continue
        rounds = [r["round"]       for r in m]
        gaps   = [r["duality_gap"] for r in m]
        hinges = [r["hinge_loss"]  for r in m]
        walls  = [r["wall_time"]   for r in m]

        axes[0].plot(rounds, gaps,   marker="o", label=f"Worker {i}")
        axes[1].plot(rounds, hinges, marker="s", label=f"Worker {i}")
        axes[2].plot(walls,  gaps,   marker="^", label=f"Worker {i}")

    axes[0].set(title="Duality Gap  P(w) − D(α)", xlabel="Gossip round", ylabel="Gap")
    axes[1].set(title="Hinge Loss",               xlabel="Gossip round", ylabel="Loss")
    axes[2].set(title="Duality Gap vs Wall Time",  xlabel="Wall time (s)", ylabel="Gap")

    for ax in axes:
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")

    plt.tight_layout()
    plot_path = plots_dir / "convergence.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("main", f"Convergence plot saved → {plot_path}")


def print_summary(lightning_modules: list) -> None:
    """Print final per-worker convergence table."""
    print(f"\n{'Worker':>6}  {'Final Gap':>12}  {'Final Hinge':>12}  "
          f"{'Primal':>10}  {'Dual':>10}  {'Time (s)':>10}")
    print("─" * 66)
    for i, lm in enumerate(lightning_modules):
        if lm._metrics:
            r = lm._metrics[-1]
            logger.info(
                "main",
                f"Worker {i}  gap={r['duality_gap']:.6f}  hinge={r['hinge_loss']:.6f}  "
                f"primal={r['primal']:.6f}  dual={r['dual']:.6f}  t={r['wall_time']:.1f}s"
            )
            print(f"{i:>6}  {r['duality_gap']:>12.6f}  {r['hinge_loss']:>12.6f}  "
                  f"{r['primal']:>10.6f}  {r['dual']:>10.6f}  {r['wall_time']:>10.1f}")


def main():
    os.chdir(CODE_DIR)

    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = CODE_DIR / "results" / f"run_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    logger.info("main", f"Run started — results → {results_dir}")

    # Data
    _data_train = DATA_DIR / "train" / "rcv1_train.binary"
    X_csr, y    = load_data(_data_train, CONFIG["SEED"])
    worker_data = split_workers(X_csr, y, CONFIG["NUM_WORKERS"])

    # Nodes
    lightning_modules, nodes = setup_nodes(worker_data)

    # Training
    run_training(nodes)

    # Teardown
    for node in nodes:
        node.stop()
    logger.info("main", "All nodes stopped")

    # Results
    save_plots(lightning_modules, results_dir)
    print_summary(lightning_modules)
    logger.info("main", "Done.")


if __name__ == "__main__":
    main()
