import os
import time
from datetime import datetime

from p2pfl.settings import Settings
Settings.general.DISABLE_RAY = True

from p2pfl.management.logger import logger

from config import CONFIG, CODE_DIR, DATA_DIR
from evaluation.metrics import print_summary, compute_node_accuracy
from evaluation.visualizer import plot_loss_vs_time, plot_gap_vs_time, plot_comm_cost_vs_time
from data.data_loader import load_rcv1_partitions, load_covtype_partitions
from src.network_layer.own_network.network_topology import connect_topology, setup_nodes


def run() -> None:
    os.chdir(CODE_DIR)

    results_dir = _next_run_dir()
    (results_dir / "logs").mkdir(parents=True, exist_ok=True)
    (results_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (results_dir / "plots").mkdir(parents=True, exist_ok=True)
    logger.info("main", f"Run started — results → {results_dir}")

    # Data — partition train across workers, split test equally (dataset via CONFIG["DATASET"])
    if CONFIG["DATASET"] == "covtype":
        worker_data = load_covtype_partitions(
            CONFIG["COVTYPE_PATH"], CONFIG["NUM_WORKERS"], CONFIG["SEED"], CONFIG["TEST_FRACTION"]
        )
    else:
        worker_data = load_rcv1_partitions(
            CONFIG["TRAIN_PATH"], CONFIG["TEST_PATH"], CONFIG["NUM_WORKERS"], CONFIG["SEED"]
        )

    # Nodes
    lightning_modules, nodes = setup_nodes(worker_data)

    # Topology + training
    connect_topology(nodes, CONFIG["TOPOLOGY"], CONFIG["BASE_PORT"])
    time.sleep(4)

    nodes[0].set_start_learning(rounds=CONFIG["ROUNDS"], epochs=CONFIG["EPOCHS"])
    logger.info(
        "main",
        f"Training started — {CONFIG['ROUNDS']} rounds, topology={CONFIG['TOPOLOGY']}, k={CONFIG['GOSSIP_K']}"
    )

    while True:
        time.sleep(1)
        if nodes[0].state.round is None:
            break

    logger.info("main", "Training complete")

    # Teardown
    for node in nodes:
        node.stop()
    logger.info("main", "All nodes stopped")

    # Test accuracy — W_tilde = averaged SDCA weight on each node's held-out shard
    accuracies = []
    for i, (lm, wd) in enumerate(zip(lightning_modules, worker_data)):
        s = lm._s
        if s.avg_cnt > 0:
            w_np = (s.w_avg / s.avg_cnt).cpu().numpy()
        else:
            w_np = lm.model.weight.data.cpu().numpy()
        acc = compute_node_accuracy(w_np, wd["X_test"], wd["y_test"])
        accuracies.append(acc)
        logger.info("main", f"Node {i} test accuracy = {acc:.4f}")
    avg_acc = sum(accuracies) / len(accuracies)
    logger.info("main", f"Average test accuracy across nodes = {avg_acc:.4f}")

    # Plots — one graph each, all workers overlaid
    all_metrics = [lm._metrics for lm in lightning_modules]
    plot_loss_vs_time(all_metrics,      results_dir / "plots" / "loss_vs_time.png")
    plot_gap_vs_time(all_metrics,       results_dir / "plots" / "duality_gap_vs_time.png")
    plot_comm_cost_vs_time(all_metrics, results_dir / "plots" / "comm_cost_vs_time.png")
    logger.info("main", f"Plots saved → {results_dir / 'plots'}")

    print_summary(lightning_modules, logger=logger)
    logger.info("main", f"Average accuracy: {avg_acc:.4f}")
    logger.info("main", "Done.")


if __name__ == "__main__":
    run()

def _next_run_dir() -> "Path":
    """Return results/run<N>_<mm-dd-yyyy>, with N auto-incremented per run."""
    from pathlib import Path
    import re
    results_root = CODE_DIR / "results"
    results_root.mkdir(parents=True, exist_ok=True)
    existing = [p.name for p in results_root.iterdir() if p.is_dir()]
    nums = [int(m.group(1)) for name in existing
            if (m := re.match(r"run(\d+)_", name))]
    next_n = max(nums, default=0) + 1
    date_str = datetime.now().strftime("%m-%d-%Y")
    return results_root / f"run{next_n}_{date_str}"
