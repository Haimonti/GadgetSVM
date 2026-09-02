"""PeerSim-Python driver for the gossip-SDCA setup.

The PeerSim counterpart of `main.py`: it runs the *same* decentralised SDCA-SVM
experiment (partition the dataset, gossip weights between nodes, aggregate per node,
track convergence) but on the pure-Python PeerSim engine in
`src/peersim_python/` instead of the p2pfl runtime — so it needs
no p2pfl install and runs entirely as an in-process P2P network simulation.

    python peersim_run.py            # run for CONFIG["ROUNDS"] cycles (or until
                                     # the duality-gap threshold is met)
    python peersim_run.py 5          # override: run at most 5 cycles (quick test)

Results land in results/peersim_run<N>_<mm-dd-yyyy>/ (separate from main.py's
run<N>_ folders).
"""

import os
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.datasets import load_svmlight_file

# Ensure the repo root is importable so `src.*` and root-level `data.*` resolve
# whether this is run as `python src/peersim_run.py` or `python -m src.peersim_run`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CONFIG, CODE_DIR
from src.evaluation.metrics import print_summary
from src.evaluation.visualizer import (
    plot_loss_vs_time, plot_gap_vs_time, plot_comm_cost_vs_time, plot_accuracy_vs_time, plot_std_band,
)

from src.peersim_python.core import (
    Network, GeneralNode, CommonState,
)
from src.peersim_python.idle_protocol import IdleProtocol
from src.peersim_python.sdca_protocol import SDCAProtocol
from src.peersim_python.dynamics import (
    WireKOut, WireRing, WireFull, WireStar, WireMesh,
)
from src.peersim_python.cdsim import CDSimulator
from src.peersim_python.observers import (
    DataInitializer, ConvergenceObserver,
)
from src.peersim_python.logger import logger

LINKABLE_PID = 0
SDCA_PID = 1


def _topology(name: str):
    """Map a CONFIG topology name to the matching Wire* control (undirected)."""
    if name == "random_kout":
        return WireKOut(LINKABLE_PID, CONFIG["GOSSIP_K"], undir=True)
    if name == "ring":
        return WireRing(LINKABLE_PID, undir=True)
    if name == "full":
        return WireFull(LINKABLE_PID, undir=True)
    if name == "star":
        return WireStar(LINKABLE_PID, undir=True)
    if name == "mesh":
        return WireMesh(LINKABLE_PID, undir=True)
    raise ValueError(
        f"Unknown topology '{name}'. Choose: random_kout | ring | full | star | mesh"
    )




def _next_run_dir() -> Path:
    """results/peersim_run<N>_<mm-dd-yyyy> with N auto-incremented per run."""
    root = CODE_DIR / "results"
    root.mkdir(parents=True, exist_ok=True)
    nums = [int(m.group(1)) for p in root.iterdir() if p.is_dir()
            if (m := re.match(r"peersim_run(\d+)_", p.name))]
    n = max(nums, default=0) + 1
    return root / f"peersim_run{n}_{datetime.now().strftime('%m-%d-%Y')}"


class _MetricsView:
    """Adapter so print_summary (expects `._metrics`) works with SDCAProtocol."""
    def __init__(self, metrics):
        self._metrics = metrics


def run(cycles: int = None) -> None:
    os.chdir(CODE_DIR)
    cycles = cycles if cycles is not None else CONFIG["ROUNDS"]

    results_dir = _next_run_dir()
    (results_dir / "plots").mkdir(parents=True, exist_ok=True)
    logger.info("main", f"PeerSim-Python run — results → {results_dir}")

    # Shared RNG (drives wiring, node visiting order, gossip peer choice)
    CommonState.initializeRandom(CONFIG["SEED"])

    # Data — one shard per node (dataset chosen by CONFIG["DATASET"])
    worker_data = load_dataset()
    n_nodes = CONFIG["NUM_WORKERS"]

    # Build the network: each node = [IdleProtocol(links), SDCAProtocol(learner)]
    prototype = GeneralNode([IdleProtocol(), SDCAProtocol()])
    Network.reset(n_nodes, prototype)
    logger.info("network", f"{n_nodes} nodes built (protocol 0=Linkable, 1=SDCA)")

    # init.* — wire topology, then assign data shards
    initializers = [
        _topology(CONFIG["TOPOLOGY"]),
        DataInitializer(
            SDCA_PID, worker_data, CONFIG["LAMBDA"], CONFIG["T0_FRACTION"],
            CONFIG["GOSSIP_K"], sgd_init=CONFIG.get("SGD_INIT", False),
        ),
    ]
    # control.* — convergence-threshold stop
    controls = [ConvergenceObserver(SDCA_PID, CONFIG["GAP_THRESHOLD"])]

    sim = CDSimulator(
        cycles=cycles,
        initializers=initializers,
        controls=controls,
        activation=CONFIG.get("ACTIVATION", "shuffle"),
    )
    logger.info(
        "main",
        f"Training — topology={CONFIG['TOPOLOGY']}, k={CONFIG['GOSSIP_K']}, "
        f"max_cycles={cycles}, gap_threshold={CONFIG['GAP_THRESHOLD']}",
    )
    sim.nextExperiment()
    logger.info("main", "Training complete")

    # Collect per-node results
    protos = [Network.get(i).getProtocol(SDCA_PID) for i in range(n_nodes)]
    all_metrics = [p.metrics for p in protos]

    accuracies = []
    for i, p in enumerate(protos):
        acc = p.accuracy()
        accuracies.append(acc)
        logger.info("main", f"Node {i} test accuracy = {acc:.4f}")
    avg_acc = sum(accuracies) / len(accuracies)
    logger.info("main", f"Average test accuracy across nodes = {avg_acc:.4f}")

    # Plots — one graph each, all workers overlaid. Every metric is drawn twice:
    # against wall-clock time and against iteration (cycle) for comparison. The
    # iteration view is the fairer convergence comparison (wall time is a
    # single-process simulator artifact).
    plots = results_dir / "plots"
    for x_key, suffix in (("wall_time", "vs_time"), ("round", "vs_iterations")):
        plot_loss_vs_time(all_metrics,     plots / f"loss_{suffix}.png",         x_key=x_key)
        plot_gap_vs_time(all_metrics,      plots / f"duality_gap_{suffix}.png",  x_key=x_key)
        plot_accuracy_vs_time(all_metrics, plots / f"accuracy_{suffix}.png",     x_key=x_key)
        plot_comm_cost_vs_time(all_metrics, plots / f"comm_cost_{suffix}.png",   x_key=x_key)

    # Aggregated view: mean trajectory ± 1 std across workers (per cycle), so the
    # swarm's average convergence is shown against how far individual workers
    # spread around it.
    plot_std_band(all_metrics, plots / "duality_gap_std_band.png",
                  "duality_gap", "Duality Gap — Mean ±1σ Across Workers",
                  "Duality gap", log_y=True)
    plot_std_band(all_metrics, plots / "loss_std_band.png",
                  "hinge_loss", "Hinge Loss — Mean ±1σ Across Workers",
                  "Hinge loss", log_y=True)
    plot_std_band(all_metrics, plots / "accuracy_std_band.png",
                  "accuracy", "Test Accuracy — Mean ±1σ Across Workers",
                  "Test accuracy", log_y=False)
    logger.info("main", f"Plots saved → {plots}")

    print_summary([_MetricsView(m) for m in all_metrics], logger=logger)
    logger.info("main", f"Average accuracy: {avg_acc:.4f}")
    logger.info("main", "Done.")


def _to_pm1(y):
    """Map a two-class label vector to {-1, +1} (larger class value -> +1).

    covtype uses {1, 2}; rcv1 uses {-1, +1}. Kept local so this driver stays
    p2pfl-free (data/data_loader.py imports p2pfl's logger at module load).
    """
    y = np.asarray(y, dtype=np.float32)
    vals = np.unique(y)
    if len(vals) == 2:
        return np.where(y == vals.max(), 1.0, -1.0).astype(np.float32)
    return np.sign(y).astype(np.float32)


def _partition(X_train, y_train, X_test, y_test, n_workers):
    """Partition train across workers and split test equally across workers."""
    tr = np.array_split(np.arange(X_train.shape[0]), n_workers)
    te = np.array_split(np.arange(X_test.shape[0]), n_workers)
    data = []
    for i in range(n_workers):
        data.append({
            "X_csr":   X_train[tr[i]].tocsr(),
            "y":       y_train[tr[i]],
            "X_test":  X_test[te[i]].tocsr(),
            "y_test":  y_test[te[i]],
            "n_local": len(tr[i]),
        })
        logger.info("data", f"Worker {i}: {len(tr[i])} train, {len(te[i])} test")
    return data


def load_rcv1(train_path, test_path, n_workers, seed):
    """rcv1: two LIBSVM files (separate train/test)."""
    rng = np.random.RandomState(seed)
    X_train, y_train = load_svmlight_file(str(train_path))
    y_train = _to_pm1(y_train)
    perm = rng.permutation(X_train.shape[0])
    X_train = X_train[perm].tocsr()
    y_train = y_train[perm]
    X_test, y_test = load_svmlight_file(str(test_path), n_features=X_train.shape[1])
    y_test = _to_pm1(y_test)
    logger.info("data", f"rcv1: {X_train.shape[0]} train, {X_test.shape[0]} test, "
                        f"{X_train.shape[1]} features")
    return _partition(X_train, y_train, X_test, y_test, n_workers)


def load_covtype(path, n_workers, seed, test_fraction):
    """covtype: one LIBSVM file — hold out `test_fraction` as test, then partition."""
    rng = np.random.RandomState(seed)
    X, y = load_svmlight_file(str(path))
    y = _to_pm1(y)
    n = X.shape[0]
    perm = rng.permutation(n)
    X = X[perm].tocsr()
    y = y[perm]
    n_test = int(test_fraction * n)
    logger.info("data", f"covtype: {n} samples, {X.shape[1]} features -> "
                        f"{n - n_test} train / {n_test} test")
    return _partition(
        X[n_test:].tocsr(), y[n_test:], X[:n_test].tocsr(), y[:n_test], n_workers
    )


def load_dataset():
    """Dispatch on CONFIG['DATASET'] and return per-node data shards."""
    ds = CONFIG.get("DATASET", "rcv1")
    if ds == "covtype":
        return load_covtype(
            CONFIG["COVTYPE_PATH"], CONFIG["NUM_WORKERS"],
            CONFIG["SEED"], CONFIG["TEST_FRACTION"],
        )
    if ds == "rcv1":
        return load_rcv1(
            CONFIG["TRAIN_PATH"], CONFIG["TEST_PATH"],
            CONFIG["NUM_WORKERS"], CONFIG["SEED"],
        )
    raise ValueError(f"Unknown DATASET '{ds}'. Choose: covtype | rcv1")

if __name__ == "__main__":
    cli_cycles = int(sys.argv[1]) if len(sys.argv) > 1 else None
    run(cli_cycles)
