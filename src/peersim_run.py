"""PeerSim-Python driver for the gossip-SDCA setup.

Thin driver: it loads config-driven data shards (`src/data_sharding.py`), hands
them to the `Simulation` orchestrator (`src/peersim_python/simulation.py`) which
assembles and runs the in-process PeerSim network, then collects per-node
metrics and renders the plots. All engine/assembly logic lives behind
`Simulation`; this file only owns data loading, the results directory, and
plotting.

    python src/peersim_run.py        # run the default dataset (CONFIG["DATASET"])
    python src/peersim_run.py rcv    # rcv1
    python src/peersim_run.py cov    # covtype
    python src/peersim_run.py gis    # gisette   (downloaded + preprocessed on first use)
    python src/peersim_run.py rsim   # real-sim  (downloaded + preprocessed on first use)

Results land in results/peersim_run<N>_<mm-dd-yyyy>/ (separate from main.py's
run<N>_ folders).
"""

import os
import re
import sys
from datetime import datetime
from pathlib import Path

# Ensure the repo root is importable so `src.*` and root-level `data.*` resolve
# whether this is run as `python src/peersim_run.py` or `python -m src.peersim_run`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CONFIG, CODE_DIR
from src.data_sharding import load_shards
from src.evaluation.metrics import print_summary
from src.evaluation.visualizer import (
    plot_loss_vs_time, plot_gap_vs_time, plot_comm_cost_vs_time, plot_accuracy_vs_time, plot_std_band,
)

from src.peersim_python.core import Network
from src.peersim_python.simulation import Simulation
from src.peersim_python.logger import logger
import json


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

    results_dir = _next_run_dir()
    (results_dir / "plots").mkdir(parents=True, exist_ok=True)
    logger.info("main", f"PeerSim-Python run — results → {results_dir}")

    # Data — one shard per node (dataset + shard count chosen by CONFIG)
    worker_data = load_shards(CONFIG)

    # Assemble + run the whole PeerSim experiment via the orchestrator
    Simulation(CONFIG, worker_data).run(cycles)

    # Collect per-node results
    protos = [Network.get(i).getProtocol(Simulation.SDCA_PID)
              for i in range(CONFIG["NUM_WORKERS"])]
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

    # Aggregated view: mean trajectory ± 1 std across workers, drawn against BOTH
    # iterations and wall-clock time (one std-band plot per metric per axis).
    std_band_specs = [
        ("duality_gap", "duality_gap_std_band", "Duality Gap",        "Duality gap",           True),
        ("hinge_loss",  "loss_std_band",        "Hinge Loss",         "Hinge loss",            True),
        ("accuracy",    "accuracy_std_band",    "Test Accuracy",      "Test accuracy",         False),
        ("comm_bytes",  "comm_cost_std_band",   "Communication Cost", "Cumulative bytes sent", False),
    ]
    for x_key, suffix in (("round", "vs_iterations"), ("wall_time", "vs_time")):
        for y_key, base, name, ylabel, log_y in std_band_specs:
            plot_std_band(all_metrics, plots / f"{base}_{suffix}.png",
                          y_key, f"{name} — Mean ±1σ Across Workers",
                          ylabel, log_y=log_y, x_key=x_key)
    logger.info("main", f"Plots saved → {plots}")

    # Communication cost — computed and saved once the run has stopped. Each
    # node's comm_bytes freezes when it converges and goes silent, so these are
    # the final per-node totals.
    final_comm = [p.comm_bytes for p in protos]
    stop_cycles = [getattr(p, "stop_cycle", None) for p in protos]
    total_comm = sum(final_comm)
    mean_comm = total_comm / len(final_comm)
    std_comm = (sum((c - mean_comm) ** 2 for c in final_comm) / len(final_comm)) ** 0.5
    logger.info("comm", f"Total communication cost = {total_comm:,} bytes "
                        f"({total_comm / 1e6:.2f} MB) over {len(final_comm)} nodes")
    logger.info("comm", f"Per-node comm cost: mean={mean_comm / 1e3:.1f} KB, "
                        f"std={std_comm / 1e3:.1f} KB")
    logger.info("comm", f"Node stop cycles: {stop_cycles}")
    comm_summary = {
        "total_comm_bytes": total_comm,
        "mean_comm_bytes_per_node": mean_comm,
        "std_comm_bytes_per_node": std_comm,
        "num_nodes": len(final_comm),
        "per_node_comm_bytes": final_comm,
        "per_node_stop_cycle": stop_cycles,
    }
    (results_dir / "comm_cost_summary.json").write_text(json.dumps(comm_summary, indent=2))
    logger.info("comm", f"Communication-cost summary → "
                        f"{results_dir / 'comm_cost_summary.json'}")

    print_summary([_MetricsView(m) for m in all_metrics], logger=logger)
    logger.info("main", f"Average accuracy: {avg_acc:.4f}")
    logger.info("main", "Done.")


DATASET_KEYWORDS = {
    "rcv":  "rcv1",
    "cov":  "covtype",
    "gis":  "gisette",
    "rsim": "real-sim",
}


def _select_dataset(keyword: str) -> None:
    """Map a CLI keyword to CONFIG['DATASET']; build the file on demand if missing."""
    name = DATASET_KEYWORDS.get(keyword.lower())
    if name is None:
        valid = " | ".join(DATASET_KEYWORDS)
        raise SystemExit(f"Unknown dataset '{keyword}'. Choose one of: {valid}")
    CONFIG["DATASET"] = name
    logger.info("main", f"Dataset '{keyword}' -> {name}")
    # covtype, gisette and real-sim are downloaded + preprocessed on first use
    # (idempotent — the raw download is deleted once the LIBSVM file is built).
    if name == "covtype" and not CONFIG["COVTYPE_PATH"].exists():
        from data.extract_data import preprocess_covertype_uci
        preprocess_covertype_uci()
    elif name == "gisette" and not CONFIG["GISETTE_PATH"].exists():
        from data.extract_data import preprocess_gisette
        preprocess_gisette()
    elif name == "real-sim" and not CONFIG["REALSIM_PATH"].exists():
        from data.extract_data import preprocess_real_sim
        preprocess_real_sim()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        _select_dataset(sys.argv[1])
    run()
