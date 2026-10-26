"""Pick a model and run it — the training worker behind `scripts/main.py`.

This module knows *which* tenant implements each model and how to execute one
(model, dataset) leg end-to-end: load the config-driven shards, run that model's
`Simulation` on the shared PeerSim engine, collect per-node metrics, and write the
canonical `results/json/<model>_<dataset>.json`. It does **no** plotting and has
no CLI — `scripts/main.py` chooses the combinations and `scripts/plot_analyzer.py` does
the plotting. All model logic proper lives in the tenant packages
(`src/linear_sdca/`, `src/baselines/gadget/`, `src/nonlinear_sdca/`); this file is only the
dispatcher that maps a model name to its simulation and saves the result.

The three models differ only in: the simulation class + the protocol id their
metrics live under, and a few model-specific JSON fields (the non-linear tenant
records its projection + gamma). Those differences are captured in `_resolve()`.
"""

import os
import sys
import json
from pathlib import Path

# Ensure the repo root is importable so `src.*` and root-level `data.*` resolve.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import CONFIG, CODE_DIR
from src.peersim_python.core import Network
from src.peersim_python.logger import logger
from src.scripts.dataset_keywords import select_dataset
from src.scripts.data_sharding import load_shards

MODELS = ["sdca", "gadget", "nonlinear", "ndsvm"]


def _resolve(model: str):
    """Return the per-model pieces that differ between the tenants:

        (simulation_class, pid_attr, extra_json_fn)

    `pid_attr`     : the attribute on the simulation class holding the protocol id
                     whose per-node `metrics` we collect.
    `extra_json_fn(CONFIG) -> dict` : model-specific fields to fold into the JSON
                     (e.g. the non-linear tenant's projection dim/dist + gamma),
                     or None. Simulation classes are imported lazily so running
                     `sdca` never imports the kernel tenant's sklearn-heavy stack.
    """
    if model == "sdca":
        from src.linear_sdca.simulation import Simulation
        return Simulation, "SDCA_PID", None
    if model == "gadget":
        from src.baselines.gadget.simulation import GadgetSimulation
        return GadgetSimulation, "PEGASOS_PID", None
    if model == "nonlinear":
        from src.nonlinear_sdca.simulation import KernelSDCASimulation
        extra = lambda cfg: {
            "projection_dim":  cfg["PROJECTION_DIM"],
            "projection_dist": cfg["PROJECTION_DIST"],
            "rbf_gamma":       cfg["RBF_GAMMA"],
        }
        return KernelSDCASimulation, "KERNEL_PID", extra
    if model == "ndsvm":
        from src.baselines.mom_ndsvm.simulation import NDSVMSimulation
        extra = lambda cfg: {
            "ndsvm_L":        cfg["NDSVM_L"],
            "ndsvm_eta":      cfg["NDSVM_ETA"],
            "ndsvm_C":        "1/(LAMBDA*n_global)",
            "lambda":         cfg["LAMBDA"],
            "rbf_gamma":      cfg["RBF_GAMMA"],
            "ndsvm_features": cfg.get("NDSVM_FEATURES", "raw"),
            "anchor_design":  "random (Forero et al. Sec. 4.1)",
        }
        return NDSVMSimulation, "NDSVM_PID", extra
    raise SystemExit(f"Unknown model '{model}'. Choose from: {' '.join(MODELS)}")


def _save_json(model, all_metrics, avg_acc, total_comm, stop_cycles, extra_fn) -> Path:
    """Write the canonical metrics file the plot analyzer reads (one shape for all
    models; `extra_fn` folds in any model-specific fields)."""
    json_dir = CODE_DIR / "results" / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
    out = json_dir / f"{model}_{CONFIG['DATASET']}.json"
    payload = {
        "algorithm":           model,
        "dataset":             CONFIG["DATASET"],
        "num_workers":         CONFIG["NUM_WORKERS"],
        "avg_accuracy":        avg_acc,
        "total_comm_bytes":    total_comm,
        "per_node_stop_cycle": stop_cycles,
        "metrics":             all_metrics,
    }
    if extra_fn:
        payload.update(extra_fn(CONFIG))
    out.write_text(json.dumps(payload, indent=2))
    return out


def run_model(model: str, dataset: str = None, cycles: int = None) -> Path:
    """Run one model on one dataset end-to-end and return the metrics-JSON path.

    `dataset` is a keyword/name (selects + downloads on first use); if omitted,
    whatever `src/config.py` currently points at is used.
    """
    os.chdir(CODE_DIR)
    if dataset:
        select_dataset(dataset)
    sim_cls, pid_attr, extra_fn = _resolve(model)
    logger.info("picker", f"{model.upper()} run — dataset={CONFIG['DATASET']}")

    worker_data = load_shards(CONFIG)
    sim_cls(CONFIG, worker_data).run(cycles)

    pid = getattr(sim_cls, pid_attr)
    protos = [Network.get(i).getProtocol(pid) for i in range(CONFIG["NUM_WORKERS"])]
    all_metrics = [p.metrics for p in protos]

    accuracies = [p.accuracy() for p in protos]
    avg_acc = sum(accuracies) / len(accuracies)
    for i, acc in enumerate(accuracies):
        logger.info("picker", f"Node {i} test accuracy = {acc:.4f}")
    logger.info("picker", f"Average test accuracy across nodes = {avg_acc:.4f}")

    final_comm = [p.comm_bytes for p in protos]
    stop_cycles = [getattr(p, "stop_cycle", None) for p in protos]
    total_comm = sum(final_comm)
    logger.info("comm", f"Total communication cost = {total_comm:,} bytes "
                        f"({total_comm / 1e6:.2f} MB) over {len(final_comm)} nodes")

    out = _save_json(model, all_metrics, avg_acc, total_comm, stop_cycles, extra_fn)
    logger.info("picker", f"Metrics → {out}")
    return out
