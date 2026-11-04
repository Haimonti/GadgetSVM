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

# central_sdca is on hold (being redesigned): not in MODELS, so no run includes it
MODELS = ["sdca", "gadget", "cola", "choco", "nonlinear", "ndsvm", "fed_ksvm"]



def parse_overrides(items) -> dict:
    """['TOPOLOGY=ring', 'LAMBDA=1e-5'] -> {'TOPOLOGY': 'ring', 'LAMBDA': 1e-5}.

    Values are parsed as Python literals when possible (numbers, None, True,
    lists), otherwise kept as strings. Unknown keys are rejected so a typo never
    silently runs the default config.
    """
    import ast
    out = {}
    for item in items or []:
        key, sep, raw = item.partition("=")
        if not sep:
            raise SystemExit(f"--set expects KEY=VAL, got '{item}'")
        if key not in CONFIG:
            raise SystemExit(f"--set: unknown CONFIG key '{key}'")
        try:
            out[key] = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            out[key] = raw
    return out


def _resolve(model: str):
    """Return the per-model pieces that differ between the tenants:

        (runner_class, pid_attr, extra_json_fn)

    `pid_attr`     : the attribute on the simulation class holding the protocol id
                     whose per-node `metrics` we collect, or None for a
                     standalone (non-PeerSim) model whose `run()` returns the
                     per-node results itself (Forero, Fed-KSVM).
    `extra_json_fn(CONFIG) -> dict` : model-specific fields to fold into the JSON
                     (e.g. the non-linear tenant's projection dim/dist + gamma),
                     or None. Simulation classes are imported lazily so running
                     `sdca` never imports the kernel tenant's sklearn-heavy stack.
    """
    if model == "sdca":
        from src.linear_sdca.simulation import Simulation
        return Simulation, "SDCA_PID", None
    if model == "central_sdca":
        from src.baselines.central_sdca.simulation import CentralSDCASimulation
        return CentralSDCASimulation, "CENTRAL_PID", None
    if model == "gadget":
        from src.baselines.gadget.simulation import GadgetSimulation
        return GadgetSimulation, "PEGASOS_PID", None
    if model == "cola":
        from src.baselines.cola.simulation import ColaSimulation
        extra = lambda cfg: {"cola_gamma": cfg["COLA_GAMMA"], "cola_local_iters": cfg["COLA_LOCAL_ITERS"],
                             "cola_C": "1/(LAMBDA*n_global)", "compressor": cfg["COMPRESSOR"]}
        return ColaSimulation, "COLA_PID", extra
    if model == "choco":
        from src.baselines.choco.simulation import ChocoSimulation
        extra = lambda cfg: {"choco_compressor": cfg["CHOCO_COMPRESSOR"],
                             "choco_consensus_lr": cfg["CHOCO_CONSENSUS_LR"],
                             "choco_lr": f"{cfg['CHOCO_LR_TYPE']}({cfg['CHOCO_INITIAL_LR']})"}
        return ChocoSimulation, "CHOCO_PID", extra
    if model == "nonlinear":
        from src.nonlinear_sdca.simulation import KernelSDCASimulation
        extra = lambda cfg: {
            "projection_dim":  cfg["PROJECTION_DIM"],
            "projection_dist": cfg["PROJECTION_DIST"],
            "rbf_gamma":       cfg["RBF_GAMMA"],
        }
        return KernelSDCASimulation, "KERNEL_PID", extra
    if model == "ndsvm":
        # Forero et al. MoM-NDSVM — standalone, in the paper's own setup (no PeerSim)
        from src.baselines.forero.algorithm import ForeroRun
        extra = lambda cfg: {
            "ndsvm_L":        cfg["NDSVM_L"],
            "ndsvm_eta":      cfg["NDSVM_ETA"],
            "lambda":         cfg["LAMBDA"],
            "kernel":         cfg.get("KERNEL", "rbf"),
            "rbf_gamma":      cfg["RBF_GAMMA"],
            "ndsvm_features": cfg.get("NDSVM_FEATURES", "raw"),
            "anchor_design":  "random (Forero et al. Sec. 4.1)",
        }
        return ForeroRun, None, extra
    if model == "fed_ksvm":
        # Zhou & Wang Fed-KSVM — standalone federated setup (1 server + M devices)
        from src.baselines.fed_ksvm.algorithm import FedKSVMRun
        return FedKSVMRun, None, lambda cfg: {"rbf_gamma": cfg["RBF_GAMMA"], "lambda": cfg["LAMBDA"]}
    raise SystemExit(f"Unknown model '{model}'. Choose from: {' '.join(MODELS)}")


def _save_json(model, cfg, all_metrics, avg_acc, total_comm, stop_cycles, extra_fn,
               out_path=None) -> Path:
    """Write the canonical metrics file the plot analyzer reads (one shape for all
    models; `extra_fn` folds in any model-specific fields). The full effective
    config is stored too, so a run is reproducible from its JSON alone.

    Default path: results/json/<model>_<dataset>_seed<k>.json (a seed per file,
    so repeated runs never overwrite each other)."""
    if out_path is None:
        out_path = CODE_DIR / "results" / "json" / f"{model}_{cfg['DATASET']}_seed{cfg['SEED']}.json"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "algorithm":           model,
        "dataset":             cfg["DATASET"],
        "seed":                cfg["SEED"],
        "run_tag":             cfg.get("RUN_TAG"),
        "num_workers":         cfg["NUM_WORKERS"],
        "topology":            cfg["TOPOLOGY"],
        "avg_accuracy":        avg_acc,
        "total_comm_bytes":    total_comm,
        "per_node_stop_cycle": stop_cycles,
        "config":              {k: (str(v) if isinstance(v, Path) else v) for k, v in cfg.items()},
        "metrics":             all_metrics,
    }
    if extra_fn:
        payload.update(extra_fn(cfg))
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    return out_path


def _stop_reason(model, cfg, all_metrics, budget):
    """Why the run ended, from its last evaluation:

    converged — every node met the model's own criterion (gap < GAP_THRESHOLD,
                Forero's residual latch, GADGET's objective-change count, ...)
    plateau   — stopped early because the convergence metric stopped improving
    budget    — used every round without either: NOT converged (raise ROUNDS or check
                the model's thresholds)
    fixed_budget — CHOCO, which by design runs a fixed number of SGD steps
    """
    last = [m[-1] for m in all_metrics if m]
    rounds = max((r["round"] for r in last), default=0)
    early = budget is not None and rounds < budget
    all_stopped = bool(last) and all(r.get("stopped") for r in last)
    gaps = [r["duality_gap"] for r in last if r.get("duality_gap") is not None]
    # each model is judged by the threshold it actually stops on (central SDCA: CENTRAL_GAP_TOL)
    thr = cfg.get("CENTRAL_GAP_TOL", 1e-6) if model == "central_sdca" else cfg["GAP_THRESHOLD"]
    met_gap = bool(gaps) and max(gaps) < thr
    latched = bool(last) and all(r.get("latched", False) for r in last)
    # gossip models latch per node on their own criterion; all latched = converged
    if early and (met_gap or latched or (all_stopped and model in ("sdca", "nonlinear", "gadget"))):
        reason = "converged"
    elif early or (all_stopped and not met_gap and budget is None):
        reason = "plateau"
    elif met_gap or latched:
        reason = "converged"
    else:
        reason = "budget"
    if reason == "budget" and model == "choco":
        reason = "fixed_budget"                  # CHOCO's protocol: a fixed number of SGD steps
    if reason == "budget":
        logger.info("picker", f"WARNING {model} on {cfg['DATASET']} seed {cfg['SEED']}: used all "
                              f"{rounds} rounds without converging or plateauing — raise ROUNDS "
                              f"or check its thresholds")
    return {"stop_reason": reason, "rounds_run": rounds, "round_budget": budget}

def run_model(model: str, dataset: str = None, cycles: int = None,
              overrides: dict = None, out_path=None) -> Path:
    """Run one model on one dataset end-to-end and return the metrics-JSON path.

    `dataset` is a keyword/name (selects + downloads on first use); if omitted,
    whatever `src/config.py` currently points at is used. `overrides` is applied
    to a *copy* of CONFIG (e.g. {"SEED": 3, "TOPOLOGY": "ring"}), so the global
    config is never mutated by a run. `out_path` overrides the JSON location
    (the ablation runner uses it). The JSON records *why* the run ended
    (`stop_reason`): converged, plateau, or budget (= did not converge).
    """
    os.chdir(CODE_DIR)
    if dataset:
        select_dataset(dataset)
    cfg = dict(CONFIG)
    cfg.update(overrides or {})
    runner, pid_attr, extra_fn = _resolve(model)
    logger.info("picker", f"{model.upper()} run — dataset={cfg['DATASET']} seed={cfg['SEED']}"
                          + (f" overrides={overrides}" if overrides else ""))

    worker_data = load_shards(cfg)
    runner_obj = runner(cfg, worker_data)
    if pid_attr is None:
        # standalone model (no PeerSim): run() returns the per-node results itself
        protos = runner_obj.run(cycles)
    else:
        runner_obj.run(cycles)
        pid = getattr(runner, pid_attr)
        protos = [Network.get(i).getProtocol(pid) for i in range(Network.size())]
    run_extra = runner_obj.extra_json() if hasattr(runner_obj, "extra_json") else {}
    all_metrics = [p.metrics for p in protos]
    run_extra.update(_stop_reason(model, cfg, all_metrics, getattr(runner_obj, "budget", None)))

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

    out = _save_json(model, cfg, all_metrics, avg_acc, total_comm, stop_cycles,
                     lambda c: {**(extra_fn(c) if extra_fn else {}), **run_extra}, out_path)
    logger.info("picker", f"Metrics → {out}")
    return out
