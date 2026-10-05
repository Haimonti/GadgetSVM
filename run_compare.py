"""Run P2P-SDCA, P2P-BDSVM, P2P-FedAvg and P2P-CoCoA(+) under one harness, on one set of shards.

Both learners are driven from the *same* `src/config.py` (Sreekar's file,
unmodified), the same seed, the same topology, and the same shard list — the
shards are loaded once and handed to both, so the two runs see byte-identical
data rather than two draws that merely agree in distribution.

Each method writes its per-node metrics to
`results/compare_<mm-dd-yyyy>/<method>_metrics.json`, which `plot_merged.py`
reads to draw the combined figure.

    python run_compare.py            # both methods on CONFIG["DATASET"]
    python run_compare.py cov        # covtype     (sreekar@5b8aca8 keywords)
    python run_compare.py gis        # gisette
    python run_compare.py rsim       # real-sim
    python run_compare.py rcv        # rcv1
    python run_compare.py w8a | ijcnn | adult | webspam      (sreekar@1f73416)
    python run_compare.py gis --only bdsvm --cycles 20      # quick test
    python run_compare.py cov --only fedavg                 # one learner
    python run_compare.py cov --only cocoa cocoa_plus       # several learners

The dataset keywords are Sreekar's, taken verbatim from `src/peersim_run.py`
so the two drivers are invoked the same way and select the same data. Nothing
else about the run changes with the keyword: same seed, same topology, same
sharding, same early stopping.

BDSVM's pre-image budget and kind DO change with the dataset — a dense
pre-image cannot discriminate in a sparse 20k-dimensional space. That mapping
lives in `src/bdsvm_simulation.py`.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import CONFIG, CODE_DIR, DATA_DIR
from src.data_sharding import load_shards
from src.peersim_python.core import Network
from src.peersim_python.logger import logger
from src.peersim_python.simulation import Simulation
from src.sdca_observers import StallAwareSimulation
from src.bdsvm_simulation import BDSVMSimulation
from src.fedavg_simulation import FedAvgSimulation
from src.cocoa_simulation import CoCoASimulation


# Sreekar's keyword convention (src/peersim_run.py @ 5b8aca8), kept identical
# so `run_compare.py gis` and `peersim_run.py gis` mean the same thing.
DATASET_KEYWORDS = {
    "rcv":     "rcv1",
    "cov":     "covtype",
    "gis":     "gisette",
    "rsim":    "real-sim",
    "w8a":     "w8a",
    "ijcnn":   "ijcnn1",
    "adult":   "a9a",
    "webspam": "webspam",
}
# CONFIG keys whose files must exist before a run; two-file datasets list both.
DATASET_PATH_KEYS = {
    "covtype":  ["COVTYPE_PATH"],
    "gisette":  ["GISETTE_PATH"],
    "real-sim": ["REALSIM_PATH"],
    "rcv1":     ["TRAIN_PATH", "TEST_PATH"],
    "w8a":      ["W8A_TRAIN_PATH", "W8A_TEST_PATH"],
    "ijcnn1":   ["IJCNN_TRAIN_PATH", "IJCNN_TEST_PATH"],
    "a9a":      ["A9A_TRAIN_PATH", "A9A_TEST_PATH"],
    "webspam":  ["WEBSPAM_PATH"],
}


def _jsonable(v):
    """CONFIG holds Paths; JSON does not."""
    return str(v) if isinstance(v, Path) else v


def _collect(protos, method, stopped_at, cycles):
    per_node = [p.metrics for p in protos]
    accuracies = [p.accuracy() for p in protos]
    comm = [p.comm_bytes for p in protos]
    stop_cycles = [getattr(p, "stop_cycle", None) for p in protos]
    stop_reasons = [getattr(p, "stop_reason", None) for p in protos]
    total = sum(comm)
    mean = total / len(comm)
    std = (sum((c - mean) ** 2 for c in comm) / len(comm)) ** 0.5

    for i, a in enumerate(accuracies):
        logger.info(method, f"Node {i} test accuracy = {a:.4f}")
    avg = sum(accuracies) / len(accuracies)
    logger.info(method, f"Average test accuracy across nodes = {avg:.4f}")
    logger.info(method, f"Total communication = {total:,} bytes "
                        f"({total / 1e6:.2f} MB) over {len(comm)} nodes")
    logger.info(method, f"Node stop cycles: {stop_cycles}")
    if any(stop_reasons):
        logger.info(method, f"Node stop reasons: {stop_reasons}")

    return {
        "method": method,
        "cycles_requested": cycles,
        "stopped_at": stopped_at,
        "config": {k: _jsonable(v) for k, v in CONFIG.items()},
        "per_node_metrics": per_node,
        "per_node_accuracy": accuracies,
        "average_accuracy": avg,
        "per_node_comm_bytes": comm,
        "total_comm_bytes": total,
        "mean_comm_bytes_per_node": mean,
        "std_comm_bytes_per_node": std,
        "per_node_stop_cycle": stop_cycles,
        "per_node_stop_reason": stop_reasons,
    }


def run_sdca(worker_data, cycles):
    # Sreekar's Simulation, with one more way for a node to stop: its iterate
    # has stalled. The gap rule is unchanged; see src/sdca_observers.py.
    stopped_at = StallAwareSimulation(CONFIG, worker_data).run(cycles)
    protos = [Network.get(i).getProtocol(Simulation.SDCA_PID)
              for i in range(CONFIG["NUM_WORKERS"])]
    return _collect(protos, "sdca", stopped_at, cycles)


def run_fedavg(worker_data, cycles):
    # Gossip-FedAvg: local Pegasos, size-weighted model averaging, stall stop.
    # Same config, seed and shards as the other two; see src/fedavg_simulation.py.
    sim = FedAvgSimulation(CONFIG, worker_data)
    stopped_at = sim.run(cycles)
    protos = [Network.get(i).getProtocol(FedAvgSimulation.FEDAVG_PID)
              for i in range(CONFIG["NUM_WORKERS"])]
    out = _collect(protos, "fedavg", stopped_at, cycles)
    out["fedavg_stall_tol"] = sim.stall_tol
    return out


def run_cocoa(worker_data, cycles, variant="cocoa"):
    # Gossip-CoCoA(+): SDCA's contribution tables with CoCoA's commit rule,
    # stopped by SDCA's gap-or-stall rule; see src/cocoa_simulation.py.
    sim = CoCoASimulation(CONFIG, worker_data, variant=variant)
    stopped_at = sim.run(cycles)
    protos = [Network.get(i).getProtocol(CoCoASimulation.COCOA_PID)
              for i in range(CONFIG["NUM_WORKERS"])]
    out = _collect(protos, variant, stopped_at, cycles)
    out["cocoa_params"] = sim.params
    out["cocoa_stall_tol"] = sim.stall_tol
    return out


def _parse_overrides(items):
    out = {}
    for kv in items:
        k, _, v = kv.partition("=")
        try:
            out[k] = float(v) if "." in v or "e" in v.lower() else int(v)
        except ValueError:
            out[k] = v
    return out


def run_bdsvm(worker_data, cycles, overrides=None, method="bdsvm"):
    from src.bdsvm_simulation import params_for
    params = {**params_for(CONFIG["DATASET"]), **(overrides or {})}
    sim = BDSVMSimulation(CONFIG, worker_data, params=params)
    stopped_at = sim.run(cycles)
    protos = [Network.get(i).getProtocol(BDSVMSimulation.BDSVM_PID)
              for i in range(CONFIG["NUM_WORKERS"])]
    out = _collect(protos, method, stopped_at, cycles)
    out["bdsvm_params"] = sim.params
    out["bdsvm_eta"] = sim.eta
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, default=None,
                    help="cap on cycles (default: CONFIG['ROUNDS'])")
    ap.add_argument("--only", nargs="+", default=None,
                    choices=["sdca", "bdsvm", "bdsvm_linear", "fedavg", "cocoa", "cocoa_plus"])
    ap.add_argument("--out", default=None, help="results directory")
    ap.add_argument("--bdsvm", action="append", default=[], metavar="KEY=VALUE",
                    help="override a BDSVM hyperparameter, e.g. --bdsvm rho=0.9 "
                         "(repeatable); numbers are parsed, anything else is a string")
    ap.add_argument("dataset", nargs="?", default=None,
                    choices=sorted(DATASET_KEYWORDS),
                    help="dataset keyword (sreekar's convention); "
                         "omit to use CONFIG['DATASET']")
    args = ap.parse_args()

    if args.dataset:
        CONFIG["DATASET"] = DATASET_KEYWORDS[args.dataset]
    ds = CONFIG["DATASET"]
    paths = [CONFIG[k] for k in DATASET_PATH_KEYS.get(ds, [])]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise SystemExit(
            f"{missing[0]} not found. Build it on the sreekar checkout "
            f"(data/extract_data.py) and copy it into data/processed/.")
    logger.info("main", f"Dataset: {ds} -> {', '.join(p.name for p in paths)}")

    out_dir = Path(args.out) if args.out else (
        CODE_DIR / "results" /
        f"compare_{ds}_{datetime.now().strftime('%m-%d-%Y')}")
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("main", f"Comparison run — results → {out_dir}")

    # One load, both methods: identical shards, not merely identically drawn.
    worker_data = load_shards(CONFIG)

    methods = args.only or ["sdca", "bdsvm", "fedavg", "cocoa", "cocoa_plus"]
    for m in methods:
        logger.info("main", f"=== {m.upper()} ===")
        if m == "sdca":
            result = run_sdca(worker_data, args.cycles)
        elif m == "bdsvm":
            result = run_bdsvm(worker_data, args.cycles, _parse_overrides(args.bdsvm))
        elif m == "bdsvm_linear":
            result = run_bdsvm(worker_data, args.cycles,
                               {**_parse_overrides(args.bdsvm), "kernel": "linear"},
                               method=m)
        elif m == "fedavg":
            result = run_fedavg(worker_data, args.cycles)
        else:
            result = run_cocoa(worker_data, args.cycles, m)
        path = out_dir / f"{m}_metrics.json"
        path.write_text(json.dumps(result, indent=2))
        logger.info("main", f"{m} metrics → {path}")

    logger.info("main", "Done.")


if __name__ == "__main__":
    main()
