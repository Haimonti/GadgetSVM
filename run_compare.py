"""Run P2P-SDCA and P2P-BDSVM under one harness, on one set of shards.

Both learners are driven from the *same* `src/config.py` (Sreekar's file,
unmodified), the same seed, the same topology, and the same shard list — the
shards are loaded once and handed to both, so the two runs see byte-identical
data rather than two draws that merely agree in distribution.

Each method writes its per-node metrics to
`results/compare_<mm-dd-yyyy>/<method>_metrics.json`, which `plot_merged.py`
reads to draw the combined figure.

    python run_compare.py                    # both methods, CONFIG["ROUNDS"] cap
    python run_compare.py --cycles 20        # quick test
    python run_compare.py --only bdsvm       # one method
    python run_compare.py --dataset uci      # the UCI Covertype build

`--dataset uci` points COVTYPE_PATH at `covtype.uci.binary.scale` exactly the
way `peersim_run_covtype_uci.py` does on the sreekar branch — the same runtime
override of the same key, so "which data" is the only thing that changes.
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
from src.bdsvm_simulation import BDSVMSimulation


def _jsonable(v):
    """CONFIG holds Paths; JSON does not."""
    return str(v) if isinstance(v, Path) else v


def _collect(protos, method, stopped_at, cycles):
    per_node = [p.metrics for p in protos]
    accuracies = [p.accuracy() for p in protos]
    comm = [p.comm_bytes for p in protos]
    stop_cycles = [getattr(p, "stop_cycle", None) for p in protos]
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
    }


def run_sdca(worker_data, cycles):
    stopped_at = Simulation(CONFIG, worker_data).run(cycles)
    protos = [Network.get(i).getProtocol(Simulation.SDCA_PID)
              for i in range(CONFIG["NUM_WORKERS"])]
    return _collect(protos, "sdca", stopped_at, cycles)


def run_bdsvm(worker_data, cycles):
    sim = BDSVMSimulation(CONFIG, worker_data)
    stopped_at = sim.run(cycles)
    protos = [Network.get(i).getProtocol(BDSVMSimulation.BDSVM_PID)
              for i in range(CONFIG["NUM_WORKERS"])]
    out = _collect(protos, "bdsvm", stopped_at, cycles)
    out["bdsvm_params"] = sim.params
    out["bdsvm_eta"] = sim.eta
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, default=None,
                    help="cap on cycles (default: CONFIG['ROUNDS'])")
    ap.add_argument("--only", choices=["sdca", "bdsvm"], default=None)
    ap.add_argument("--out", default=None, help="results directory")
    ap.add_argument("--dataset", choices=["libsvm", "uci"], default="libsvm",
                    help="covtype build: the ready-made LIBSVM file (default) "
                         "or the UCI one built by data/extract_data.py")
    args = ap.parse_args()

    if args.dataset == "uci":
        CONFIG["COVTYPE_PATH"] = DATA_DIR / "covtype.uci.binary.scale"
        if not CONFIG["COVTYPE_PATH"].exists():
            raise SystemExit(f"{CONFIG['COVTYPE_PATH']} not found — build it "
                             f"with data/extract_data.py first")
    logger.info("main", f"Dataset: {CONFIG['COVTYPE_PATH']}")

    out_dir = Path(args.out) if args.out else (
        CODE_DIR / "results" /
        f"compare_{args.dataset}_{datetime.now().strftime('%m-%d-%Y')}")
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("main", f"Comparison run — results → {out_dir}")

    # One load, both methods: identical shards, not merely identically drawn.
    worker_data = load_shards(CONFIG)

    methods = [args.only] if args.only else ["sdca", "bdsvm"]
    for m in methods:
        logger.info("main", f"=== {m.upper()} ===")
        result = (run_sdca if m == "sdca" else run_bdsvm)(worker_data, args.cycles)
        path = out_dir / f"{m}_metrics.json"
        path.write_text(json.dumps(result, indent=2))
        logger.info("main", f"{m} metrics → {path}")

    logger.info("main", "Done.")


if __name__ == "__main__":
    main()
