#!/usr/bin/env bash
# Run the P2P experiment grid on this machine.
#
#   ./run_grid.sh              # everything, jobs = cores - 2
#   ./run_grid.sh 22           # 22 parallel workers
#   ./run_grid.sh 22 covtype   # covtype only  (or: rcv1)
#
# Resumable: rerun the same command after an interruption and it picks up
# where it stopped. Results land in results/grid_covtype.csv and
# results/grid_rcv1.csv.

set -euo pipefail
cd "$(dirname "$0")"

JOBS="${1:-}"
ONLY="${2:-all}"

# python3 on Linux/macOS, python on Windows (Git Bash / WSL)
PY=$(command -v python3 || command -v python)
[ -n "$PY" ] || { echo "no python found on PATH"; exit 1; }

if [ -z "$JOBS" ]; then
  CORES=$("$PY" -c "import os; print(os.cpu_count() or 4)")
  JOBS=$(( CORES > 2 ? CORES - 2 : 1 ))
fi

# The datasets are ~1.3 GB extracted; fetch them once if they are not there.
if [ ! -f data/covtype.libsvm.binary.scale ] || [ ! -f data/rcv1_train.binary ]; then
  echo "== fetching datasets =="
  "$PY" -c "import benchmark; benchmark.download_all('./data')"
fi

echo "== running grid: jobs=$JOBS set=$ONLY =="
"$PY" run_grid.py --jobs "$JOBS" --only "$ONLY"

echo
echo "== summary =="
"$PY" - <<'EOF'
import csv, glob, os
from collections import defaultdict
for f in sorted(glob.glob("results/grid_*.csv")):
    rows = list(csv.DictReader(open(f)))
    if not rows:
        continue
    g = defaultdict(list)
    for r in rows:
        g[(r["method"], r["gossip_k"], r["scheme"])].append(
            (float(r["test_acc"]), int(r["components"])))
    print(f"\n{os.path.basename(f)}  ({len(rows)} rows)")
    print(f"{'method':>12} {'k':>2} " +
          " ".join(f"{s:>15}" for s in sorted({k[2] for k in g})))
    for m in sorted({k[0] for k in g}):
        for k in sorted({kk[1] for kk in g}):
            cells = []
            for s in sorted({kk[2] for kk in g}):
                v = g[(m, k, s)]
                if v:
                    acc = sum(x[0] for x in v) / len(v)
                    dis = "*" if max(x[1] for x in v) > 1 else " "
                    cells.append(f"{acc:>14.4f}{dis}")
                else:
                    cells.append(f"{'-':>15}")
            print(f"{m:>12} {k:>2} " + " ".join(cells))
    print("  * = overlay was disconnected for at least one run in that cell")
EOF
