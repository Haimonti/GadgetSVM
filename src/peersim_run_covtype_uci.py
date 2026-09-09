"""Run the gossip-SDCA experiment on the UCI Covertype dataset — one file.

This is the sibling of `peersim_run.py`, wired specifically to the UCI Covertype
build (`covtype.uci.binary.scale`, class 2 vs. the rest, features scaled to
[0,1]). It is self-contained: if that dataset file does not exist yet, it first
builds it (download + unzip + preprocess via `data/extract_data.py`), then runs
the exact same PeerSim pipeline as `peersim_run.py`.

    python src/peersim_run_covtype_uci.py        # full run (CONFIG["ROUNDS"] cap,
                                                 # ends when every node converges)
    python src/peersim_run_covtype_uci.py 60     # cap at 60 cycles (quick test)

Results land in results/peersim_run<N>_<mm-dd-yyyy>/ just like peersim_run.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main(cycles: int = None) -> None:
    from src.config import CONFIG, DATA_DIR

    # Point the pipeline at the UCI Covertype build.
    CONFIG["DATASET"] = "covtype"
    CONFIG["COVTYPE_PATH"] = DATA_DIR / "covtype.uci.binary.scale"

    # Build it if it isn't there yet (idempotent: download + extract + preprocess).
    if not CONFIG["COVTYPE_PATH"].exists():
        print(f"{CONFIG['COVTYPE_PATH'].name} not found — building it first ...")
        from data.extract_data import preprocess_covertype_uci
        preprocess_covertype_uci()

    # Run the same experiment driver as peersim_run.py, now on the UCI data.
    from src.peersim_run import run
    run(cycles)


if __name__ == "__main__":
    cli_cycles = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(cli_cycles)
