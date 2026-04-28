from pathlib import Path

# getting file path for current dir
CODE_DIR = Path(__file__).parent

# file path for raw data dir
RAW_DIR  = Path(__file__).parent / "raw"

# file path for extracted data dir
DATA_DIR = Path(__file__).parent / "data"

CONFIG = {
    # p2p setup
    "NUM_WORKERS":   5,
    "TOPOLOGY":      "ring",
    "ROUNDS":        10,
    "EPOCHS":        1,           # Lightning epochs per gossip round
    "BASE_PORT":     6000,

    # SDCA hyperparameters
    "LAMBDA":        1e-4,        # regularisation threshold
    "T0_FRACTION":   0.5,         # fraction of steps before primal averaging begins

    # Dataset
    "SEED":          42,
}
