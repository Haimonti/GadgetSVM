from pathlib import Path

CODE_DIR = Path(__file__).parent
RAW_DIR  = CODE_DIR / "data" / "raw"
DATA_DIR = CODE_DIR / "data" / "processed"

CONFIG = {
    # p2p setup
    "TOPOLOGY":      "random_kout",
    "EPOCHS":        1,
    "BASE_PORT":     6000,

    # SDCA hyperparameters
    "LAMBDA":        1e-4,
    "T0_FRACTION":   0.5,

    # Dataset
    "SEED":          42,
    "NUM_WORKERS": 2,
    "ROUNDS": 100,
    "GOSSIP_K": 1,
    "TRAIN_PATH": DATA_DIR / "rcv1_train.binary",
    "TEST_PATH": DATA_DIR / "rcv1_test.binary",
    "GAP_THRESHOLD": 0.001,
    "ACTIVATION": "shuffle",
    "DATASET": "covtype",
    "COVTYPE_PATH": DATA_DIR / "covtype.libsvm.binary.scale",
    "TEST_FRACTION": 0.2,
}
