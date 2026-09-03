from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent.parent   # repo root (config.py lives in src/)
RAW_DIR  = CODE_DIR / "data" / "raw"
DATA_DIR = CODE_DIR / "data" / "processed"

CONFIG = {
    # p2p setup
    "TOPOLOGY":      "random_kout",
    "EPOCHS":        1,
    "BASE_PORT":     6000,

    # SDCA hyperparameters
    "LAMBDA":        1e-4,
    "T0_FRACTION":   0.5,          # used by the model.py / own_network path only

    # Coordinate updates each node performs per gossip cycle. This sets the
    # computation-to-communication ratio of a PeerSim round; 5000 rounds at a
    # full local pass each would be hours of single-core work for no extra
    # information, since the run converges in the first few hundred.
    "SDCA_LOCAL_STEPS": 4096,
    # Damping on each dual step. 1.0 takes the exact SDCA step against the last
    # known global w; 1/NUM_WORKERS is the conservative CoCoA+ setting to fall
    # back on if a topology turns out to oscillate.
    "SDCA_STEP_SCALE":  1.0,

    # Dataset
    "SEED":          42,
    "TRAIN_PATH": DATA_DIR / "rcv1_train.binary",
    "TEST_PATH": DATA_DIR / "rcv1_test.binary",
    # Reported, not enforced: STOP_ON_THRESHOLD stays False so a run completes
    # its full cycle budget. Halting the moment the gap first dips below a
    # threshold would show only that it touched that value once.
    "GAP_THRESHOLD": 1e-4,
    "STOP_ON_THRESHOLD": False,
    "EVAL_EVERY": 10,              # cycles between global evaluations
    "ACTIVATION": "shuffle",
    "DATASET": "covtype",
    "COVTYPE_PATH": DATA_DIR / "covtype.libsvm.binary.scale",
    "TEST_FRACTION": 0.2,
    "NUM_WORKERS": 10,
    "WARM_START": True,
    "GOSSIP_K": 3,
    "ROUNDS": 5000,
}
