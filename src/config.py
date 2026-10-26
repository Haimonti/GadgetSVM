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
    "STOP_ON_THRESHOLD": True,     # per-node early stop; end the run once all stop
    "EVAL_EVERY": 10,              # cycles between global evaluations
    "ACTIVATION": "shuffle",
    "DATASET": "covtype",
    "COVTYPE_PATH": DATA_DIR / "covtype.uci.binary.scale",
    "GISETTE_PATH": DATA_DIR / "gisette.binary.scale",
    "REALSIM_PATH": DATA_DIR / "real-sim",
    "TEST_FRACTION": 0.2,
    "NUM_WORKERS": 10,
    "WARM_START": True,
    "GOSSIP_K": 3,
    "ROUNDS": 5000,
    "W8A_TRAIN_PATH": DATA_DIR / "w8a",
    "W8A_TEST_PATH": DATA_DIR / "w8a.t",
    "IJCNN_TRAIN_PATH": DATA_DIR / "ijcnn1",
    "IJCNN_TEST_PATH": DATA_DIR / "ijcnn1.t",
    "A9A_TRAIN_PATH": DATA_DIR / "a9a",
    "A9A_TEST_PATH": DATA_DIR / "a9a.t",
    "WEBSPAM_PATH": DATA_DIR / "webspam_unigram.svm",
    "CONVERGENCE_COUNT": 10,
    "OBJ_THRESHOLD": 0.01,
    "STOP_ON_PLATEAU": True,
    "PLATEAU_PATIENCE": 20,
    "PLATEAU_REL_EPS": 1e-3,
    "KERNEL": "rbf",
    "PROJECTION_DIM": 512,
    "PROJECTION_DIST": "sparse",
    "DSRR_TAU": 0.0,
    "RBF_GAMMA": 0.005,
    "NDSVM_L": 100,
    "NDSVM_ETA": 10.0,
    "NDSVM_QP_TOL": 1e-8,
    "NDSVM_QP_MAX_SWEEPS": 10000,
    "NDSVM_RESIDUAL_TOL": 1e-4,
    "NDSVM_FEATURES": "raw"   # RBF kernel width (gisette: 0.1 overfits->all SVs; 0.005 is a healthy value)     # DSRR dual-sparse regularizer -> reduced hinge margin (1 - tau); 0 = plain   # DSRR projection family: "gaussian" (N(0,1/m)) or "sparse" (hashing)     # DSRR reduced dimension m (x_hat = A x, A shared via SEED, never sent)     # RBF kernel width: K(a,b)=exp(-gamma||a-b||^2); tune per dataset    # non-linear tenant: kernel in the projected space (DSRR suggests none -> RBF)     # an eval counts as progress only if it beats the best-so-far by this fraction       # consecutive evaluations with no improvement before the plateau stop fires     # safety stop: end a run once the global metric stops improving   # GADGET's native objective-change tolerance (EPSILON_VAL in the Java source),
}
