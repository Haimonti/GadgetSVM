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
    "OBJ_THRESHOLD": 0.01,         # GADGET's native objective-change tolerance (EPSILON_VAL in the Java source)
    "STOP_ON_PLATEAU": True,       # safety stop: end a run once the global metric stops improving
    "PLATEAU_PATIENCE": 20,        # consecutive evaluations with no improvement before the plateau stop fires
    "PLATEAU_REL_EPS": 1e-3,       # an eval counts as progress only if it beats the best-so-far by this fraction

    # Non-linear (kernel) SDCA tenant
    "KERNEL": "rbf",               # rbf | poly — the only two kernels in scope (nonlinear_sdca/kernel.py)
    "POLY_DEGREE": 3,              # polynomial kernel (g <a,b> + coef0)^degree
    "POLY_GAMMA": None,            # g; None = 1/n_features (sklearn default)
    "POLY_COEF0": 1.0,
    "PROJECTION_DIM": 512,         # DSRR reduced dimension m (x_hat = A x, A shared via SEED, never sent)
    "PROJECTION_DIST": "sparse",   # DSRR projection family: "gaussian" (N(0,1/m)) or "sparse" (hashing)
    "DSRR_TAU": 0.0,               # DSRR dual-sparse regularizer -> reduced hinge margin (1 - tau); 0 = plain
    "RBF_GAMMA": 0.005,            # RBF width K(a,b)=exp(-gamma||a-b||^2) (gisette: 0.1 overfits -> all SVs)
    "DSRR_RECOVERY": True,         # also score with the recovered (original-space) model: accuracy_recovered

    # MoM-NDSVM (Forero et al.) baseline
    "NDSVM_L": 100,
    "NDSVM_ETA": 10.0,
    "NDSVM_QP_TOL": 1e-8,
    "NDSVM_QP_MAX_SWEEPS": 10000,
    "NDSVM_RESIDUAL_TOL": 1e-4,
    "NDSVM_FEATURES": "raw",
    # Forero's own experimental setup (baselines/forero/network.py; paper Sec. 5)
    "FORERO_J": None,              # nodes; None = NUM_WORKERS
    "FORERO_NETWORK": "random",    # random (geometric, connected) | ring | full
    "FORERO_RADIUS": None,         # random-graph radius; None = smallest connected radius
    "FORERO_PARTITION": "balanced",  # balanced | biased (paper's two extremes) | given (our shards)
    "FORERO_COMM_MODEL": "broadcast",  # paper counts one broadcast per node per iteration; or "p2p"
    "FORERO_REFERENCES": True,     # also fit the paper's centralised + local-only SVM benchmarks
    # Fed-KSVM (Zhou & Wang, TNNLS 2024) — federated, own setup (baselines/fed_ksvm/)
    "FEDKSVM_D": 2000,             # random Fourier features (paper: larger D -> higher accuracy)
    "FEDKSVM_L": 10,               # feature blocks for block boosting (paper's experiments: 10)
    "FEDKSVM_Q": None,             # block-boosting iterations per round; None = L
    "FEDKSVM_H": 1.0,              # SDCA passes over the local data per block solve
    "FEDKSVM_BETA": 1.0,           # server step beta_M in (7); 1.0 = CoCoA averaging
    "FEDKSVM_SECURE_AGG": False,   # pairwise-mask secure aggregation (Sec. IV-C)

    # Experiment infrastructure (runs, ablations)
    "RUN_TAG": None,               # free-form label stored in each result JSON
    "SHARDING": "iid",             # iid | dirichlet | label_sorted | kmeans (see scripts/data_sharding.py)
    "DIRICHLET_ALPHA": 1.0,        # label-skew strength for SHARDING="dirichlet" (smaller = more skewed)
    "SHARDING_CLUSTERS": 2,        # k for SHARDING="kmeans" (Fed-KSVM's non-IID protocol)
    "CYCLE_CONNECTIVITY": 2,       # c for TOPOLOGY="connected_cycle" (links to c nodes on each side)
    "CENTRAL_GAP_TOL": 1e-6,       # centralized SDCA stops at this gap (P* itself comes from liblinear, exact)
    "STOP_RULE": "node_global_gap",  # SDCA early-stop gap: node_global_gap | node_local_gap | mean_gap
    "ABLATION_EPS": 1e-3,          # suboptimality target for "iterations to reach P(w) - P* <= eps"
    "COMPRESSOR": "full",          # message compression for the quantization ablation (peersim_python/compression.py)

    # COLA baseline (epfml/cola, LinearSVM)
    "COLA_GAMMA": 1.0,             # aggregation parameter; run_cola.py fixes 1.0 (sigma = gamma * K)
    "COLA_LOCAL_ITERS": 1.0,       # local solver passes over the shard per round (--local_iters)
    "COLA_THETA": 0.0,             # local solver early-exit tolerance (--theta); 0 = always run all passes
    "COLA_ROUNDS": 20000,          # COLA round budget (O(1/t) gap: ~7k rounds to 1e-4 on a9a); max with ROUNDS

    # CHOCO-SGD baseline (epfml/ChocoSGD convex code)
    "CHOCO_COMPRESSOR": "full",    # Q(.) for CHOCO-Gossip: full | top_k:f | random_k:f | qsgd:s ...
    "CHOCO_CONSENSUS_LR": 1.0,     # consensus step gamma (paper-tuned per compressor)
    "CHOCO_LR_TYPE": "decay",      # constant | decay (lr/(lambda(t+tau))) | bottou
    "CHOCO_INITIAL_LR": 0.1,
    "CHOCO_TAU": None,             # None = d, as in their experiments
    "CHOCO_BATCH": 1,              # samples per SGD step (official code: 1)
    "CHOCO_EPOCHS": 20,            # CHOCO fixed budget: max(epochs*n_local/batch, CHOCO_MIN_ROUNDS, ROUNDS)
    "CHOCO_MIN_ROUNDS": 50000,     # SGD needs steps, not epochs (tiny a9a: ~45k steps -> ~2% of P*)
}
