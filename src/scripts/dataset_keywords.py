"""Shared dataset-keyword handling for the PeerSim entry point.

The unified driver ``src/scripts/main.py`` takes a single CLI keyword that selects the
dataset and, on first use, downloads and preprocesses it. The mapping and the
on-demand build logic live here so every model shares the same dataset handling.

    from src.scripts.dataset_keywords import DATASET_KEYWORDS, select_dataset
    select_dataset(sys.argv[1])          # sets CONFIG["DATASET"], builds if missing
"""

from src.config import CONFIG
from src.peersim_python.logger import logger

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


def select_dataset(keyword: str) -> None:
    """Map a CLI keyword to CONFIG['DATASET']; build the file on demand if missing.

    covtype, gisette, real-sim, w8a, ijcnn1, a9a and webspam are downloaded +
    preprocessed on first use (idempotent --- raw downloads are deleted once the
    LIBSVM file is built; w8a/ijcnn1/a9a are already LIBSVM so they only need
    downloading, and ijcnn1/webspam additionally need decompressing).
    """
    name = DATASET_KEYWORDS.get(keyword.lower())
    if name is None:
        valid = " | ".join(DATASET_KEYWORDS)
        raise SystemExit(f"Unknown dataset '{keyword}'. Choose one of: {valid}")
    CONFIG["DATASET"] = name
    logger.info("main", f"Dataset '{keyword}' -> {name}")

    if name == "covtype" and not CONFIG["COVTYPE_PATH"].exists():
        from src.scripts.extract_data import preprocess_covertype_uci
        preprocess_covertype_uci()
    elif name == "gisette" and not CONFIG["GISETTE_PATH"].exists():
        from src.scripts.extract_data import preprocess_gisette
        preprocess_gisette()
    elif name == "real-sim" and not CONFIG["REALSIM_PATH"].exists():
        from src.scripts.extract_data import preprocess_real_sim
        preprocess_real_sim()
    elif name == "w8a" and not CONFIG["W8A_TRAIN_PATH"].exists():
        from src.scripts.extract_data import preprocess_w8a
        preprocess_w8a()
    elif name == "ijcnn1" and not CONFIG["IJCNN_TRAIN_PATH"].exists():
        from src.scripts.extract_data import preprocess_ijcnn1
        preprocess_ijcnn1()
    elif name == "a9a" and not CONFIG["A9A_TRAIN_PATH"].exists():
        from src.scripts.extract_data import preprocess_a9a
        preprocess_a9a()
    elif name == "webspam" and not CONFIG["WEBSPAM_PATH"].exists():
        from src.scripts.extract_data import preprocess_webspam
        preprocess_webspam()
