"""Config-driven data sharding for the PeerSim-Python simulation.

This lives OUTSIDE the engine (`src/peersim_python/` never touches datasets). It
reads CONFIG, loads the chosen dataset, and splits it into exactly
``config["NUM_WORKERS"]`` shards — one per node. The orchestrator receives these
shards and hands shard *i* to node *i*; it never loads data itself.

Kept p2pfl-free (unlike `data/data_loader.py`, which imports p2pfl at module
load) so the PeerSim path runs without a p2pfl install.

    shards = load_shards(CONFIG)   # -> list of NUM_WORKERS dicts
"""

import numpy as np
from sklearn.datasets import load_svmlight_file

from src.peersim_python.logger import logger


def _to_pm1(y):
    """Map a two-class label vector to {-1, +1} (larger class value -> +1).

    covtype uses {1, 2}; rcv1 uses {-1, +1}. Kept local so this module stays
    p2pfl-free.
    """
    y = np.asarray(y, dtype=np.float32)
    vals = np.unique(y)
    if len(vals) == 2:
        return np.where(y == vals.max(), 1.0, -1.0).astype(np.float32)
    return np.sign(y).astype(np.float32)


SHARDINGS = ("iid", "dirichlet", "label_sorted", "kmeans")


def _split_indices(X_train, y_train, n_workers, config=None):
    """Assign training rows to workers according to CONFIG["SHARDING"].

    iid          : contiguous equal chunks of the (already shuffled) data — default.
    dirichlet    : label skew; for each class, its rows are split across workers
                   with proportions ~ Dirichlet(DIRICHLET_ALPHA) (small alpha =
                   more skewed). Redrawn until no worker is empty.
    label_sorted : rows sorted by label, then equal chunks — the extreme case
                   (Forero's "biased" split, CHOCO's `label-sorted`).
    kmeans       : Fed-KSVM's non-IID protocol — k-means into SHARDING_CLUSTERS
                   clusters, workers evenly split into that many groups, and each
                   cluster's rows spread over its group.
    """
    cfg = config or {}
    mode = cfg.get("SHARDING", "iid")
    n = X_train.shape[0]
    rng = np.random.default_rng([cfg.get("SEED", 0), 7])
    if mode == "iid":
        return np.array_split(np.arange(n), n_workers)
    if mode == "label_sorted":
        return np.array_split(np.argsort(y_train, kind="stable"), n_workers)
    if mode == "dirichlet":
        alpha = cfg.get("DIRICHLET_ALPHA", 1.0)
        for _ in range(100):
            parts = [[] for _ in range(n_workers)]
            for c in np.unique(y_train):
                idx = rng.permutation(np.flatnonzero(y_train == c))
                cuts = (np.cumsum(rng.dirichlet(alpha * np.ones(n_workers)))[:-1]
                        * len(idx)).astype(int)
                for w, chunk in enumerate(np.split(idx, cuts)):
                    parts[w].extend(chunk.tolist())
            if all(parts):
                return [np.sort(np.array(p)) for p in parts]
        raise ValueError(f"dirichlet sharding (alpha={alpha}) left a worker empty "
                         f"after 100 draws; use a larger DIRICHLET_ALPHA")
    if mode == "kmeans":
        from sklearn.cluster import MiniBatchKMeans
        k = cfg.get("SHARDING_CLUSTERS", 2)
        if k > n_workers:
            raise ValueError(f"SHARDING_CLUSTERS={k} exceeds NUM_WORKERS={n_workers}")
        labels = MiniBatchKMeans(n_clusters=k, random_state=cfg.get("SEED", 0),
                                 n_init=3).fit_predict(X_train)
        parts = []
        for c, group in enumerate(np.array_split(np.arange(n_workers), k)):
            parts.extend(np.array_split(np.flatnonzero(labels == c), len(group)))
        if not all(len(p) for p in parts):
            raise ValueError("kmeans sharding left a worker empty; lower SHARDING_CLUSTERS")
        return parts
    raise ValueError(f"Unknown SHARDING '{mode}'. Choose: {' | '.join(SHARDINGS)}")


def _partition(X_train, y_train, X_test, y_test, n_workers, config=None):
    """Shard the training data; give every worker the SAME evaluation set.

    Training data is private and must be disjoint — that is the point of the
    setup. Test data is not: splitting it too made each worker's accuracy a
    measurement of a different quantity, so the spread between workers mixed
    "the models disagree" with "they were graded on different papers", and the
    mean was not any model's accuracy on the full test set.
    """
    tr = _split_indices(X_train, y_train, n_workers, config)
    X_test = X_test.tocsr()
    data = []
    for i in range(n_workers):
        data.append({
            "X_csr":   X_train[tr[i]].tocsr(),
            "y":       y_train[tr[i]],
            "X_test":  X_test,
            "y_test":  y_test,
            "n_local": len(tr[i]),
        })
        logger.info("data", f"Worker {i}: {len(tr[i])} train, {len(y_test)} test")
    return data






def load_shards(config):
    """Dispatch on config['DATASET'] and return exactly NUM_WORKERS data shards."""
    ds = config.get("DATASET", "rcv1")
    seed = config["SEED"]
    n_workers = config["NUM_WORKERS"]
    test_frac = config["TEST_FRACTION"]
    if ds == "rcv1":
        return load_two_files(config["TRAIN_PATH"], config["TEST_PATH"], n_workers, seed, "rcv1", config=config)
    if ds == "covtype":
        return load_single_file(config["COVTYPE_PATH"], n_workers, seed, test_frac, "covtype", config=config)
    if ds == "gisette":
        return load_single_file(config["GISETTE_PATH"], n_workers, seed, test_frac, "gisette", config=config)
    if ds == "real-sim":
        return load_single_file(config["REALSIM_PATH"], n_workers, seed, test_frac, "real-sim", config=config)
    if ds == "w8a":
        return load_two_files(config["W8A_TRAIN_PATH"], config["W8A_TEST_PATH"], n_workers, seed, "w8a", config=config)
    if ds == "ijcnn1":
        return load_two_files(config["IJCNN_TRAIN_PATH"], config["IJCNN_TEST_PATH"], n_workers, seed, "ijcnn1", config=config)
    if ds == "a9a":
        return load_two_files(config["A9A_TRAIN_PATH"], config["A9A_TEST_PATH"], n_workers, seed, "a9a", config=config)
    if ds == "webspam":
        return load_single_file(config["WEBSPAM_PATH"], n_workers, seed, test_frac, "webspam", config=config)
    raise ValueError(
        f"Unknown DATASET '{ds}'. Choose: rcv1 | covtype | gisette | real-sim | "
        f"w8a | ijcnn1 | a9a | webspam"
    )

def load_single_file(path, n_workers, seed, test_fraction, tag="dataset", config=None):
    """One LIBSVM file: shuffle, hold out `test_fraction` as test, then partition.

    Used by every single-file binary dataset (covtype, gisette, real-sim). The
    labels are mapped to +/-1 and the training data is split disjointly across
    workers while the test set is shared (see _partition).
    """
    rng = np.random.RandomState(seed)
    X, y = load_svmlight_file(str(path))
    y = _to_pm1(y)
    n = X.shape[0]
    perm = rng.permutation(n)
    X = X[perm].tocsr()
    y = y[perm]
    n_test = int(test_fraction * n)
    logger.info("data", f"{tag}: {n} samples, {X.shape[1]} features -> "
                        f"{n - n_test} train / {n_test} test")
    return _partition(
        X[n_test:].tocsr(), y[n_test:], X[:n_test].tocsr(), y[:n_test], n_workers, config
    )

def load_two_files(train_path, test_path, n_workers, seed, tag="dataset", config=None):
    """Two separate LIBSVM files (train + test), e.g. rcv1, w8a, ijcnn1.

    Labels are mapped to +/-1, the training rows are shuffled then split
    disjointly across workers, and the test file is loaded with the training
    feature count so both matrices have the same width (see _partition).
    """
    rng = np.random.RandomState(seed)
    X_train, y_train = load_svmlight_file(str(train_path))
    y_train = _to_pm1(y_train)
    perm = rng.permutation(X_train.shape[0])
    X_train = X_train[perm].tocsr()
    y_train = y_train[perm]
    X_test, y_test = load_svmlight_file(str(test_path), n_features=X_train.shape[1])
    y_test = _to_pm1(y_test)
    logger.info("data", f"{tag}: {X_train.shape[0]} train, {X_test.shape[0]} test, "
                        f"{X_train.shape[1]} features")
    return _partition(X_train, y_train, X_test, y_test, n_workers, config)
