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


def _partition(X_train, y_train, X_test, y_test, n_workers):
    """Partition train across workers and split test equally across workers."""
    tr = np.array_split(np.arange(X_train.shape[0]), n_workers)
    te = np.array_split(np.arange(X_test.shape[0]), n_workers)
    data = []
    for i in range(n_workers):
        data.append({
            "X_csr":   X_train[tr[i]].tocsr(),
            "y":       y_train[tr[i]],
            "X_test":  X_test[te[i]].tocsr(),
            "y_test":  y_test[te[i]],
            "n_local": len(tr[i]),
        })
        logger.info("data", f"Worker {i}: {len(tr[i])} train, {len(te[i])} test")
    return data


def load_rcv1(train_path, test_path, n_workers, seed):
    """rcv1: two LIBSVM files (separate train/test)."""
    rng = np.random.RandomState(seed)
    X_train, y_train = load_svmlight_file(str(train_path))
    y_train = _to_pm1(y_train)
    perm = rng.permutation(X_train.shape[0])
    X_train = X_train[perm].tocsr()
    y_train = y_train[perm]
    X_test, y_test = load_svmlight_file(str(test_path), n_features=X_train.shape[1])
    y_test = _to_pm1(y_test)
    logger.info("data", f"rcv1: {X_train.shape[0]} train, {X_test.shape[0]} test, "
                        f"{X_train.shape[1]} features")
    return _partition(X_train, y_train, X_test, y_test, n_workers)


def load_covtype(path, n_workers, seed, test_fraction):
    """covtype: one LIBSVM file — hold out `test_fraction` as test, then partition."""
    rng = np.random.RandomState(seed)
    X, y = load_svmlight_file(str(path))
    y = _to_pm1(y)
    n = X.shape[0]
    perm = rng.permutation(n)
    X = X[perm].tocsr()
    y = y[perm]
    n_test = int(test_fraction * n)
    logger.info("data", f"covtype: {n} samples, {X.shape[1]} features -> "
                        f"{n - n_test} train / {n_test} test")
    return _partition(
        X[n_test:].tocsr(), y[n_test:], X[:n_test].tocsr(), y[:n_test], n_workers
    )


def load_shards(config):
    """Dispatch on config['DATASET'] and return exactly NUM_WORKERS data shards."""
    ds = config.get("DATASET", "rcv1")
    if ds == "covtype":
        return load_covtype(
            config["COVTYPE_PATH"], config["NUM_WORKERS"],
            config["SEED"], config["TEST_FRACTION"],
        )
    if ds == "rcv1":
        return load_rcv1(
            config["TRAIN_PATH"], config["TEST_PATH"],
            config["NUM_WORKERS"], config["SEED"],
        )
    raise ValueError(f"Unknown DATASET '{ds}'. Choose: covtype | rcv1")
