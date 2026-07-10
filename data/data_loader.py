import sys
import numpy as np
import torch
import datasets
from pathlib import Path
from sklearn.datasets import load_svmlight_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from p2pfl.management.logger import logger


def load_data(train_path: Path, seed: int) -> tuple:
    """Load, shuffle, and return an svmlight dataset as (X_csr, y)."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    X_csr, y = load_svmlight_file(str(train_path))
    y = np.sign(y).astype(np.float32)

    n = X_csr.shape[0]
    perm  = np.random.permutation(n)
    X_csr = X_csr[perm].tocsr()
    y     = y[perm]

    logger.info("data_loader", f"Loaded {n} samples, {X_csr.shape[1]} features from {train_path.name}")
    return X_csr, y


def split_workers(X_csr, y, n_workers: int) -> list[dict]:
    """Partition data equally across workers and wrap each shard in an HF dataset."""
    n         = X_csr.shape[0]
    part_idxs = np.array_split(np.arange(n), n_workers)
    worker_data = []

    for i, local_idxs in enumerate(part_idxs):
        X_part  = X_csr[local_idxs].tocsr()
        y_part  = y[local_idxs]
        n_local = len(local_idxs)

        hf_split = datasets.Dataset.from_dict({
            "idx": list(range(n_local)),
            "y":   y_part.tolist(),
        })
        hf_ds = datasets.DatasetDict({"train": hf_split, "test": hf_split})

        worker_data.append({
            "hf_dataset": hf_ds,
            "X_csr":      X_part,
            "y":          y_part,
            "n_local":    n_local,
        })
        logger.info(
            "data_loader",
            f"Worker {i}: {n_local} samples  avg nnz/sample ≈ {X_part.nnz // n_local}"
        )

    return worker_data

def load_rcv1_partitions(train_path: Path, test_path: Path, n_workers: int, seed: int) -> list[dict]:
    """Load RCV1 train+test, partition train across n_workers, split test equally."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    X_train, y_train = load_svmlight_file(str(train_path))
    y_train = np.sign(y_train).astype(np.float32)
    n_train = X_train.shape[0]
    perm    = np.random.permutation(n_train)
    X_train = X_train[perm].tocsr()
    y_train = y_train[perm]
    logger.info("data_loader", f"Train: {n_train} samples, {X_train.shape[1]} features")

    X_test_full, y_test_full = load_svmlight_file(str(test_path), n_features=X_train.shape[1])
    y_test_full = np.sign(y_test_full).astype(np.float32)
    n_test = X_test_full.shape[0]
    logger.info("data_loader", f"Test:  {n_test} samples")

    train_parts = np.array_split(np.arange(n_train), n_workers)
    test_parts  = np.array_split(np.arange(n_test),  n_workers)

    worker_data = []
    for i in range(n_workers):
        tr_idx  = train_parts[i]
        te_idx  = test_parts[i]
        X_part  = X_train[tr_idx].tocsr()
        y_part  = y_train[tr_idx]
        X_te    = X_test_full[te_idx].tocsr()
        y_te    = y_test_full[te_idx]
        n_local = len(tr_idx)

        hf_split = datasets.Dataset.from_dict({
            "idx": list(range(n_local)),
            "y":   y_part.tolist(),
        })
        hf_ds = datasets.DatasetDict({"train": hf_split, "test": hf_split})

        worker_data.append({
            "hf_dataset": hf_ds,
            "X_csr":      X_part,
            "y":          y_part,
            "X_test":     X_te,
            "y_test":     y_te,
            "n_local":    n_local,
        })
        logger.info(
            "data_loader",
            f"Worker {i}: {n_local} train, {len(te_idx)} test  "
            f"avg nnz/sample ≈ {X_part.nnz // n_local}"
        )
    return worker_data

def to_pm1(y) -> np.ndarray:
    """Map a two-class label vector to {-1, +1} (larger class value -> +1).

    covtype.binary uses labels {1, 2}; rcv1 already uses {-1, +1}. This handles
    both: the max distinct label becomes +1, the min becomes -1. (np.sign is
    wrong for {1, 2} — it would map both classes to +1.)
    """
    y = np.asarray(y, dtype=np.float32)
    vals = np.unique(y)
    if len(vals) == 2:
        return np.where(y == vals.max(), 1.0, -1.0).astype(np.float32)
    return np.sign(y).astype(np.float32)


def load_covtype_partitions(path: Path, n_workers: int, seed: int,
                            test_fraction: float = 0.2) -> list[dict]:
    """Load covtype (a single LIBSVM file), split off a test set, then partition.

    covtype has no canonical train/test split, so we hold out `test_fraction`
    of the (shuffled) data as test, partition the remaining train across workers,
    and split the test set equally across workers — mirroring
    `load_rcv1_partitions`' output schema (incl. hf_dataset for the p2pfl path).
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    X, y = load_svmlight_file(str(path))
    y = to_pm1(y)
    n = X.shape[0]
    perm = np.random.permutation(n)
    X = X[perm].tocsr()
    y = y[perm]

    n_test = int(test_fraction * n)
    X_test_full, y_test_full = X[:n_test].tocsr(), y[:n_test]
    X_train, y_train = X[n_test:].tocsr(), y[n_test:]
    logger.info(
        "data_loader",
        f"covtype: {n} samples, {X.shape[1]} features -> "
        f"{X_train.shape[0]} train / {n_test} test",
    )

    train_parts = np.array_split(np.arange(X_train.shape[0]), n_workers)
    test_parts  = np.array_split(np.arange(n_test),           n_workers)

    worker_data = []
    for i in range(n_workers):
        tr_idx  = train_parts[i]
        te_idx  = test_parts[i]
        X_part  = X_train[tr_idx].tocsr()
        y_part  = y_train[tr_idx]
        X_te    = X_test_full[te_idx].tocsr()
        y_te    = y_test_full[te_idx]
        n_local = len(tr_idx)

        hf_split = datasets.Dataset.from_dict({
            "idx": list(range(n_local)),
            "y":   y_part.tolist(),
        })
        hf_ds = datasets.DatasetDict({"train": hf_split, "test": hf_split})

        worker_data.append({
            "hf_dataset": hf_ds,
            "X_csr":      X_part,
            "y":          y_part,
            "X_test":     X_te,
            "y_test":     y_te,
            "n_local":    n_local,
        })
        logger.info(
            "data_loader",
            f"Worker {i}: {n_local} train, {len(te_idx)} test  "
            f"avg nnz/sample ~ {X_part.nnz // max(1, n_local)}"
        )
    return worker_data
