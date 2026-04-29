import numpy as np
import torch
import datasets
from pathlib import Path
from sklearn.datasets import load_svmlight_file

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
