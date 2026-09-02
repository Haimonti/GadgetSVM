"""Dataset download / loading / partitioning for the SVM benchmarks.

Reconstructed from its call sites in `run_benchmark.py` — the original file was
never committed to any branch. Public surface, unchanged from those call sites:

    download_all(data_dir)
    load_dataset(name, split="train"|"test", data_dir=...) -> (X_csr, y)
    partition(y, n_clients, scheme, seed=..., **kwargs)    -> list[np.ndarray]

Labels are always mapped to {-1, +1}. Features stay scipy CSR (rcv1 is 47k-d
sparse; densifying it is not an option).

Datasets are the LIBSVM binary collection:
  rcv1     rcv1_train.binary / rcv1_test.binary  (separate train & test files)
  covtype  covtype.libsvm.binary.scale           (single file, split 80/20)
"""
from __future__ import annotations

import bz2
import shutil
import subprocess
import urllib.request
from pathlib import Path

import numpy as np

LIBSVM_BASE = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary"

# name -> {split: (remote filename, local extracted filename)}
FILES = {
    "rcv1": {
        "train": ("rcv1_train.binary.bz2", "rcv1_train.binary"),
        "test":  ("rcv1_test.binary.bz2",  "rcv1_test.binary"),
    },
    "covtype": {
        "train": ("covtype.libsvm.binary.scale.bz2", "covtype.libsvm.binary.scale"),
        "test":  ("covtype.libsvm.binary.scale.bz2", "covtype.libsvm.binary.scale"),
    },
}

# covtype ships as one file; hold out the last 20% (post-shuffle) as test.
_SINGLE_FILE = {"covtype"}
_TEST_FRACTION = 0.2
_SPLIT_SEED = 0  # fixed so train/test never overlap across separate calls


# ── download ────────────────────────────────────────────────────────────────

def _fetch(remote: str, data_dir: Path) -> Path:
    """Download `remote` into data_dir/raw and decompress into data_dir. Idempotent."""
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    archive = raw_dir / remote
    target = data_dir / remote[: -len(".bz2")]
    # A zero-byte target means a previous extraction was interrupted: bz2 creates
    # the file before writing to it. Treat that as missing, not as done.
    if target.exists() and target.stat().st_size > 0:
        print(f"[{remote}] already extracted -> {target}")
        return target

    if not archive.exists():
        url = f"{LIBSVM_BASE}/{remote}"
        print(f"[{remote}] downloading {url} ...")
        _download(url, archive)

    print(f"[{remote}] extracting -> {target}", flush=True)
    tmp = target.with_suffix(target.suffix + ".part")
    with bz2.open(archive, "rb") as f_in, open(tmp, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    tmp.replace(target)          # atomic: the target only ever appears complete
    return target


def _download(url: str, dest: Path) -> None:
    """Fetch `url` to `dest`, falling back to curl.

    The LIBSVM host serves a certificate without a Subject Key Identifier, which
    newer OpenSSL builds reject outright — so urllib fails on some machines that
    can nonetheless reach the site fine. curl accepts the chain, so it is tried
    second rather than making the caller disable verification globally.
    """
    try:
        urllib.request.urlretrieve(url, dest)
        return
    except Exception as exc:
        print(f"    urllib failed ({exc}); retrying with curl")
    try:
        subprocess.run(["curl", "-fsSL", "--retry", "3", "-o", str(dest), url],
                       check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"could not download {url}: {exc}") from exc


def download_all(data_dir="./data") -> None:
    """Download + extract every dataset file (skipping what is already present)."""
    data_dir = Path(data_dir)
    for name, splits in FILES.items():
        for remote, _ in dict.fromkeys(splits.values()):  # dedupe single-file sets
            _fetch(remote, data_dir)


# ── loading ─────────────────────────────────────────────────────────────────

def _to_pm1(y):
    """Map a two-class label vector to {-1, +1} (larger class value -> +1)."""
    y = np.asarray(y, dtype=np.float64)
    vals = np.unique(y)
    if len(vals) == 2:
        return np.where(y == vals.max(), 1.0, -1.0)
    return np.sign(y)


_N_FEATURES = {}  # name -> feature count, so train/test matrices stay conformable


def load_dataset(name, split="train", data_dir="./data"):
    """Load one split as (X_csr, y) with y in {-1, +1}.

    rcv1's test file has fewer columns than its train file, so the feature count
    seen on train is pinned and reused — otherwise `X_test @ w` would not align.
    """
    if name not in FILES:
        raise ValueError(f"Unknown dataset '{name}'. Choose: {sorted(FILES)}")
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")

    from sklearn.datasets import load_svmlight_file  # only needed for file I/O

    data_dir = Path(data_dir)
    _, local = FILES[name][split]
    path = data_dir / local
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run with --download, or place the LIBSVM file there."
        )

    X, y = load_svmlight_file(str(path), n_features=_N_FEATURES.get(name))
    y = _to_pm1(y)
    _N_FEATURES.setdefault(name, X.shape[1])

    if name in _SINGLE_FILE:
        rng = np.random.RandomState(_SPLIT_SEED)
        perm = rng.permutation(X.shape[0])
        n_test = int(_TEST_FRACTION * X.shape[0])
        idx = perm[:n_test] if split == "test" else perm[n_test:]
        X, y = X[idx].tocsr(), y[idx]

    return X.tocsr(), y


# ── partitioning ────────────────────────────────────────────────────────────

def _iid(y, K, rng):
    idx = rng.permutation(len(y))
    return [np.sort(p) for p in np.array_split(idx, K)]


def _dirichlet(y, K, rng, alpha=1.0):
    """Per-class Dirichlet split — the standard non-IID FL benchmark protocol.

    For each class c, draw p ~ Dir(alpha * 1_K) and split class c's samples
    across clients in those proportions. Small alpha => highly skewed clients.
    """
    parts = [[] for _ in range(K)]
    for c in np.unique(y):
        idx_c = np.where(y == c)[0]
        rng.shuffle(idx_c)
        p = rng.dirichlet(np.repeat(alpha, K))
        cuts = (np.cumsum(p)[:-1] * len(idx_c)).astype(int)
        for k, chunk in enumerate(np.split(idx_c, cuts)):
            parts[k].extend(chunk)
    return [np.sort(np.array(p, dtype=int)) for p in parts]


def _label_skew(y, K, rng):
    """Pathological split: each client sees (almost) a single class.

    Binary labels, so clients alternate between the two classes; each class's
    samples are shared equally among the clients assigned to it. This is the
    hardest non-IID case — a client's local optimum is degenerate on its own.
    """
    classes = np.unique(y)
    parts = [[] for _ in range(K)]
    owners = {c: [k for k in range(K) if k % len(classes) == i]
              for i, c in enumerate(classes)}
    for c in classes:
        idx_c = np.where(y == c)[0]
        rng.shuffle(idx_c)
        ks = owners[c] or list(range(K))
        for k, chunk in zip(ks, np.array_split(idx_c, len(ks))):
            parts[k].extend(chunk)
    return [np.sort(np.array(p, dtype=int)) for p in parts]


def partition(y, n_clients, scheme="iid", seed=0, **kwargs):
    """Split sample indices across `n_clients`.

    scheme: "iid" | "dirichlet" (kwarg: alpha) | "label_skew"
    Returns a list of `n_clients` index arrays into y (possibly empty ones under
    a skewed scheme — every caller in this repo already handles len(ci) == 0).
    """
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    if scheme == "iid":
        return _iid(y, n_clients, rng)
    if scheme == "dirichlet":
        return _dirichlet(y, n_clients, rng, alpha=kwargs.get("alpha", 1.0))
    if scheme == "label_skew":
        return _label_skew(y, n_clients, rng)
    raise ValueError(f"Unknown scheme '{scheme}'. Choose: iid | dirichlet | label_skew")
