# Project Context: Federated SVM Benchmarking on SDCA Datasets

> Self-contained handoff document. Drop this into your project root as `CLAUDE.md` — Claude Code reads it automatically and uses it as the project's standing context.

## Goal

Reproduce / benchmark the experiments in 4 federated SVM papers using the standard SDCA benchmark datasets (astro-ph, CCAT, cov1). The end goal is a clean comparison of FL-SVM methods vs FedAvg under varying data heterogeneity.

## The 4 papers (all federated/distributed SVM)

| # | Paper | Year | Key idea |
|---|---|---|---|
| 1 | BDSVM (ACM TIST) — `https://dl.acm.org/doi/10.1145/3539734` | 2022 | Budget Distributed SVM for Non-IID FL |
| 2 | FDR-SVM — `https://arxiv.org/abs/2410.03877` | 2024 | Distributionally Robust SVM + Wasserstein balls + ADMM |
| 3 | Over-the-Air FL — `https://arxiv.org/abs/1812.11750` | 2018 | SVM via FedAvg with wireless aggregation |
| 4 | FedSSL-AMC — `https://arxiv.org/abs/2510.04927` | 2025 | FedAvg encoder + local SVM classifier |

**Key takeaway:** Naively applying FedAvg to SVM is suboptimal due to non-smooth hinge loss. Papers 1 & 2 show SVM-specific methods (especially ADMM-based) outperform MLP+FedAvg on non-IID data with lower communication. Paper 4 sidesteps the issue — federate the encoder, keep SVM local.

## The 3 benchmark datasets

These are **centralized** SDCA benchmarks from Shalev-Shwartz & Zhang (JMLR 2013). To use them in FL we partition them into K simulated clients (see `sdca_fl_data.py`).

| Name in SDCA paper | What it actually is | n / d | Source |
|---|---|---|---|
| **CCAT** | `rcv1.binary` (positive=CCAT+ECAT, negative=GCAT+MCAT) | 20,242 train + 677,399 test / 47,236 | LIBSVM ✅ |
| **cov1** | `covtype.binary.scale` | 581,012 / 54 | LIBSVM ✅ |
| **astro-ph** | Joachims's "Physics ArXiv" | ~62k / ~99k | NOT on LIBSVM ⚠️ |

**Download URLs:**
- `https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/rcv1_train.binary.bz2`
- `https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/rcv1_test.binary.bz2`
- `https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/covtype.libsvm.binary.scale.bz2`
- astro-ph: try `http://download.joachims.org/svm_perf/examples/example3.tar.gz` first (may 404 — Joachims's page has moved over the years). Fallbacks: OpenML "arxiv physics" subset, or contact the authors directly.

## Conceptual mismatch to be aware of

The SDCA paper's `O(n + 1/(λε))` complexity bound is for **centralized** SDCA. It does NOT directly apply to FL. The FL-equivalent is the communication-computation tradeoff bound from CoCoA+ (Smith et al.) — which is exactly the right baseline to compare paper #1 (BDSVM) and paper #2 (FDR-SVM) against.

The CCAT test set being huge (677k vs 20k train) is the "massive n regime" the SDCA paper specifically targets. In FL terms, this matters because per-round communication scales with n only at the client level — partitioning across K clients changes the effective regime.

## Existing code: `sdca_fl_data.py`

Already implemented and tested. Lives at the project root. Provides:

1. **`download_all(data_dir)`** — downloads the 3 datasets, skips files already present, gracefully handles astro-ph 404.
2. **`load_dataset(name, split, data_dir)`** — returns `(X, y)` where X is `scipy.sparse.csr_matrix` and y is `np.ndarray` of `±1`. Uses `sklearn.datasets.load_svmlight_file` (auto-decompresses bz2, parses LIBSVM format into sparse matrix — critical for CCAT which would be ~240 GB dense).
3. **Three FL partitioners**, all returning `List[np.ndarray[int64]]` (one index array per client):
   - `partition_iid(y, num_clients, seed)` — uniform random split. Sanity-check baseline.
   - `partition_label_skew(y, num_clients, classes_per_client=1, seed)` — half the clients get only +1, half get only -1. Worst-case non-IID, what BDSVM uses to break FedAvg.
   - `partition_dirichlet(y, num_clients, alpha, seed)` — for each class, draw client proportions from Dirichlet(α, ..., α) and split that class accordingly. Standard modern FL benchmark (FedAvg/FedProx/SCAFFOLD/FDR-SVM all use this).
4. **`partition(y, num_clients, scheme, **kwargs)`** — dispatcher.
5. **`summarize_partition(y, client_idx)`** — prints per-client n / n_pos / n_neg / pos_frac for sanity-checking.

### Known gotcha already fixed
When Dirichlet α is very small, some clients receive 0 samples. Empty `np.array([])` defaults to `float64` dtype, which crashes `y[ci]` indexing. Fix: explicit `np.array(ci, dtype=np.int64)` in the partitioner return.

### Full source

```python
"""
SDCA benchmark datasets (astro-ph, CCAT/rcv1, cov1/covtype) loader
+ Federated Learning partitioners (IID / label-skew / Dirichlet)
"""
import os
import sys
import bz2
import urllib.request
from pathlib import Path

import numpy as np
from sklearn.datasets import load_svmlight_file


LIBSVM_BASE = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary"

DATASETS = {
    "rcv1_train": (f"{LIBSVM_BASE}/rcv1_train.binary.bz2", "rcv1_train.binary.bz2"),
    "rcv1_test":  (f"{LIBSVM_BASE}/rcv1_test.binary.bz2",  "rcv1_test.binary.bz2"),
    "covtype":    (f"{LIBSVM_BASE}/covtype.libsvm.binary.scale.bz2",
                   "covtype.libsvm.binary.scale.bz2"),
    "astro_ph":   ("http://download.joachims.org/svm_perf/examples/example3.tar.gz",
                   "svmperf_example3.tar.gz"),
}


def _download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"[skip] {dest.name} already exists ({dest.stat().st_size/1e6:.1f} MB)")
        return
    print(f"[download] {url} -> {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    print(f"[done] {dest.stat().st_size/1e6:.1f} MB")


def download_all(data_dir: str = "./data", include_astro_ph: bool = True) -> None:
    data_dir = Path(data_dir)
    for name, (url, fname) in DATASETS.items():
        if name == "astro_ph" and not include_astro_ph:
            continue
        try:
            _download(url, data_dir / fname)
        except Exception as e:
            print(f"[warn] failed to download {name}: {e}")
            if name == "astro_ph":
                print("       astro-ph is the trickiest one — Joachims's URL may have moved.")
                print("       Try OpenML (search 'arxiv physics') or contact the authors.")


def load_dataset(name: str, split: str = "train", data_dir: str = "./data"):
    data_dir = Path(data_dir)
    if name == "rcv1":
        fname = f"rcv1_{split}.binary.bz2"
    elif name == "covtype":
        fname = "covtype.libsvm.binary.scale.bz2"
    elif name == "astro_ph":
        sub = "train.dat" if split == "train" else "test.dat"
        fname = f"svmperf_example3/example3/{sub}"
    else:
        raise ValueError(f"unknown dataset: {name}")

    path = data_dir / fname
    if not path.exists():
        raise FileNotFoundError(f"{path} — run download_all() first")

    X, y = load_svmlight_file(str(path))
    y = np.where(y > 0, 1.0, -1.0).astype(np.float64)
    return X, y


def partition_iid(y, num_clients, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    return np.array_split(idx, num_clients)


def partition_label_skew(y, num_clients, classes_per_client=1, seed=0):
    rng = np.random.default_rng(seed)
    if classes_per_client == 2:
        return partition_iid(y, num_clients, seed)
    pos_idx = np.where(y > 0)[0]
    neg_idx = np.where(y < 0)[0]
    rng.shuffle(pos_idx); rng.shuffle(neg_idx)
    half = num_clients // 2
    pos_chunks = np.array_split(pos_idx, half)
    neg_chunks = np.array_split(neg_idx, num_clients - half)
    return list(pos_chunks) + list(neg_chunks)


def partition_dirichlet(y, num_clients, alpha=0.3, seed=0):
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    client_idx = [[] for _ in range(num_clients)]
    for c in classes:
        idx_c = np.where(y == c)[0]
        rng.shuffle(idx_c)
        proportions = rng.dirichlet([alpha] * num_clients)
        cuts = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
        splits = np.split(idx_c, cuts)
        for i, s in enumerate(splits):
            client_idx[i].extend(s.tolist())
    return [np.array(ci, dtype=np.int64) for ci in client_idx]


def partition(y, num_clients, scheme="iid", **kwargs):
    if scheme == "iid":
        return partition_iid(y, num_clients, **kwargs)
    elif scheme == "label_skew":
        return partition_label_skew(y, num_clients, **kwargs)
    elif scheme == "dirichlet":
        return partition_dirichlet(y, num_clients, **kwargs)
    else:
        raise ValueError(f"unknown scheme: {scheme}")


def summarize_partition(y, client_idx):
    print(f"{'client':>8} {'n':>10} {'n_pos':>10} {'n_neg':>10} {'pos_frac':>10}")
    for i, ci in enumerate(client_idx):
        yi = y[ci]
        n_pos = int((yi > 0).sum()); n_neg = int((yi < 0).sum())
        n = len(ci); frac = n_pos / max(n, 1)
        print(f"{i:>8} {n:>10} {n_pos:>10} {n_neg:>10} {frac:>10.3f}")
```

## Recommended next steps (TODO)

The following are the natural extensions. Pick whichever the user asks for next:

1. **Clone the 4 papers' source code** into `./external/` (subdir per paper) and write thin adapter scripts so each accepts `(X_local, y_local)` from our partitioner. Most likely sources:
   - BDSVM: search for authors' GitHub (Wang/Liu, ACM TIST 2022)
   - FDR-SVM: arXiv 2410.03877 — check the paper for code URL
   - Over-the-Air FL: arXiv 1812.11750 — older, may need reimplementation
   - FedSSL-AMC: arXiv 2510.04927 — recent, likely has GitHub
2. **Centralized SDCA baseline** (`liblinear` `-s 3` is the standard one) on each dataset — needed as the "ceiling" reference. Without this we can't say whether an FL method is "close to optimal."
3. **FedAvg baseline** for SVM (just average local SGD-on-hinge-loss updates). Needed as the "floor" — papers 1, 2, 4 all claim to beat this.
4. **Experiment runner** (`run_benchmark.py`) that sweeps:
   - datasets ∈ {astro-ph, rcv1, covtype}
   - schemes ∈ {iid, dirichlet(α∈{1.0, 0.3, 0.1, 0.05}), label_skew}
   - num_clients ∈ {10, 50, 100}
   - methods ∈ {centralized SDCA, FedAvg, BDSVM, FDR-SVM, ...}
   - logs to CSV/parquet, plots accuracy-vs-α curves to reproduce the FDR-SVM paper's main figure.
5. **astro-ph fallback**: if Joachims's URL is dead, write a loader that pulls from OpenML or reconstructs the dataset from arXiv abstract data.

## Coding style preferences

- Python, functional style preferred over OOP unless state genuinely needs to be carried.
- **No comments in code** unless something is genuinely non-obvious — names should carry the meaning.
- Minimal, targeted edits over rewrites. If a function works and just needs one line changed, change one line.
- Prefer `numpy` / `scipy.sparse` / `sklearn` over PyTorch for the SVM-specific math (these datasets are sparse and small enough that GPU is overkill — except possibly for FedSSL-AMC's encoder).
- Use `np.random.default_rng(seed)` not legacy `np.random.seed`.
- Type hints are nice but not required.

## Environment

- Python 3.10+
- `numpy`, `scipy`, `scikit-learn` are required.
- `liblinear` (pip: `liblinear-official`) for the centralized SDCA baseline.
- `requests` or stdlib `urllib` for downloads — already using `urllib`.

# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:

- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:

- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:

- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:

- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:

```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.