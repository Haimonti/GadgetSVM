"""BDSVM — faithful reproduction of the ACM TIST 2022 paper.

    A. Navia-Vázquez, R. Díaz-Morales, M. Fernández-Díaz.
    "Budget Distributed Support Vector Machine for Non-ID Federated Learning
    Scenarios." ACM Trans. Intell. Syst. Technol. 13(6), Article 100, 2022.
    DOI 10.1145/3539734

This replaces an earlier implementation that had drifted from the paper: it was
a linear SVM trained by ADMM consensus, with "budget" taken to mean the B
highest-hinge-loss samples per round. Nothing in the paper corresponds to that,
and with the budget actually engaged it scored at chance (0.484 on covtype,
against a 0.516 majority-class baseline).

What the paper actually does
----------------------------
A *nonlinear kernel* SVM of controlled complexity, solved by IRWLS, in which
workers exchange kernel Gram matrices rather than model weights or support
vectors.

"Budget" means the model's architecture is fixed a priori at P basis elements:

    w_B = sum_{i=1..P} beta_i phi(p_i)                                    (4)
    f(x) = sum_{i=1..P} beta_i k(p_i, x) + b                              (6)

The P pre-images p_j are *randomly generated*, not selected from the data —
deliberately, since then no training pattern is part of the shipped model and
confidentiality is preserved. They are reproduced at every worker from a shared
seed, so only the seed travels.

The IRWLS loop (Algorithm 1) alternates:

    e_i = y_i - f(x_i)                                                    (8)
    a_i = 0            if e_i*y_i <  0
          2C/(e_i y_i) if e_i*y_i >= 0                                    (9)
    [K''^T D_a K'' + K''_p] beta'' = K''^T D_a y                          (12)
    beta''^(n+1) = lambda*beta''^(n) + (1-lambda)*beta''_new

with K'' = [K | 1], (K)_{i,j} = k(x_i, p_j), and K''_p = [[K_p, 0], [0, 0]].

Federating it (Algorithms 2 and 3) is exact rather than approximate: worker m
computes

    C_m = K''_m^T D_a^m K''_m                                            (15)
    d_m = K''_m^T D_a^m y^m                                              (16)

and the aggregator solves

    (sum_m C_m + K''_p) beta'' = sum_m d_m                               (17)

Because (17) is *formally equivalent* to the centralized (12), the federated
solution equals the centralized one regardless of how the data was split. That
identity is the whole point of the paper: BDSVM is intrinsically robust to
non-ID partitioning, with no drift term and no client-drift correction needed.

Choosing C
----------
C is the penalty in the weighting rule (9), a_i = 2C/(e_i y_i). Too small and
the data term is outweighed by the regulariser (1/2) beta^T K_p beta, so IRWLS
converges to an underfitted solution — and, awkwardly, to one *worse* than the
barely-trained iterate it passes through on epoch 1. That is what produced the
"accuracy peaks at epoch 1 then declines" behaviour: not divergence, but
convergence to a poor optimum.

Grid on covtype (10 nodes, 150 epochs, held-out validation slice), reporting
test accuracy at the converged point rather than the early peak:

    gamma \ C     0.001    0.01     0.1     1.0    10.0
    0.0152       0.5175  0.6062  0.6945  0.6925  0.7563
    0.0607       0.5175  0.6140  0.6872  0.6912  0.7562   <- median heuristic
    0.2427       0.5175  0.5175  0.5175  0.6537  0.6785

C=10 reaches 0.7562 against a centralized upper bound of 0.7577. gamma barely
matters below ~0.12, which is where the median heuristic lands anyway. lambda
does not help at all: every value in {0, 0.25, ..., 0.99} peaked at 0.7512 on
epoch 1 and settled at 0.68-0.72, and a per-epoch validation line search (what
the paper suggests) reached only 0.6975 — lambda sets how fast the iterate
moves, not where it converges.

`run_benchmark.py`'s existing METHOD_KWARGS entry for "bdsvm" —
{"P": 100, "C": 1.0, "lam": 0.5, "eta": 5e-3} — maps onto this algorithm's
symbols exactly (P pre-images, penalty C, mixing lambda, stopping eta). Those
kwargs raised TypeError against the previous implementation, which took
budget/n_local_steps/lambda_reg/rho, so every server BDSVM run was swallowed by
run_benchmark.py's broad except and logged as NaN. The call site was evidently
written against this paper; it now matches again, and works unchanged.
"""
import numpy as np


# ---------------------------------------------------------------------------
# Kernel and architecture
# ---------------------------------------------------------------------------

def _rbf(A, B, gamma):
    """Gaussian kernel matrix k(a_i, b_j) = exp(-gamma ||a_i - b_j||^2).

    Accepts sparse A (CSR) and dense B, which is the shape needed here: the
    training data is sparse, the P pre-images are dense.
    """
    if hasattr(A, "multiply"):                       # sparse
        a_sq = np.asarray(A.multiply(A).sum(axis=1)).ravel()
        cross = np.asarray(A.dot(B.T))
    else:
        a_sq = np.einsum("ij,ij->i", A, A)
        cross = A @ B.T
    b_sq = np.einsum("ij,ij->i", B, B)
    d2 = a_sq[:, None] + b_sq[None, :] - 2.0 * cross
    np.maximum(d2, 0.0, out=d2)
    return np.exp(-gamma * d2)


def _make_preimages(P, n_features, seed, scale=1.0, kind="uniform", nnz=64):
    """The P randomly generated pre-image vectors p_j (Algorithm 2, step 2).

    Generated from a seed rather than drawn from the data, so every worker can
    reproduce them locally and only the seed needs to be transmitted.

    The paper does not pin down the distribution, and the choice matters once
    the input space is high-dimensional:

      "uniform"  Uniform on [0, scale]^N. Suits the LIBSVM `.scale` datasets
                 whose features already live in that range (covtype, N=54).
      "unit"     Dense Gaussian, normalised to unit length.
      "sparse"   Sparse: `nnz` randomly placed Gaussian entries, normalised to
                 unit length. This is the one that works on sparse
                 high-dimensional data.

    Why the choice matters, measured on rcv1 (N=47236, unit-norm rows, ~73
    nonzeros each), reporting the spread of the resulting kernel values:

        uniform, dense   ||p|| ~ 125   every ||x-p||^2 collapses to ~||p||^2
        unit,    dense   ||p|| = 1     ||x-p||^2 in [1.957, 2.039] -- all data
                                       points are nearly equidistant from a
                                       dense random direction, which is just
                                       concentration of measure in 47k
                                       dimensions. Kernel std 1.7e-03 at best.
        sparse           ||p|| = 1     distances actually vary, because a sparse
                                       pre-image overlaps different documents
                                       differently.

    A dense pre-image in a sparse high-dimensional space is nearly orthogonal to
    every data point, so it cannot discriminate between them however gamma is
    set. Use "sparse" whenever the data is sparse.
    """
    rng = np.random.default_rng(seed)
    if kind == "unit":
        p = rng.standard_normal((P, n_features))
        return p / np.maximum(np.linalg.norm(p, axis=1, keepdims=True), 1e-12)
    if kind == "sparse":
        p = np.zeros((P, n_features))
        for i in range(P):
            cols = rng.choice(n_features, size=min(nnz, n_features), replace=False)
            p[i, cols] = rng.standard_normal(len(cols))
        return p / np.maximum(np.linalg.norm(p, axis=1, keepdims=True), 1e-12)
    return rng.uniform(0.0, scale, size=(P, n_features))


def _median_gamma(X, p, sample=2000, seed=0):
    """gamma = 1 / median(||x - p||^2), estimated from a subsample.

    The obvious default, 1/n_features, silently destroys the model on
    high-dimensional data. On rcv1 (N=47236, unit-norm rows) the squared
    distances sit around 2.0, so 1/N = 2.1e-05 puts every kernel value at
    exp(-4e-05) ~ 1: the kernel matrix becomes constant to within 2e-07 and the
    architecture carries no information at all. The whole grid's rcv1 BDSVM
    column was chance-level for exactly this reason. The median heuristic puts
    the exponent at O(1) by construction and does not care about N.
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = rng.choice(n, size=min(sample, n), replace=False)
    Xs = X[idx]
    a = (np.asarray(Xs.multiply(Xs).sum(axis=1)).ravel()
         if hasattr(Xs, "multiply") else np.einsum("ij,ij->i", Xs, Xs))
    b = np.einsum("ij,ij->i", p, p)
    d2 = a[:, None] + b[None, :] - 2.0 * np.asarray(Xs.dot(p.T))
    med = float(np.median(np.maximum(d2, 0.0)))
    return 1.0 / med if med > 1e-12 else 1.0


# ---------------------------------------------------------------------------
# The pieces each worker computes (Algorithm 3)
# ---------------------------------------------------------------------------

def _worker_contribution(K_m, y_m, beta, C):
    """Return (C_m, d_m) for one worker, given the current global weights.

    K_m is the worker's K'' = [K | 1], shape (n_m, P+1).
    """
    e = y_m - K_m @ beta                                   # step 5
    ey = e * y_m
    a = np.where(ey < 0.0, 0.0, 2.0 * C / np.where(ey == 0.0, 1e-12, ey))   # (9)
    Ka = K_m * a[:, None]                                  # D_a K''
    return K_m.T @ Ka, Ka.T @ y_m                          # (15), (16)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(X_train, y_train, X_test, y_test, client_idx,
        n_rounds=50, P=100, C=10.0, lam=0.5, eta=5e-3,
        gamma=None, preimage="uniform", seed=0):
    """Train BDSVM federated over `client_idx` and return metrics.

    Args:
        client_idx: list of index arrays, one per worker
        n_rounds:   maximum IRWLS epochs (Algorithm 2's repeat loop)
        P:          budget — number of pre-image vectors
        C:          SVM penalty, used in the weighting rule (9). Defaults to
                    10.0, chosen on a validation set — see the note below
        lam:        mixing weight in beta <- lam*beta_old + (1-lam)*beta_new
        eta:        stopping threshold on the relative change of beta
        gamma:      RBF width; defaults to the median heuristic — see
                    _median_gamma, and do not substitute 1/n_features
        preimage:   "uniform", "unit" or "sparse" — see _make_preimages
    """
    n_features = X_train.shape[1]

    # --- architecture, shared by construction (Algorithm 2, steps 2-3) ------
    p = _make_preimages(P, n_features, seed, kind=preimage)
    if gamma is None:
        gamma = _median_gamma(X_train, p)
    K_p = _rbf(p, p, gamma)
    Kpp = np.zeros((P + 1, P + 1))
    Kpp[:P, :P] = K_p                                      # K''_p

    # --- each worker's fixed kernel matrix (Algorithm 3, step 3) -----------
    workers = []
    for ci in client_idx:
        if len(ci) == 0:
            continue
        Km = _rbf(X_train[ci], p, gamma)
        workers.append((np.hstack([Km, np.ones((Km.shape[0], 1))]),
                        np.asarray(y_train[ci], dtype=np.float64)))
    if not workers:
        return {"test_acc": 0.5, "beta": np.zeros(P + 1)}

    beta = np.zeros(P + 1)                                 # beta'' = [beta; b]
    n_epochs = 0
    for _ in range(n_rounds):
        n_epochs += 1
        C_sum = np.zeros((P + 1, P + 1))
        d_sum = np.zeros(P + 1)
        for K_m, y_m in workers:                           # in parallel, really
            C_m, d_m = _worker_contribution(K_m, y_m, beta, C)
            C_sum += C_m
            d_sum += d_m

        # Aggregator: solve (sum_m C_m + K''_p) beta = sum_m d_m        (17)
        # cond(C_sum + K''_p) runs ~1e8 on covtype, and Eq (9)'s hard
        # a_i = 0 threshold turns tiny numerical differences into different
        # active sets across IRWLS epochs. A small ridge keeps the solve stable
        # without perturbing the solution meaningfully.
        A = C_sum + Kpp
        A[np.diag_indices_from(A)] += 1e-8 * np.trace(A) / A.shape[0]
        try:
            beta_new = np.linalg.solve(A, d_sum)
        except np.linalg.LinAlgError:
            beta_new = np.linalg.lstsq(A, d_sum, rcond=None)[0]

        beta_prev = beta
        beta = lam * beta_prev + (1.0 - lam) * beta_new     # Algorithm 2, step 8

        denom = np.linalg.norm(beta_prev)
        if denom > 0 and np.linalg.norm(beta - beta_prev) / denom < eta:
            break

    K_te = _rbf(X_test, p, gamma)
    scores = np.hstack([K_te, np.ones((K_te.shape[0], 1))]) @ beta
    preds = np.where(scores >= 0, 1.0, -1.0)
    acc = float(np.mean(preds == y_test))
    return {"test_acc": acc, "beta": beta, "preimages": p,
            "gamma": gamma, "epochs": n_epochs}
