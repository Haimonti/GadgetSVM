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


def _make_preimages(P, n_features, seed, scale=1.0):
    """The P randomly generated pre-image vectors p_j (Algorithm 2, step 2).

    Generated from a seed rather than drawn from the data, so every worker can
    reproduce them locally and only the seed needs to be transmitted. Uniform on
    [0, scale]^N suits the LIBSVM `.scale` datasets used here, whose features are
    already normalised to that range.
    """
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, scale, size=(P, n_features))


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
        n_rounds=50, P=100, C=1.0, lam=0.5, eta=5e-3,
        gamma=None, seed=0):
    """Train BDSVM federated over `client_idx` and return metrics.

    Args:
        client_idx: list of index arrays, one per worker
        n_rounds:   maximum IRWLS epochs (Algorithm 2's repeat loop)
        P:          budget — number of pre-image vectors
        C:          SVM penalty, used in the weighting rule (9)
        lam:        mixing weight in beta <- lam*beta_old + (1-lam)*beta_new
        eta:        stopping threshold on the relative change of beta
        gamma:      RBF width; defaults to 1/n_features
    """
    n_features = X_train.shape[1]
    gamma = (1.0 / n_features) if gamma is None else gamma

    # --- architecture, shared by construction (Algorithm 2, steps 2-3) ------
    p = _make_preimages(P, n_features, seed)
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
