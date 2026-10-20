"""FDR-SVM: Federated Distributionally Robust SVM via Wasserstein balls + ADMM.

Represents paper 2 — arXiv 2410.03877 (UAI 2025).

Key ideas implemented:
  1. Each client k minimises a Wasserstein-robust SVM over an ε_k-ball around
     its empirical distribution P_k. By Kantorovich duality the worst-case
     expectation of the hinge loss within a Wasserstein-1 ball of radius ε_k
     adds a penalty ε_k‖w‖* (dual norm). For L2-regularised linear SVMs the
     dual of the L2-norm is again the L2-norm, so the robust objective becomes:
         min_w  (λ + ε_k)·½‖w‖² + C·∑_i hinge(y_i wᵀx_i)
     i.e. clients with less data (higher ε_k) are more strongly regularised.
  2. ADMM global consensus, same structure as BDSVM but with per-client ε_k.
  3. ε_k is set inversely proportional to client dataset size n_k (larger data →
     smaller ambiguity ball, consistent with statistical theory).
"""
import numpy as np


def _local_robust_update(X_k, y_k, v_k, lambda_reg, eps_k, rho, n_steps, rng):
    """Wasserstein-robust local problem solved via Pegasos.

    Effective regularisation = λ + ε_k (see module docstring).
    """
    n = len(y_k)
    if n == 0:
        return v_k.copy()
    w = v_k.copy()
    eff_lambda = lambda_reg + eps_k + rho
    for t in range(1, n_steps + 1):
        eta = 1.0 / (eff_lambda * t)
        i = int(rng.integers(n))
        xi = X_k[i]
        if hasattr(xi, "toarray"):
            xi = xi.toarray().ravel()
        yi = y_k[i]
        margin = yi * float(xi @ w)
        w = w - eta * ((lambda_reg + eps_k) * w + rho * (w - v_k))
        if margin < 1.0:
            w = w + eta * yi * xi
    return w


def run(X_train, y_train, X_test, y_test, client_idx,
        n_rounds=50, n_local_steps=100,
        lambda_reg=0.01, rho=1.0, eps_scale=1.0, seed=0):
    """
    eps_scale: multiplier for Wasserstein ball radii ε_k = eps_scale / sqrt(n_k).
               Larger eps_scale → more robustness, more regularisation.
    """
    rng = np.random.default_rng(seed)
    d = X_train.shape[1]
    K = len(client_idx)

    sizes = [len(ci) for ci in client_idx]
    n_max = max(sizes) if sizes else 1
    eps = [eps_scale / max(np.sqrt(n), 1.0) for n in sizes]

    z = np.zeros(d)
    u = [np.zeros(d) for _ in range(K)]

    for _ in range(n_rounds):
        w_locals = []
        for k, ci in enumerate(client_idx):
            if len(ci) == 0:
                w_locals.append(z.copy())
                continue
            v_k = z - u[k]
            w_k = _local_robust_update(
                X_train[ci], y_train[ci], v_k,
                lambda_reg, eps[k], rho, n_local_steps, rng,
            )
            w_locals.append(w_k)

        total = max(sum(sizes), 1)
        z = sum(sizes[k] / total * (w_locals[k] + u[k]) for k in range(K))

        for k in range(K):
            u[k] = u[k] + w_locals[k] - z

    scores = X_test @ z
    preds = np.sign(scores)
    preds[preds == 0] = 1.0
    acc = float((preds == y_test).mean())
    return {"test_acc": acc, "w": z}
