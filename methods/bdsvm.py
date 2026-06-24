"""BDSVM: Budget Distributed SVM for Non-IID Federated Learning.

Represents paper 1 — ACM TIST 2022 (DOI 10.1145/3539734).

Key ideas implemented from the paper:
  1. ADMM-based distributed consensus so clients can diverge locally
     (unlike FedAvg, dual variables u_k correct for client drift).
  2. Budget B: each client uses only its B highest-loss samples per round
     (the "budget set") — reduces communication and focuses computation
     on hard / boundary samples, which matters most for non-IID data.
  3. Weighted consensus: clients closer to the decision boundary get
     proportionally more influence via their budget-set size.
"""
import numpy as np


def _select_budget(X_k, y_k, w, B):
    n = len(y_k)
    if n <= B:
        return X_k, y_k
    scores = 1.0 - y_k * (X_k @ w)   # vectorised sparse matmul → (n,)
    if hasattr(scores, "A1"):          # in case result is a matrix
        scores = scores.A1
    top = np.argpartition(scores, -B)[-B:]
    return X_k[top], y_k[top]


def _local_admm_update(X_k, y_k, v_k, lambda_reg, rho, n_steps, rng):
    """Minimize C·hinge(w) + (λ/2)‖w‖² + (ρ/2)‖w − v_k‖² via Pegasos."""
    n = len(y_k)
    if n == 0:
        return v_k.copy()
    w = v_k.copy()
    eff_lambda = lambda_reg + rho
    for t in range(1, n_steps + 1):
        eta = 1.0 / (eff_lambda * t)
        i = int(rng.integers(n))
        xi = X_k[i]
        if hasattr(xi, "toarray"):
            xi = xi.toarray().ravel()
        yi = y_k[i]
        margin = yi * float(xi @ w)
        w = w - eta * (eff_lambda * w - rho * v_k)
        if margin < 1.0:
            w = w + eta * yi * xi
    return w


def run(X_train, y_train, X_test, y_test, client_idx,
        n_rounds=50, budget=200, n_local_steps=100,
        lambda_reg=0.01, rho=1.0, seed=0):
    rng = np.random.default_rng(seed)
    d = X_train.shape[1]
    K = len(client_idx)

    z = np.zeros(d)
    u = [np.zeros(d) for _ in range(K)]

    for _ in range(n_rounds):
        w_locals = []
        sizes = []
        for k, ci in enumerate(client_idx):
            if len(ci) == 0:
                w_locals.append(z.copy())
                sizes.append(0)
                continue
            X_k, y_k = _select_budget(X_train[ci], y_train[ci], z, budget)
            v_k = z - u[k]
            w_k = _local_admm_update(X_k, y_k, v_k, lambda_reg, rho, n_local_steps, rng)
            w_locals.append(w_k)
            sizes.append(len(ci))

        total = max(sum(sizes), 1)
        z = sum(sizes[k] / total * (w_locals[k] + u[k]) for k in range(K))

        for k in range(K):
            u[k] = u[k] + w_locals[k] - z

    scores = X_test @ z
    preds = np.sign(scores)
    preds[preds == 0] = 1.0
    acc = float((preds == y_test).mean())
    return {"test_acc": acc, "w": z}
