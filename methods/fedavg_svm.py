"""FedAvg-SVM: FedAvg applied to linear SVM via Pegasos subgradient descent.

Represents paper 3 — Over-the-Air Federated Learning (arXiv 1812.11750).
Over-the-air wireless aggregation is modelled here as perfect FedAvg,
which is the algorithmic core of that paper's SVM experiments.
"""
import numpy as np


def _pegasos_steps(X_k, y_k, w, n_steps, lambda_reg, rng):
    n = len(y_k)
    if n == 0:
        return w
    w = w.copy()
    for t in range(1, n_steps + 1):
        eta = 1.0 / (lambda_reg * t)
        i = int(rng.integers(n))
        xi = X_k[i]
        if hasattr(xi, "toarray"):
            xi = xi.toarray().ravel()
        else:
            xi = np.asarray(xi).ravel()
        yi = y_k[i]
        margin = yi * float(xi @ w)
        w = w * (1.0 - eta * lambda_reg)
        if margin < 1.0:
            w = w + eta * yi * xi
    return w


def run(X_train, y_train, X_test, y_test, client_idx,
        n_rounds=50, n_local_steps=100, lambda_reg=0.01, seed=0):
    rng = np.random.default_rng(seed)
    d = X_train.shape[1]
    w = np.zeros(d)

    for _ in range(n_rounds):
        local_ws = []
        for ci in client_idx:
            if len(ci) == 0:
                continue
            w_k = _pegasos_steps(X_train[ci], y_train[ci], w,
                                  n_local_steps, lambda_reg, rng)
            local_ws.append((len(ci), w_k))

        if not local_ws:
            break
        total = sum(n for n, _ in local_ws)
        w = sum(n * wk for n, wk in local_ws) / total

    scores = X_test @ w
    preds = np.sign(scores)
    preds[preds == 0] = 1.0
    acc = float((preds == y_test).mean())
    return {"test_acc": acc, "w": w}
