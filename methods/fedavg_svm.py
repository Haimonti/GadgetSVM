"""FedAvg-SVM: Standard FedAvg applied to linear SVM via Pegasos subgradient descent.

Reference:
    Nair, D.G., Aswartha Narayana, C.V., Jaideep Reddy, K., Nair, J.J. (2022).
    "Exploring SVM for Federated Machine Learning Applications."
    In: Advances in Distributed Computing and Machine Learning,
    Lecture Notes in Networks and Systems, vol 427, Springer, Singapore.
    DOI: 10.1007/978-981-19-1018-0_25

Algorithm:
    Each client runs local Pegasos (subgradient SVM) for n_local_steps iterations.
    Server aggregates client weight vectors via weighted averaging (FedAvg).
    Repeated for n_rounds communication rounds.
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
    """
    Args:
        X_train, y_train: full training data (y in {-1, +1})
        X_test, y_test:   test data
        client_idx:       list of index arrays, one per client
        n_rounds:         number of FL communication rounds
        n_local_steps:    Pegasos steps per client per round
        lambda_reg:       SVM regularisation parameter (lambda)
        seed:             RNG seed for reproducibility
    Returns:
        dict with 'test_acc' and 'w' (final global model)
    """
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