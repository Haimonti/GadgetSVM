"""FDR-SVM paper reference; previous quadratic approximation is archived.

Eq. (9) and Algorithm 2: arXiv:2410.03877v3.
"""
import numpy as np


def run(X_train, y_train, X_test, y_test, client_idx,
        n_rounds=100, rho=.01, epsilon=None, kappa=1., tau=0.,
        dual_norm=1, seed=0):
    from paper_reference.fdr import train
    shards = [(X_train[ci], np.asarray(y_train)[ci]) for ci in client_idx]
    result = train(shards, rounds=n_rounds, rho=rho, epsilon=epsilon,
                   kappa=kappa, tau=tau, dual_norm=dual_norm)
    prediction = np.where(X_test @ result.weight >= 0, 1., -1.)
    return {"test_acc": float(np.mean(prediction == y_test)), "w": result.weight,
            "converged": None, "stop_reason": result.stop_reason,
            "communication_bytes": result.communication_bytes,
            "paper_version": "2410.03877v3", "history": result.history}
