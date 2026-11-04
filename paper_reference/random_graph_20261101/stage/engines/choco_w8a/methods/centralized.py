"""Centralized SVM baseline using liblinear via sklearn (L2-regularized, dual=SDCA).

This is the performance ceiling: trained on all data at once without federation.

Regularisation alignment:
    Federated methods use lambda_reg (lambda in lambda/2 * ||w||^2 + hinge loss).
    sklearn LinearSVC uses C = 1 / lambda_reg for fair comparison.
    Call run(..., lambda_reg=0.01) to match federated baselines (C=100).
"""
import numpy as np
from sklearn.svm import LinearSVC


def run(X_train, y_train, X_test, y_test,
        lambda_reg=0.01, max_iter=5000, subsample=50_000, seed=0):
    """
    Args:
        lambda_reg: regularisation strength; converted to C = 1/lambda_reg
                    for sklearn, aligning with federated methods.
    """
    if subsample and len(y_train) > subsample:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(y_train), subsample, replace=False)
        X_train, y_train = X_train[idx], y_train[idx]

    C = 1.0 / lambda_reg
    clf = LinearSVC(C=C, loss="squared_hinge", dual="auto", max_iter=max_iter)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    acc = float((preds == y_test).mean())
    return {"test_acc": acc, "w": clf.coef_[0], "b": clf.intercept_[0]}
