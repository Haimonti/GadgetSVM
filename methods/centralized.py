"""Centralized SVM baseline using liblinear via sklearn (L2-regularized L2-loss, dual = SDCA).

This is the ceiling: trained on all data at once without federation.
"""
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MaxAbsScaler


def run(X_train, y_train, X_test, y_test,
        C=1.0, max_iter=5000, subsample=50_000, seed=0):
    if subsample and len(y_train) > subsample:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(y_train), subsample, replace=False)
        X_train, y_train = X_train[idx], y_train[idx]

    clf = LinearSVC(C=C, loss="squared_hinge", dual="auto", max_iter=max_iter)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    acc = float((preds == y_test).mean())
    return {"test_acc": acc, "w": clf.coef_[0], "b": clf.intercept_[0]}
