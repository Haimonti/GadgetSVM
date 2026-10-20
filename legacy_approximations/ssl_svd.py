"""FedSSL-AMC: Federated Self-Supervised Encoder + Local SVM Classifier.

Represents paper 4 — arXiv 2510.04927 (2025).

Key ideas adapted to SDCA tabular / sparse datasets:
  Phase 1 — Federated Encoder (the "SSL" part):
    Each client computes a local truncated SVD (like local PCA, an unsupervised
    representation learning step analogous to SSL pre-training). The server
    aggregates the local right-singular-vector matrices via weighted average and
    re-orthogonalises (federated power iteration). This gives a shared dense
    encoder W ∈ R^{d_enc × d} without accessing raw data centrally.

  Phase 2 — Local SVM + FedAvg:
    Each client encodes its data Z_k = X_k @ W^T ∈ R^{n_k × d_enc}, trains a
    local linear SVM (sklearn LinearSVC) on Z_k, then the server averages the
    SVM weight vectors (FedAvg on the classifier, as in the original paper).

  Test-time:
    Encode X_test with the global W, apply the federated SVM weight.

Note on d_enc: defaults to 64 (practical for 47k-d rcv1 sparse data).
For covtype (d=54) d_enc is clamped to min(d_enc, d-1).
"""
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import LinearSVC


def _local_svd_components(X_k, d_enc):
    """Return right singular vectors (d_enc, d) from local data."""
    n_k = X_k.shape[0]
    n_comp = min(d_enc, n_k - 1, X_k.shape[1] - 1)
    if n_comp <= 0:
        return None, 0
    svd = TruncatedSVD(n_components=n_comp, random_state=0)
    svd.fit(X_k)
    explained = svd.explained_variance_ratio_.sum()
    return svd.components_, explained


def _federated_encoder(X_train, client_idx, d_enc):
    """Aggregate local SVDs into a global encoder W (d_enc, d)."""
    d = X_train.shape[1]
    V_agg = np.zeros((d_enc, d))
    total_weight = 0.0

    for ci in client_idx:
        if len(ci) < 2:
            continue
        V_k, _ = _local_svd_components(X_train[ci], d_enc)
        if V_k is None:
            continue
        n_k = len(ci)
        n_comp = V_k.shape[0]
        V_agg[:n_comp] += n_k * V_k
        total_weight += n_k

    if total_weight == 0:
        rng = np.random.default_rng(0)
        V_agg = rng.standard_normal((d_enc, d))
    else:
        V_agg /= total_weight

    # Re-orthogonalise so rows form an orthonormal basis
    Q, _ = np.linalg.qr(V_agg.T)       # Q is (d, d_enc)
    W = Q.T[:d_enc]                     # (d_enc, d)
    return W


def _local_svm_weight(Z_k, y_k, C=1.0):
    """Fit a local LinearSVC and return its weight vector."""
    if len(np.unique(y_k)) < 2:
        return np.zeros(Z_k.shape[1])
    clf = LinearSVC(C=C, dual=True, max_iter=2000, random_state=0)
    clf.fit(Z_k, y_k)
    return clf.coef_[0]


def run(X_train, y_train, X_test, y_test, client_idx,
        d_enc=64, svm_C=1.0, seed=0):
    d = X_train.shape[1]
    d_enc = min(d_enc, d - 1, X_train.shape[0] - 1)

    # Phase 1: federated encoder
    W = _federated_encoder(X_train, client_idx, d_enc)  # (d_enc, d)

    # Phase 2: encode + local SVM + FedAvg
    local_ws = []
    for ci in client_idx:
        if len(ci) == 0:
            continue
        Z_k = X_train[ci] @ W.T      # (n_k, d_enc)
        if hasattr(Z_k, "toarray"):
            Z_k = Z_k.toarray()
        y_k = y_train[ci]
        w_k = _local_svm_weight(Z_k, y_k, C=svm_C)
        local_ws.append((len(ci), w_k))

    if not local_ws:
        return {"test_acc": 0.5, "w": np.zeros(d_enc)}

    total = sum(n for n, _ in local_ws)
    w = sum(n * wk for n, wk in local_ws) / total

    Z_test = X_test @ W.T
    if hasattr(Z_test, "toarray"):
        Z_test = Z_test.toarray()
    scores = Z_test @ w
    preds = np.sign(scores)
    preds[preds == 0] = 1.0
    acc = float((preds == y_test).mean())
    return {"test_acc": acc, "w": w, "encoder": W}
