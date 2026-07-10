"""Accuracy, runtime, and memory metric calculators."""


def compute_accuracy(y_true, y_pred) -> float:
    """Binary classification accuracy."""
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true)


def compute_duality_gap(primal: float, dual: float) -> float:
    return primal - dual



def print_summary(lightning_modules: list, logger=None) -> None:
    """Print final per-worker convergence table and optionally log each row."""
    print(f"\n{'Worker':>6}  {'Final Gap':>12}  {'Final Hinge':>12}  "
          f"{'Primal':>10}  {'Dual':>10}  {'Time (s)':>10}")
    print("─" * 66)
    for i, lm in enumerate(lightning_modules):
        if lm._metrics:
            r = lm._metrics[-1]
            if logger:
                logger.info(
                    "main",
                    f"Worker {i}  gap={r['duality_gap']:.6f}  hinge={r['hinge_loss']:.6f}  "
                    f"primal={r['primal']:.6f}  dual={r['dual']:.6f}  t={r['wall_time']:.1f}s"
                )
            print(f"{i:>6}  {r['duality_gap']:>12.6f}  {r['hinge_loss']:>12.6f}  "
                  f"{r['primal']:>10.6f}  {r['dual']:>10.6f}  {r['wall_time']:>10.1f}")

def compute_node_accuracy(w_np, X_test, y_test) -> float:
    """Binary accuracy: sign(X_test @ w_np) vs y_test."""
    import numpy as np
    preds = np.sign(X_test.dot(w_np))
    return float(np.mean(preds == y_test))
