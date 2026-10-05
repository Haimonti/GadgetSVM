"""The measuring Control for P2P-BDSVM, matched to SDCA's `GlobalEvaluator`.

BDSVM has no duality gap: it is an IRWLS fixed-point iteration, not a
primal-dual method, so there is no certificate to plot and none to stop on.
What the paper does define (Algorithm 2, step 9) is a stopping rule on the
relative movement of the iterate,

    ||beta^(n+1) - beta^(n)|| / ||beta^(n)||  <  eta,      eta = 5e-3

and that is what this Control latches on, one node at a time, exactly where
`GlobalEvaluator` latches on `gap < GAP_THRESHOLD`. The second half of SDCA's
condition carries over unchanged: a node may only stop once it has heard from
every origin, so a peer that is merely under-informed cannot mistake silence
for convergence.

Everything else is deliberately identical to `GlobalEvaluator` so the two
methods' curves are measurements of the same thing on the same schedule:

  * the same `eval_every` sampling (and the last cycle always measured),
  * hinge loss measured **globally** — over every shard in the network, not the
    node's own — so a node is not graded on the data it trained on,
  * accuracy on the shared held-out test set every node already has,
  * consensus error as the distance from the mean iterate,
  * the same metrics keys, with `duality_gap`/`primal`/`dual` left as NaN
    because BDSVM does not have them. A NaN is the honest entry; a zero would
    plot as a converged gap.
"""

import time

import numpy as np

from src.peersim_python.cdsim import CDState
from src.peersim_python.core import Control, Network
from src.peersim_python.logger import logger



class BDSVMEvaluator(Control):
    """Record every peer's global state; optionally stop on the paper's eta."""

    # The kernel block against the full training set is rows x P float64, and P
    # ranges from 100 (covtype) to 1000 (real-sim, rcv1), so a fixed row count
    # would be 10x bigger on the high-P datasets. Budget the block by entries
    # instead — ~20M entries, 160 MB — and derive the rows from P.
    CHUNK_ENTRIES = 20_000_000

    def _chunk_rows(self, P):
        return max(1_000, self.CHUNK_ENTRIES // max(P, 1))

    def __init__(self, pid, eta=5e-3, eval_every=1, total_cycles=None,
                 stop_on_threshold=False):
        self.pid = pid
        self.eta = float(eta)
        self.eval_every = max(1, int(eval_every))
        self.total_cycles = total_cycles
        self.stop_on_threshold = stop_on_threshold
        self._X_all = None
        self._y_all = None

    def _stacked_training_set(self, protos):
        """Cache the concatenated shards — they never change during a run."""
        if self._X_all is None:
            import scipy.sparse as sp
            self._X_all = sp.vstack([p.X for p in protos]).tocsr()
            self._y_all = np.concatenate([p.y for p in protos])
        return self._X_all, self._y_all

    def _global_hinge(self, p, X_all, y_all):
        """Mean hinge of node `p`'s model over every shard in the network.

        Built in row chunks: the score is k(x, p_j) . beta + b, so each chunk
        needs only its own slice of the kernel.
        """
        total, n = 0.0, X_all.shape[0]
        if p.kernel == "linear":
            scores = X_all @ (p.p.T @ p.beta[:p.P]) + p.beta[p.P]
            return float(np.mean(np.maximum(0.0, 1.0 - y_all * scores)))
        step = self._chunk_rows(p.P)
        for s in range(0, n, step):
            e = min(s + step, n)
            K = p._kernel(X_all[s:e], p.p)
            scores = K @ p.beta[:p.P] + p.beta[p.P]
            total += float(np.sum(np.maximum(0.0, 1.0 - y_all[s:e] * scores)))
        return total / n

    def execute(self):
        protos = [
            Network.get(i).getProtocol(self.pid) for i in range(Network.size())
        ]
        if not protos:
            return False

        cycle = CDState.getCycle()
        is_last = self.total_cycles is not None and cycle >= self.total_cycles - 1
        if cycle % self.eval_every and not is_last:
            return False

        X_all, y_all = self._stacked_training_set(protos)
        B = np.stack([p.beta for p in protos])
        mean_beta = B.mean(axis=0)

        changes, consensus_errors = [], []
        for p in protos:
            hinge = self._global_hinge(p, X_all, y_all)
            consensus_error = float(np.linalg.norm(p.beta - mean_beta))
            is_complete = len(p.contributions) == len(protos)
            # Per-node early stop (latching), the mirror of SDCA's rule: the
            # paper's eta on the iterate's relative movement, AND every origin
            # heard from. A stopped node stops training and gossiping, so its
            # communication cost freezes here.
            if (self.stop_on_threshold and not p.stopped
                    and is_complete and p.rel_change < self.eta):
                p.stopped = True
                p.stop_cycle = cycle
            p.metrics.append({
                "round": cycle + 1,
                "primal": float("nan"),
                "dual": float("nan"),
                "duality_gap": float("nan"),
                "rel_change": float(p.rel_change),
                "hinge_loss": float(hinge),
                "accuracy": p.accuracy(),
                "consensus_error": consensus_error,
                "known_origins": len(p.contributions),
                "stopped": bool(p.stopped),
                "wall_time": time.time() - p.start,
                "comm_bytes": p.comm_bytes,
            })
            changes.append(p.rel_change)
            consensus_errors.append(consensus_error)

        n_stopped = sum(1 for p in protos if p.stopped)
        finite = [c for c in changes if np.isfinite(c)]
        logger.info(
            "observer",
            f"cycle={cycle}  mean_rel_change="
            f"{(np.mean(finite) if finite else float('nan')):.3e}  "
            f"max_rel_change={(np.max(finite) if finite else float('nan')):.3e}  "
            f"max_consensus={np.max(consensus_errors):.3e}  "
            f"stopped={n_stopped}/{len(protos)}",
        )
        if not self.stop_on_threshold:
            return False
        # End the whole experiment once every node has stopped (converged).
        return all(p.stopped for p in protos)
