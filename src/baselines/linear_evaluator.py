"""Shared evaluator for the synchronous linear baselines (COLA, CHOCO-SGD).

Records the same per-node row schema as the gossip tenants, with each node's own
model `p.w` scored on *all* global training data and on the shared test set:

    round, primal (= objective), hinge_loss, accuracy, consensus_error,
    comm_bytes, wall_time, compute_time, stopped  [+ duality_gap, + extras]

`dual_fn(protos) -> D` (optional) supplies a global dual value in our scaling,
D(alpha) = (1/n) sum_i a_i - lambda/2 ||X^T alpha / (lambda n)||^2, so each node
reports gap_k = P(w_k) - D. `extra_fn(p) -> dict` adds algorithm-specific
fields (COLA's consensus violation, local-solver gap, ...).

Stopping is run-level, because both algorithms are synchronous and a silent
node would change its neighbours' updates: every node stops together when all
node gaps are below GAP_THRESHOLD (only if a dual exists), or when the mean
convergence metric (gap, else objective) plateaus.
"""

import time

import numpy as np
import scipy.sparse as sp

from src.peersim_python.core import Control, Network
from src.peersim_python.cdsim.cd_state import CDState
from src.peersim_python.logger import logger


class SyncLinearEvaluator(Control):

    def __init__(self, pid, lam, gap_threshold, eval_every=1, total_cycles=None,
                 stop_on_threshold=False, stop_on_plateau=False, plateau_patience=20,
                 plateau_rel_eps=1e-3, dual_fn=None, extra_fn=None, tag="eval"):
        self.pid = pid
        self.lam = float(lam)
        self.gap_threshold = gap_threshold
        self.eval_every = max(1, int(eval_every))
        self.total_cycles = total_cycles
        self.stop_on_threshold = stop_on_threshold
        self.stop_on_plateau = stop_on_plateau
        self.plateau_patience = max(1, int(plateau_patience))
        self.plateau_rel_eps = float(plateau_rel_eps)
        self.dual_fn = dual_fn
        self.extra_fn = extra_fn
        self.tag = tag
        self._best = float("inf")
        self._stall = 0
        self._X_all = None
        self._y_all = None

    def execute(self):
        protos = [Network.get(i).getProtocol(self.pid) for i in range(Network.size())]
        cycle = CDState.getCycle()
        is_last = self.total_cycles is not None and cycle >= self.total_cycles - 1
        if cycle % self.eval_every and not is_last:
            return False
        if self._X_all is None:
            self._X_all = sp.vstack([p.X for p in protos]).tocsr()
            self._y_all = np.concatenate([p.y for p in protos])
        X_all, y_all = self._X_all, self._y_all

        W = np.stack([p.w for p in protos])
        hinges = np.maximum(0.0, 1.0 - y_all[:, None] * X_all.dot(W.T)).mean(axis=0)
        mean_w = W.mean(axis=0)
        dual = self.dual_fn(protos) if self.dual_fn else None

        conv = []
        for p, hinge in zip(protos, hinges):
            primal = float(hinge) + 0.5 * self.lam * float(np.dot(p.w, p.w))
            row = {
                "round": cycle + 1,
                "primal": primal,
                "objective": primal,
                "hinge_loss": float(hinge),
                "accuracy": p.accuracy(),
                "consensus_error": float(np.linalg.norm(p.w - mean_w)),
                "comm_bytes": p.comm_bytes,
                "wall_time": time.time() - p.start,
                "compute_time": p.compute_time,
                "stopped": bool(p.stopped),
            }
            if dual is not None:
                row["dual"] = dual
                row["duality_gap"] = primal - dual
            if self.extra_fn:
                row.update(self.extra_fn(p))
            p.metrics.append(row)
            conv.append(row["duality_gap"] if dual is not None else primal)

        mean_conv = float(np.mean(conv))
        if mean_conv < self._best * (1.0 - self.plateau_rel_eps):
            self._best, self._stall = mean_conv, 0
        else:
            self._stall += 1
        what = "mean_gap" if dual is not None else "mean_obj"
        logger.info(self.tag, f"cycle={cycle}  {what}={mean_conv:.3e}  "
                              f"max_consensus={max(r.metrics[-1]['consensus_error'] for r in protos):.3e}  "
                              f"stall={self._stall}/{self.plateau_patience}")

        stop = False
        if self.stop_on_threshold and dual is not None and max(conv) < self.gap_threshold:
            stop = True
        elif self.stop_on_plateau and self._stall >= self.plateau_patience:
            logger.info(self.tag, f"plateau stop at cycle {cycle} (best {what}={self._best:.3e})")
            stop = True
        if stop:
            for p in protos:
                p.stopped = True
                p.stop_cycle = cycle
                p.metrics[-1]["stopped"] = True
        return stop
