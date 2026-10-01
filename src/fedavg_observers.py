"""FedAvg's measuring Control: global hinge, accuracy, consensus, and a stall stop.

FedAvg is primal-only, so like BDSVM it has no duality gap to plot or stop on.
It stops on the iterate stalling — the second of SDCA's two conditions, applied
alone:

    ||w - w_prev|| / ||w_prev||  <  stall_tol        (per node, latching)

measured over one eval window, exactly as SDCA does. There is no "heard from
every origin" gate: FedAvg keeps no origin table, and a node that has stalled
has, by construction, stopped changing under whatever it is receiving.

Metrics rows carry the same keys as the SDCA and BDSVM observers, with the
dual-side entries left as NaN, so plot_merged.py reads all three alike.
"""

import time

import numpy as np

from src.peersim_python.cdsim import CDState
from src.peersim_python.core import Control, Network
from src.peersim_python.logger import logger


class FedAvgEvaluator(Control):
    """Record every peer's global state; optionally stop on the stall rule."""

    def __init__(self, pid, stall_tol=1e-3, eval_every=1, total_cycles=None,
                 stop_on_threshold=False):
        self.pid = pid
        self.stall_tol = float(stall_tol)
        self.eval_every = max(1, int(eval_every))
        self.total_cycles = total_cycles
        self.stop_on_threshold = stop_on_threshold
        self._X_all = None
        self._y_all = None
        self._w_prev = {}          # node index -> w at the last evaluation

    def _stacked_training_set(self, protos):
        if self._X_all is None:
            import scipy.sparse as sp
            self._X_all = sp.vstack([p.X for p in protos]).tocsr()
            self._y_all = np.concatenate([p.y for p in protos])
        return self._X_all, self._y_all

    def execute(self):
        protos = [Network.get(i).getProtocol(self.pid)
                  for i in range(Network.size())]
        if not protos:
            return False

        cycle = CDState.getCycle()
        is_last = self.total_cycles is not None and cycle >= self.total_cycles - 1
        if cycle % self.eval_every and not is_last:
            return False

        X_all, y_all = self._stacked_training_set(protos)
        W = np.stack([p.w for p in protos])                 # (K, d)
        margins = 1.0 - y_all[:, None] * X_all.dot(W.T)     # (N, K)
        hinges = np.maximum(0.0, margins).mean(axis=0)
        mean_w = W.mean(axis=0)

        changes, consensus_errors = [], []
        for i, (p, hinge) in enumerate(zip(protos, hinges)):
            consensus_error = float(np.linalg.norm(p.w - mean_w))
            # Movement across one eval window, the same quantity SDCA's stall
            # rule compares. Compared to the window rather than to one cycle so
            # a converged-but-jittering iterate does not read as still moving.
            prev = self._w_prev.get(i)
            if prev is None:
                window_change = float("inf")
            else:
                denom = float(np.linalg.norm(prev))
                window_change = (float(np.linalg.norm(p.w - prev)) / denom
                                 if denom > 0 else float("inf"))
            self._w_prev[i] = p.w.copy()

            if (self.stop_on_threshold and not p.stopped
                    and window_change < self.stall_tol):
                p.stopped = True
                p.stop_cycle = cycle

            p.metrics.append({
                "round": cycle + 1,
                "primal": float(hinge) + float((p.lambda_reg / 2.0) * np.dot(p.w, p.w)),
                "dual": float("nan"),
                "duality_gap": float("nan"),
                "rel_change": float(window_change),
                "hinge_loss": float(hinge),
                "accuracy": p.accuracy(),
                "consensus_error": consensus_error,
                "known_origins": float("nan"),
                "stopped": bool(p.stopped),
                "wall_time": time.time() - p.start,
                "comm_bytes": p.comm_bytes,
            })
            changes.append(window_change)
            consensus_errors.append(consensus_error)

        n_stopped = sum(1 for p in protos if p.stopped)
        finite = [c for c in changes if np.isfinite(c)]
        logger.info(
            "observer",
            f"cycle={cycle}  mean_rel_change="
            f"{(np.mean(finite) if finite else float('nan')):.3e}  "
            f"max_consensus={np.max(consensus_errors):.3e}  "
            f"stopped={n_stopped}/{len(protos)}",
        )
        if not self.stop_on_threshold:
            return False
        return all(p.stopped for p in protos)
