"""Controls for the GADGET baseline: data assignment + objective evaluation.

Mirrors `src/peersim_python/observers.py`, but for the Pegasos/weight-averaging
model. `PegasosDataInitializer` hands each node its shard and the run constants;
`PegasosEvaluator` is the measuring Control --- it computes the SVM **primal
objective** (hinge + (lambda/2)||w||^2) for every node over the whole training
set, plus accuracy and communication cost, and applies GADGET's own
objective-change stopping rule (a node stops once its objective stops changing
for `CONVERGENCE_COUNT` consecutive evaluations). There is no duality gap here;
that is the SDCA-specific quantity this baseline is being compared against.
"""

import time

import numpy as np
import scipy.sparse as sp

from src.peersim_python.cdsim import CDState
from src.peersim_python.core import Control, Network
from src.peersim_python.logger import logger


class PegasosDataInitializer(Control):
    """Assign each node its shard and the run's constants (GADGET baseline)."""

    def __init__(self, pid, worker_data, lambda_reg, gossip_k, local_steps=None):
        self.pid = pid
        self.worker_data = worker_data
        self.lambda_reg = lambda_reg
        self.gossip_k = gossip_k
        self.local_steps = local_steps

    def execute(self):
        n_global = sum(int(wd["n_local"]) for wd in self.worker_data)
        n_workers = len(self.worker_data)
        for i, wd in enumerate(self.worker_data):
            proto = Network.get(i).getProtocol(self.pid)
            proto.gossip_k = self.gossip_k
            proto.set_data(
                wd["X_csr"], wd["y"], wd["X_test"], wd["y_test"], self.lambda_reg,
            )
            proto.configure_network(i, n_global, n_workers, self.local_steps)
            logger.info(
                "init",
                f"Node {i} loaded — {wd['n_local']} local samples (GADGET/Pegasos)",
            )
        return False


class PegasosEvaluator(Control):
    """Record every node's primal objective/accuracy; stop on objective-change.

    Evaluation touches all N training samples for all K peers, so once the local
    solve is cheap this is the most expensive thing in a cycle. `eval_every`
    trades curve resolution for run time; the final cycle is always measured so
    a run never ends without a reading.
    """

    def __init__(self, pid, obj_threshold, convergence_count=10, eval_every=1,
                 total_cycles=None, stop_on_threshold=False,
                 stop_on_plateau=False, plateau_patience=20, plateau_rel_eps=1e-3):
        self.pid = pid
        self.obj_threshold = obj_threshold
        self.convergence_count = max(1, int(convergence_count))
        self.eval_every = max(1, int(eval_every))
        self.total_cycles = total_cycles
        self.stop_on_threshold = stop_on_threshold
        # Plateau guard: a second, run-level stop that fires when the global
        # mean objective stops improving, so a run never grinds to the cycle
        # cap once it has effectively stalled.
        self.stop_on_plateau = stop_on_plateau
        self.plateau_patience = max(1, int(plateau_patience))
        self.plateau_rel_eps = float(plateau_rel_eps)
        self._best_metric = float("inf")
        self._stall = 0
        self._X_all = None
        self._y_all = None

    def _stacked_training_set(self, protos):
        """Cache the concatenated shards — they never change during a run."""
        if self._X_all is None:
            self._X_all = sp.vstack([p.X for p in protos]).tocsr()
            self._y_all = np.concatenate([p.y for p in protos])
        return self._X_all, self._y_all

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
        W = np.stack([p.w for p in protos])                  # (K, d)
        # One sparse-dense product for the whole swarm, not K*K spmv.
        margins = 1.0 - y_all[:, None] * X_all.dot(W.T)      # (N, K)
        hinges = np.maximum(0.0, margins).mean(axis=0)       # (K,)

        objectives = []
        for p, hinge in zip(protos, hinges):
            reg = float((p.lambda_reg / 2.0) * np.dot(p.w, p.w))
            objective = float(hinge) + reg
            preds = np.where(p.X_test.dot(p.w) >= 0.0, 1.0, -1.0)
            # GADGET objective-change convergence: a node stops once its primal
            # objective stops moving for `convergence_count` consecutive
            # evaluations (and it has not already stopped).
            if p.prev_obj is not None and abs(objective - p.prev_obj) < self.obj_threshold:
                p.converge_count += 1
            else:
                p.converge_count = 0
            p.prev_obj = objective
            if (self.stop_on_threshold and not p.stopped
                    and p.converge_count >= self.convergence_count):
                p.stopped = True
                p.stop_cycle = cycle
            p.metrics.append({
                "round": cycle + 1,
                "objective": objective,
                "hinge_loss": float(hinge),
                "accuracy": float(np.mean(preds == p.y_test)),
                "converge_count": p.converge_count,
                "stopped": bool(p.stopped),
                "wall_time": time.time() - p.start,
                "comm_bytes": p.comm_bytes,
            })
            objectives.append(objective)

        n_stopped = sum(1 for p in protos if p.stopped)
        # Plateau tracking: an eval counts as progress only if the global mean
        # objective beats the best seen by at least `plateau_rel_eps`.
        mean_obj = float(np.mean(objectives))
        if mean_obj < self._best_metric * (1.0 - self.plateau_rel_eps):
            self._best_metric = mean_obj
            self._stall = 0
        else:
            self._stall += 1
        logger.info(
            "observer",
            f"cycle={cycle}  mean_obj={mean_obj:.4e}  "
            f"min_obj={np.min(objectives):.4e}  "
            f"stopped={n_stopped}/{len(protos)}  "
            f"stall={self._stall}/{self.plateau_patience}",
        )
        # Criterion 1 — convergence: every node's objective has settled.
        if self.stop_on_threshold and all(p.stopped for p in protos):
            return True
        # Criterion 2 — plateau: the run stopped improving before converging.
        if self.stop_on_plateau and self._stall >= self.plateau_patience:
            logger.info(
                "observer",
                f"plateau stop at cycle {cycle}: mean_obj flat for "
                f"{self._stall} evals (best={self._best_metric:.4e})",
            )
            return True
        return False
