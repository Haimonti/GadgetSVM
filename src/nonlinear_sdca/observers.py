"""Controls for the non-linear (kernel) tenant: data assignment + evaluation.

Mirrors `src/peersim_python/observers.py`, kernelized. `KernelDataInitializer`
hands each node its shard plus the projection/kernel parameters (so the node can
build its shared DSRR projection and local Gram) and the global-objective
constants. `KernelEvaluator` is the measuring Control: it computes, for every
node, the kernel-space duality gap of the model represented by that node's known
support vectors (own + accumulated), measured over the whole training set — the
exact kernel analogue of the linear `GlobalEvaluator`. It carries the same two
early stops: convergence (every node's gap below threshold) and plateau (the
global mean gap stops improving).
"""

import time

import numpy as np

from src.peersim_python.cdsim import CDState
from src.peersim_python.core import Control, Network
from src.peersim_python.logger import logger
from src.nonlinear_sdca.kernel import rbf_gram, kernel_scores


class KernelDataInitializer(Control):
    """Assign each node its shard, projection/kernel params, and global constants."""

    def __init__(self, pid, worker_data, lambda_reg, gossip_k,
                 seed, m, dist, gamma, tau, local_steps=None, step_scale=None):
        self.pid = pid
        self.worker_data = worker_data
        self.lambda_reg = lambda_reg
        self.gossip_k = gossip_k
        self.seed = seed
        self.m = m
        self.dist = dist
        self.gamma = gamma
        self.tau = tau
        self.local_steps = local_steps
        self.step_scale = step_scale

    def execute(self):
        n_global = sum(int(wd["n_local"]) for wd in self.worker_data)
        n_workers = len(self.worker_data)
        for i, wd in enumerate(self.worker_data):
            proto = Network.get(i).getProtocol(self.pid)
            proto.gossip_k = self.gossip_k
            proto.set_data(
                wd["X_csr"], wd["y"], wd["X_test"], wd["y_test"], self.lambda_reg,
                self.seed, self.m, self.dist, self.gamma, self.tau,
            )
            proto.configure_network(
                i, n_global, n_workers, self.local_steps, self.step_scale
            )
            logger.info(
                "init",
                f"Node {i} loaded — {wd['n_local']} local samples, "
                f"projected to m={self.m} ({self.dist}), RBF gamma={self.gamma}",
            )
        return False


class KernelEvaluator(Control):
    """Record every node's kernel-space primal/dual gap; optionally stop early.

    For each peer the model is the union of its known support vectors (own +
    accumulated). The primal hinge is measured over EVERY training sample across
    all shards, so a partly-informed peer reports a larger gap, never a smaller
    one — the same honesty property the linear evaluator has.
    """

    def __init__(self, pid, gap_threshold, eval_every=1, total_cycles=None,
                 stop_on_threshold=False, stop_on_plateau=False,
                 plateau_patience=20, plateau_rel_eps=1e-3):
        self.pid = pid
        self.gap_threshold = gap_threshold
        self.eval_every = max(1, int(eval_every))
        self.total_cycles = total_cycles
        self.stop_on_threshold = stop_on_threshold
        self.stop_on_plateau = stop_on_plateau
        self.plateau_patience = max(1, int(plateau_patience))
        self.plateau_rel_eps = float(plateau_rel_eps)
        self._best_metric = float("inf")
        self._stall = 0
        self._X_all = None
        self._y_all = None

    def _stacked_training_set(self, protos):
        """Cache the concatenated PROJECTED shards — they never change in a run."""
        if self._X_all is None:
            self._X_all = np.vstack([p.Xhat for p in protos])
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

        n_global = sum(p.n for p in protos)
        if n_global == 0:
            return False

        X_all, y_all = self._stacked_training_set(protos)
        gaps = []
        for p in protos:
            Xsv, alpha_sv, y_sv = p.all_support_vectors()
            # score f over the whole (projected) training set for this peer's model
            f_all = kernel_scores(Xsv, alpha_sv, X_all, p.gamma) / p.lam_n
            hinge = float(np.mean(np.maximum(0.0, 1.0 - y_all * f_all)))
            # ||f||_H^2 = (1/lam_n^2) * alpha^T K(SV,SV) alpha = (alpha . f_on_SV)/lam_n
            if len(alpha_sv):
                f_on_sv = kernel_scores(Xsv, alpha_sv, Xsv, p.gamma) / p.lam_n
                norm_h2 = float(np.dot(alpha_sv, f_on_sv) / p.lam_n)
                alpha_sum = float(np.dot(alpha_sv, y_sv))   # sum of unsigned duals
            else:
                norm_h2 = 0.0
                alpha_sum = 0.0
            reg = (p.lambda_reg / 2.0) * norm_h2
            primal = hinge + reg
            dual = alpha_sum / n_global - reg
            gap = primal - dual

            scores_test = kernel_scores(Xsv, alpha_sv, p.Xhat_test, p.gamma)
            preds = np.where(scores_test >= 0.0, 1.0, -1.0)

            if (self.stop_on_threshold and not p.stopped
                    and len(p.contributions) == len(protos)
                    and gap < self.gap_threshold):
                p.stopped = True
                p.stop_cycle = cycle
            p.metrics.append({
                "round": cycle + 1,
                "primal": primal,
                "dual": dual,
                "duality_gap": gap,
                "hinge_loss": hinge,
                "accuracy": float(np.mean(preds == p.y_test)),
                "known_origins": len(p.contributions),
                "n_support_vectors": int(len(alpha_sv)),
                "stopped": bool(p.stopped),
                "wall_time": time.time() - p.start,
                "comm_bytes": p.comm_bytes,
            })
            gaps.append(gap)

        n_stopped = sum(1 for p in protos if p.stopped)
        mean_gap = float(np.mean(gaps))
        if mean_gap < self._best_metric * (1.0 - self.plateau_rel_eps):
            self._best_metric = mean_gap
            self._stall = 0
        else:
            self._stall += 1
        logger.info(
            "observer",
            f"cycle={cycle}  mean_gap={mean_gap:.3e}  "
            f"max_gap={np.max(gaps):.3e}  "
            f"stopped={n_stopped}/{len(protos)}  "
            f"stall={self._stall}/{self.plateau_patience}",
        )
        if self.stop_on_threshold and all(p.stopped for p in protos):
            return True
        if self.stop_on_plateau and self._stall >= self.plateau_patience:
            logger.info(
                "observer",
                f"plateau stop at cycle {cycle}: mean_gap flat for "
                f"{self._stall} evals (best={self._best_metric:.3e})",
            )
            return True
        return False
