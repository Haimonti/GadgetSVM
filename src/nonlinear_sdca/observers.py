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
from src.nonlinear_sdca.kernel import kernel_scores, make_kernel


class KernelDataInitializer(Control):
    """Assign each node its shard, projection/kernel params, and global constants."""

    def __init__(self, pid, worker_data, lambda_reg, gossip_k,
                 seed, m, dist, gamma, tau, local_steps=None, step_scale=None, kernel=None):
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
        self.kernel = kernel

    def execute(self):
        n_global = sum(int(wd["n_local"]) for wd in self.worker_data)
        n_workers = len(self.worker_data)
        for i, wd in enumerate(self.worker_data):
            proto = Network.get(i).getProtocol(self.pid)
            proto.gossip_k = self.gossip_k
            proto.set_data(
                wd["X_csr"], wd["y"], wd["X_test"], wd["y_test"], self.lambda_reg,
                self.seed, self.m, self.dist, self.gamma, self.tau, kernel=self.kernel,
            )
            proto.configure_network(
                i, n_global, n_workers, self.local_steps, self.step_scale
            )
            logger.info(
                "init",
                f"Node {i} loaded — {wd['n_local']} local samples, "
                f"projected to m={self.m} ({self.dist})",
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
                 plateau_patience=20, plateau_rel_eps=1e-3, recovery=True):
        self.pid = pid
        self.recovery = recovery
        self.gap_threshold = gap_threshold
        self.eval_every = max(1, int(eval_every))
        self.total_cycles = total_cycles
        self.stop_on_threshold = stop_on_threshold
        self.stop_on_plateau = stop_on_plateau
        self.plateau_patience = max(1, int(plateau_patience))
        self.plateau_rel_eps = float(plateau_rel_eps)
        self._best_metric = float("inf")
        self._stall = 0
        self.resumes = 0
        self._X_all = None
        self._y_all = None
        self._offsets = None            # first row of each node's shard in X_all
        self._cache = {}                # (origin, version) -> {train|test|raw: partial score}
        self._used = set()

    def _stacked_training_set(self, protos):
        """Cache the concatenated PROJECTED shards — they never change in a run."""
        if self._X_all is None:
            self._X_all = np.vstack([p.Xhat for p in protos])
            self._y_all = np.concatenate([p.y for p in protos])
            self._offsets = np.concatenate([[0], np.cumsum([p.n for p in protos])[:-1]])
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
        self._used = set()
        for p in protos:
            blocks = self._blocks(p)
            # score f over the whole (projected) training set for this peer's model:
            # the sum of its blocks' partial scores (each computed once per version)
            f_all = np.zeros(X_all.shape[0])
            for blk in blocks:
                f_all += self._part(blk, "train", lambda b: kernel_scores(
                    b["X"], b["alpha"], X_all, p.kernel))
            f_all /= p.lam_n
            n_sv = sum(len(b["alpha"]) for b in blocks)
            # DSRR's tau||alpha||_1 dual term == hinge with the reduced margin (1 - tau)
            # in the primal (Yang et al. 2015, eq. 6), so the gap pairs
            #   P = mean max(0, (1-tau) - y f) + lambda/2 ||f||^2
            #   D = (1-tau) mean(a)            - lambda/2 ||f||^2
            margin = 1.0 - p.tau
            hinge_tau = float(np.mean(np.maximum(0.0, margin - y_all * f_all)))
            # the reported hinge_loss keeps margin 1, comparable across every model
            hinge = float(np.mean(np.maximum(0.0, 1.0 - y_all * f_all)))
            # ||f||_H^2 = (1/lam_n^2) * alpha^T K(SV,SV) alpha = (alpha . f_on_SV)/lam_n;
            # every SV is a training point, so f_on_SV is read off f_all at its row
            if n_sv:
                alpha_cat = np.concatenate([b["alpha"] for b in blocks])
                rows = np.concatenate([self._offsets[b["origin"]] + b["idx"] for b in blocks])
                norm_h2 = float(np.dot(alpha_cat, f_all[rows]) / p.lam_n)
                alpha_sum = float(np.dot(alpha_cat,                # sum of unsigned duals
                                         np.concatenate([b["y"] for b in blocks])))
            else:
                norm_h2 = 0.0
                alpha_sum = 0.0
            reg = (p.lambda_reg / 2.0) * norm_h2
            primal = hinge_tau + reg
            dual = margin * alpha_sum / n_global - reg
            gap = primal - dual

            scores_test = np.zeros(p.Xhat_test.shape[0])
            for blk in blocks:
                scores_test += self._part(blk, "test", lambda b: kernel_scores(
                    b["X"], b["alpha"], p.Xhat_test, p.kernel))
            preds = np.where(scores_test >= 0.0, 1.0, -1.0)

            if (self.stop_on_threshold and not p.stopped
                    and len(p.contributions) == len(protos)
                    and gap < self.gap_threshold):
                p.stopped = True
                p.stop_cycle = cycle
            elif self.stop_on_threshold and p.stopped and gap >= self.gap_threshold:
                # hysteresis, as in the linear evaluator: the dual is coupled across
                # nodes, so a latched node whose gap rose back above the threshold resumes
                p.stopped = False
                p.stop_cycle = None
                self.resumes += 1
            p.metrics.append({
                "round": cycle + 1,
                "primal": primal,
                "dual": dual,
                "duality_gap": gap,
                "hinge_loss": hinge,
                "accuracy": float(np.mean(preds == p.y_test)),
                **(self._recovered(p, protos, blocks) if self.recovery else {}),
                "known_origins": len(p.contributions),
                "n_support_vectors": int(n_sv),
                "stopped": bool(p.stopped),
                "wall_time": time.time() - p.start,
                "comm_bytes": p.comm_bytes,
            })
            gaps.append(gap)

        # keep only the blocks some node still holds (unchanged blocks are reused next time)
        self._cache = {k: v for k, v in self._cache.items() if k in self._used}
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
            f"stopped={n_stopped}/{len(protos)} (resumes {self.resumes})  "
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

    def _recovered(self, p, protos, blocks):
        """DSRR recovery step (Yang et al. 2015, before Thm. 5): keep the duals learned in
        the reduced space, but score with the ORIGINAL-space kernel,

            F_rec(x) = (1/(lambda n)) sum_j alpha_j K(x_j, x)        x_j, x raw.

        Decentralised by partial scores: raw data never leaves its node, so each origin
        o computes its own part  sum_{j in SV_o} alpha_j K(x_j, x)  on its raw rows,
        using the alphas node p currently knows for o (p's own live duals for itself,
        the versioned block for every other origin); the parts are summed. Per test
        query that costs one query broadcast (its raw nnz as index+value pairs) to the
        other K-1 nodes and K-1 returned float64 scores — reported as
        `recovery_bytes_per_query`, not added to the training `comm_bytes`.
        The projected-space `accuracy` is kept alongside, so the two can be compared.
        """
        X_test = p.X_test_raw
        score = np.zeros(X_test.shape[0])
        for blk in blocks:
            score += self._part(blk, "raw", lambda b: kernel_scores(
                protos[b["origin"]].X_raw[b["idx"]], b["alpha"], X_test, p.kernel))
        preds = np.where(score >= 0.0, 1.0, -1.0)
        K = len(protos)
        nnz_mean = X_test.nnz / max(X_test.shape[0], 1)
        return {"accuracy_recovered": float(np.mean(preds == p.y_test)),
                "recovery_bytes_per_query": float((K - 1) * (12 * nnz_mean + 8))}

    def _blocks(self, p):
        """The support-vector blocks that make up node p's model, as p holds them:
        its own live duals plus the versioned block it holds for every other origin.

        Each block carries a cache key (origin, version). A block's content is fixed by
        that pair, so its partial scores are computed once per evaluation and reused by
        every node holding the same version (measurement only; nothing goes back to the
        nodes). p's own block uses its published key when that equals its live duals,
        otherwise it is scored uncached.
        """
        blocks = []
        own = np.nonzero(p.alpha)[0]
        if len(own):
            pub = p.contributions.get(p.node_id)
            same = (pub is not None and np.array_equal(pub[4], own)
                    and np.array_equal(pub[2], p.alpha[own]))
            blocks.append({"key": (p.node_id, pub[0]) if same else None,
                           "origin": p.node_id, "idx": own, "X": p.Xhat[own],
                           "alpha": p.alpha[own], "y": p.y[own]})
        for origin, entry in p.contributions.items():
            if origin == p.node_id or len(entry[2]) == 0:
                continue
            blocks.append({"key": (origin, entry[0]), "origin": origin, "idx": entry[4],
                           "X": entry[1], "alpha": entry[2], "y": entry[3]})
        return blocks

    def _part(self, blk, what, compute):
        """Partial score of one block on `what` (train | test | raw), cached by key."""
        if blk["key"] is None:
            return compute(blk)
        entry = self._cache.setdefault(blk["key"], {})
        self._used.add(blk["key"])
        if what not in entry:
            entry[what] = compute(blk)
        return entry[what]
