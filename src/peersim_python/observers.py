"""Controls for the gossip-SDCA simulation: data assignment + global evaluation.

`DataInitializer` plays PeerSim's NodeInitializer role — it hands each node its
shard, plus the constants of the *global* objective (total sample count, worker
count), after the network is cloned.

`GlobalEvaluator` is the measuring Control. It is the only place a duality gap
is computed, and it computes it globally: primal and dual are read off the same
node's state, with the primal hinge measured over every shard in the network
rather than the node's own. A gap that pairs a primal containing other nodes'
contributions with a dual built from one node's local `alpha` is not a bound on
anything and must not be plotted as one, let alone used as a stopping rule.

Stopping is off by default (`stop_on_threshold=False`). A run that halts the
moment the gap first dips below a threshold shows only that the gap *touched*
that value; it says nothing about whether the algorithm stays there. Letting the
full cycle budget run is what distinguishes convergence from a lucky sample.
"""

import time

import numpy as np
import scipy.sparse as sp

from src.peersim_python.cdsim import CDState
from src.peersim_python.core import Control, Network
from src.peersim_python.logger import logger


class DataInitializer(Control):
    """Assign each node its shard and the global objective's constants."""

    def __init__(self, pid, worker_data, lambda_reg, gossip_k,
                 warm_start=False, local_steps=None, step_scale=None):
        self.pid = pid
        self.worker_data = worker_data
        self.lambda_reg = lambda_reg
        self.gossip_k = gossip_k
        self.warm_start = warm_start
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
            )
            proto.configure_network(
                i, n_global, n_workers, self.local_steps, self.step_scale
            )
            if self.warm_start:
                proto.warm_start()
            logger.info(
                "init",
                f"Node {i} loaded — {wd['n_local']} local samples"
                + ("  (+warm start)" if self.warm_start else ""),
            )
        return False


class GlobalEvaluator(Control):
    """Record every peer's global primal/dual state; optionally stop on the gap.

    Evaluation touches all N training samples for all K peers, so once the local
    solve is cheap this is the most expensive thing in a cycle. `eval_every`
    trades curve resolution for run time; the final cycle is always measured so
    a run never ends without a reading.
    """

    def __init__(self, pid, gap_threshold, eval_every=1, total_cycles=None,
                 stop_on_threshold=False):
        self.pid = pid
        self.gap_threshold = gap_threshold
        self.eval_every = max(1, int(eval_every))
        self.total_cycles = total_cycles
        self.stop_on_threshold = stop_on_threshold
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

        n_global = sum(p.n for p in protos)
        if n_global == 0:
            return False

        X_all, y_all = self._stacked_training_set(protos)
        W = np.stack([p.w for p in protos])                  # (K, d)
        # One sparse-dense product for the whole swarm, not K*K spmv.
        margins = 1.0 - y_all[:, None] * X_all.dot(W.T)      # (N, K)
        hinges = np.maximum(0.0, margins).mean(axis=0)       # (K,)

        mean_w = W.mean(axis=0)
        gaps, complete, consensus_errors = [], [], []
        for p, hinge in zip(protos, hinges):
            reg = float((p.lambda_reg / 2.0) * np.dot(p.w, p.w))
            primal = float(hinge) + reg
            # Only origins this peer has actually heard from contribute dual
            # mass, so a partly-informed peer reports a larger gap, never a
            # smaller one.
            alpha_sum = sum(entry[2] for entry in p.contributions.values())
            dual = float(alpha_sum / n_global - reg)
            gap = primal - dual
            preds = np.where(p.X_test.dot(p.w) >= 0.0, 1.0, -1.0)
            consensus_error = float(np.linalg.norm(p.w - mean_w))
            p.metrics.append({
                "round": cycle + 1,
                "primal": primal,
                "dual": dual,
                "duality_gap": gap,
                "hinge_loss": float(hinge),
                "accuracy": float(np.mean(preds == p.y_test)),
                "consensus_error": consensus_error,
                "known_origins": len(p.contributions),
                "wall_time": time.time() - p.start,
                "comm_bytes": p.comm_bytes,
            })
            gaps.append(gap)
            complete.append(len(p.contributions) == len(protos))
            consensus_errors.append(consensus_error)

        logger.info(
            "observer",
            f"cycle={cycle}  mean_gap={np.mean(gaps):.3e}  "
            f"max_gap={np.max(gaps):.3e}  "
            f"max_consensus={np.max(consensus_errors):.3e}",
        )
        if not self.stop_on_threshold:
            return False
        # Both conditions: a small gap on a peer that has not yet heard from
        # everyone would be a claim about a model it does not hold.
        return all(complete) and float(np.max(gaps)) < self.gap_threshold


# The old name, kept so existing drivers keep importing successfully.
ConvergenceObserver = GlobalEvaluator
