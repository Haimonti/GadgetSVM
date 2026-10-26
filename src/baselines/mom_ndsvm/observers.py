"""Controls for the MoM-NDSVM baseline: setup, the synchronous round barrier, and
evaluation.

* ``NDSVMDataInitializer`` — gives every node its shard and everything Algorithm 3
  "starts by having all nodes agree on": the anchors Gamma (Sec. 4.1 random design,
  via max/min consensus over the graph and a shared seed), JC, eta and the kernel.
  It then sends each node's random ``(w~_j(0), b_j(0))`` to its neighbours.
* ``SynchronousRoundBarrier`` — runs after every node's ``nextCycle``. It performs
  Algorithm 3's broadcast (line 8) and dual step (lines 11-12) for all nodes, so one
  engine cycle is exactly one iteration ``t``.
* ``NDSVMEvaluator`` — records the same per-node metric rows as the other tenants
  (``hinge_loss`` over all global training data with each node's own model,
  ``accuracy`` on the shared test set, cumulative ``comm_bytes``) plus
  ``objective = (lambda/2)||w_j||^2 + mean hinge``, which is GADGET's objective and
  SDCA's primal. It stops the whole network at once, because ADMM cannot drop a
  node without changing its neighbours' updates.
"""

import time

import numpy as np

from src.peersim_python.cdsim import CDState
from src.peersim_python.core import Control, Network
from src.peersim_python.logger import logger
from src.baselines.mom_ndsvm.anchors import (
    adjacency, check_graph, local_extremes, flood_extremes, draw_anchors,
)


def _dense(M):
    return M.toarray() if hasattr(M, "toarray") else np.asarray(M, dtype=np.float64)


class NDSVMDataInitializer(Control):
    """Shards, common quantities (Gamma, JC, eta, K) and the initial broadcast."""

    def __init__(self, pid, linkable_pid, worker_data, lambda_reg, kernel, L, eta,
                 seed, qp_tol, qp_max_passes, features="raw", projection=None):
        self.pid = pid
        self.linkable_pid = linkable_pid
        self.worker_data = worker_data
        self.lambda_reg = float(lambda_reg)
        self.kernel = kernel
        self.L = int(L)
        self.eta = float(eta)
        self.seed = int(seed)
        self.qp_tol = qp_tol
        self.qp_max_passes = qp_max_passes
        self.features = features
        self.projection = projection or {}

    def _features(self):
        """Dense training blocks and the shared dense test block."""
        if self.features == "raw":
            X_test = _dense(self.worker_data[0]["X_test"])
            return [_dense(wd["X_csr"]) for wd in self.worker_data], X_test
        if self.features == "dsrr":
            from src.nonlinear_sdca.projection import project_shard
            blocks, X_test = [], None
            for wd in self.worker_data:
                X_hat, X_hat_test, _ = project_shard(
                    wd["X_csr"], wd["X_test"], self.seed,
                    self.projection["m"], self.projection["dist"])
                blocks.append(X_hat)
                X_test = X_hat_test if X_test is None else X_test
            return blocks, X_test
        raise ValueError(f"Unknown NDSVM_FEATURES '{self.features}'. Choose: raw | dsrr")

    def execute(self):
        n = len(self.worker_data)
        protos = [Network.get(i).getProtocol(self.pid) for i in range(n)]
        links = [Network.get(i).getProtocol(self.linkable_pid) for i in range(n)]
        adj = adjacency(links)
        diameter = check_graph(adj)
        n_global = sum(int(wd["n_local"]) for wd in self.worker_data)
        C = 1.0 / (self.lambda_reg * n_global)        # same regularised problem as SDCA/GADGET
        JC = n * C

        blocks, X_test = self._features()
        for i, (p, wd) in enumerate(zip(protos, self.worker_data)):
            p.set_data(blocks[i], wd["y"], X_test, wd["y_test"], self.lambda_reg)

        # Sec. 4.1: max/min consensus (diameter rounds), then the shared-seed draw.
        ext = [local_extremes(b) for b in blocks]
        mins, maxs, sent = flood_extremes(adj, [e[0] for e in ext], [e[1] for e in ext],
                                          rounds=diameter)
        for j in range(1, n):
            if not (np.array_equal(mins[j], mins[0]) and np.array_equal(maxs[j], maxs[0])):
                raise RuntimeError("max/min consensus did not agree after diameter rounds")
        gammas = [draw_anchors(mins[j], maxs[j], self.L, self.seed) for j in range(n)]
        Gamma = gammas[0]
        if not all(np.array_equal(G, Gamma) for G in gammas[1:]):
            raise RuntimeError("anchor draws differ across nodes")
        K_GG = self.kernel(Gamma, Gamma)                                   # (30)

        for j, p in enumerate(protos):
            p.comm_bytes += sent[j]
            p.configure_network(
                j, n_global, n, adj[j], Gamma, K_GG, self.kernel, self.eta, JC,
                self.qp_tol, self.qp_max_passes,
                init_rng=np.random.default_rng([self.seed, 1, j]))
            logger.info("init", f"Node {j} ready — {p.model.N} samples, "
                                f"|B_j|={len(adj[j])} (MoM-NDSVM)")
        for p in protos:
            p.broadcast_initial(protos)
        logger.info("init", f"MoM-NDSVM: L={self.L}, eta={self.eta}, C={C:.4g}, JC={JC:.4g}, "
                            f"features={self.features}, diameter={diameter}, "
                            f"degrees={[len(a) for a in adj]}")
        return False


class SynchronousRoundBarrier(Control):
    """Algorithm 3 lines 7-14 for every node, after all nodes finished lines 2-6."""

    def __init__(self, pid):
        self.pid = pid

    def execute(self):
        protos = [Network.get(i).getProtocol(self.pid) for i in range(Network.size())]
        live = [p for p in protos if p.ready and not p.stopped]
        if not live:
            return False
        if any(p._pending is None for p in live):
            raise RuntimeError("a node skipped phase A; synchronous rounds are broken")
        for p in live:
            p.deliver(protos)            # line 8
        for p in live:
            p.commit()
        for p in live:
            p.finish_round()             # lines 11-12
        return False


class NDSVMEvaluator(Control):
    """Per-node metric rows every ``eval_every`` cycles (and on the last); global stop."""

    def __init__(self, pid, residual_tol, eval_every=1, total_cycles=None,
                 stop_on_threshold=False, stop_on_plateau=False,
                 plateau_patience=20, plateau_rel_eps=1e-3):
        self.pid = pid
        self.residual_tol = float(residual_tol)
        self.eval_every = max(1, int(eval_every))
        self.total_cycles = total_cycles
        self.stop_on_threshold = stop_on_threshold
        self.stop_on_plateau = stop_on_plateau
        self.plateau_patience = max(1, int(plateau_patience))
        self.plateau_rel_eps = float(plateau_rel_eps)
        self._best_metric = float("inf")
        self._stall = 0
        self._cache = None

    def _shared(self, protos):
        """Stacked training set and the kernels to the shared anchors (computed once)."""
        if self._cache is None:
            p0 = protos[0]
            X_all = np.vstack([p.X for p in protos])
            y_all = np.concatenate([p.y for p in protos])
            K_allG = p0.kernel(X_all, p0.Gamma)
            K_testG = p0.kernel(p0.X_test, p0.Gamma)
            coverage = float(K_allG.max(axis=1).mean())
            self._cache = (X_all, y_all, K_allG, K_testG, coverage)
        return self._cache

    def execute(self):
        protos = [Network.get(i).getProtocol(self.pid) for i in range(Network.size())]
        if not protos or not protos[0].ready:
            return False
        cycle = CDState.getCycle()
        is_last = self.total_cycles is not None and cycle >= self.total_cycles - 1
        if cycle % self.eval_every and not is_last:
            return False

        X_all, y_all, K_allG, K_testG, coverage = self._shared(protos)
        anchor_vals = np.stack([p.w_tilde + p.b for p in protos])   # g_j(chi_l) = w~_jl + b_j
        anchor_mean = anchor_vals.mean(axis=0)

        objectives, latched, residuals = [], [], []
        for j, p in enumerate(protos):
            t0 = time.perf_counter()
            scores = p.decision_function(X_all, K_allG)
            hinge = float(np.maximum(0.0, 1.0 - y_all * scores).mean())
            accuracy = p.accuracy(K_testG)
            norm_sq = p.model.norm_sq(p.lam, p.Qlam, p.a, p.c)
            objective = 0.5 * p.lambda_reg * norm_sq + hinge
            local_gap = p.model.local_gap(p.lam, p.Qlam, p._f, p._h)
            primal_res = max(max(float(np.linalg.norm(p.w_tilde - p.inbox[i][0])),
                                 abs(p.b - p.inbox[i][1])) for i in p.neighbours)
            dual_res = p.eta * float(np.sqrt(np.sum((p.w_tilde - p._prev_w) ** 2)
                                             + (p.b - p._prev_b) ** 2))
            is_latched = primal_res <= self.residual_tol and dual_res <= self.residual_tol
            p.eval_time += time.perf_counter() - t0
            p.metrics.append({
                "round": cycle + 1,
                "objective": objective,
                "hinge_loss": hinge,
                "accuracy": accuracy,
                "comm_bytes": p.comm_bytes,
                "wall_time": time.time() - p.start,
                "primal_residual": primal_res,
                "dual_residual": dual_res,
                "anchor_disagreement": float(np.abs(anchor_vals[j] - anchor_mean).max()),
                "anchor_coverage": coverage,
                "n_support_vectors": int(np.count_nonzero(p.lam)),
                "qp_sweeps": p.qp.last_passes,
                "qp_converged": bool(p.qp.converged),
                "local_gap": local_gap,
                "compute_time": p.compute_time,
                "eval_time": p.eval_time,
                "latched": is_latched,
                "stopped": bool(p.stopped),
            })
            objectives.append(objective)
            latched.append(is_latched)
            residuals.append(primal_res)

        mean_obj = float(np.mean(objectives))
        if mean_obj < self._best_metric * (1.0 - self.plateau_rel_eps):
            self._best_metric = mean_obj
            self._stall = 0
        else:
            self._stall += 1
        logger.info(
            "observer",
            f"cycle={cycle}  mean_obj={mean_obj:.4e}  max_primal_res={max(residuals):.2e}  "
            f"latched={sum(latched)}/{len(protos)}  stall={self._stall}/{self.plateau_patience}",
        )
        stop = False
        if self.stop_on_threshold and all(latched):
            logger.info("observer", f"residual stop at cycle {cycle}: every node within "
                                    f"{self.residual_tol:g} of its neighbours and its last iterate")
            stop = True
        elif self.stop_on_plateau and self._stall >= self.plateau_patience:
            logger.info("observer", f"plateau stop at cycle {cycle}: mean_obj flat for "
                                    f"{self._stall} evals (best={self._best_metric:.4e})")
            stop = True
        if stop:
            for p in protos:
                p.stopped = True
                p.stop_cycle = cycle
                p.metrics[-1]["stopped"] = True
        return stop
