"""MoM-NDSVM (Forero, Cano & Giannakis, JMLR 2010, Algorithm 3) — standalone, in the
paper's own setup. No PeerSim: a plain synchronous loop over J nodes, exactly as
the paper simulates it, fed the user's datasets through `load_shards`.

    Require  Gamma (Sec. 4.1 random design), JC, eta, kernel agreed by all nodes
    init     random w~_j(0), b_j(0); alpha_j(0) = 0, beta_j(0) = 0; broadcast
    for t:   every node: (35) lambda_j(t+1) [E1-corrected], (36) w~_j(t+1), (37) b_j(t+1)
             every node broadcasts (w~_j(t+1), b_j(t+1)) to its neighbourhood
             every node: (38) alpha_j(t+1), (39) beta_j(t+1)
             readout g_j at t+1: lambda_j(t+1) with f~_j(t), h_j(t)   [E2]

All arithmetic is in `core.py` (validated, errata E1-E3 documented in
`__init__.py`); this file is only the paper's driver plus evaluation.

    nodes = ForeroRun(cfg, worker_data).run(cycles)     # model_picker's standalone contract
"""

import time

import numpy as np
from sklearn.svm import SVC

from src.peersim_python.logger import logger
from src.baselines.forero.core import LocalProblem, f_tilde, h_scalar, dual_ascent
from src.baselines.forero.qp import BoxQPSolver
from src.baselines.forero.anchors import check_graph, local_extremes, flood_extremes, draw_anchors
from src.baselines.forero.network import make_network, algebraic_connectivity, partition


def _dense(M):
    return M.toarray() if hasattr(M, "toarray") else np.asarray(M, dtype=np.float64)


class ForeroNode:
    """State of node j; exposes the per-node result interface the picker reads."""

    def __init__(self, j, X, y, X_test, y_test, neighbours, Gamma, K_GG, kernel, eta, JC,
                 lam_reg, qp_tol, qp_max_passes, rng):
        self.node_id = j
        self.X, self.y = X, np.asarray(y, dtype=np.float64)
        self.X_test, self.y_test = X_test, np.asarray(y_test, dtype=np.float64)
        self.neighbours = list(neighbours)
        self.Gamma, self.kernel, self.eta, self.lambda_reg = Gamma, kernel, float(eta), lam_reg
        self.model = LocalProblem(self.X, self.y, Gamma, K_GG, kernel, self.eta,
                                  len(self.neighbours), JC)
        self.qp = BoxQPSolver(self.model.Q, JC, tol=qp_tol, max_passes=qp_max_passes)
        L = Gamma.shape[0]
        self.alpha, self.beta = np.zeros(L), 0.0
        self.w_tilde, self.b = rng.standard_normal(L), float(rng.standard_normal())
        self._prev_w, self._prev_b = self.w_tilde.copy(), self.b
        self.lam = self.Qlam = self.a = np.zeros(self.model.N)
        self.c, self.b_read = np.zeros(L), self.b
        self._f = self._h = None
        self.inbox = {}
        self.metrics, self.comm_bytes, self.comm_bytes_p2p = [], 0, 0
        self.stopped, self.stop_cycle = False, None
        self.compute_time = self.eval_time = 0.0
        self.start = time.time()

    # lines 2-6
    def phase_a(self):
        t0 = time.perf_counter()
        w_nb = [self.inbox[i][0] for i in self.neighbours]
        b_nb = [self.inbox[i][1] for i in self.neighbours]
        f = f_tilde(self.alpha, self.w_tilde, w_nb, self.eta)
        h = h_scalar(self.beta, self.b, b_nb, self.eta)
        self.lam = self.qp.solve(self.model.lin_term(f, h)).copy()
        self.Qlam = self.qp.Qlam.copy()
        self.a, self.c, w_new, b_new = self.model.iterate(self.lam, f, h)
        self.b_read, self._f, self._h = b_new, f, h
        self.compute_time += time.perf_counter() - t0
        return w_new, b_new

    # lines 11-12, after the broadcast
    def phase_c(self):
        w_nb = [self.inbox[i][0] for i in self.neighbours]
        b_nb = [self.inbox[i][1] for i in self.neighbours]
        self.alpha, self.beta = dual_ascent(self.alpha, self.beta, self.w_tilde, self.b,
                                            w_nb, b_nb, self.eta)

    def decision_function(self, Z, K_ZG=None, chunk=4096):
        """(31) over the support vectors only, in row chunks."""
        sv = np.flatnonzero(self.a)
        X_sv, a_sv = self.X[sv], self.a[sv]
        out = np.empty(Z.shape[0])
        for s in range(0, Z.shape[0], chunk):
            Zc = Z[s:s + chunk]
            K_cG = K_ZG[s:s + chunk] if K_ZG is not None else self.kernel(Zc, self.Gamma)
            part = self.kernel(Zc, X_sv) @ a_sv if sv.size else 0.0
            out[s:s + chunk] = part + K_cG @ self.c + self.b_read
        return out

    def accuracy(self, K_testG=None):
        preds = np.where(self.decision_function(self.X_test, K_testG) >= 0.0, 1.0, -1.0)
        return float(np.mean(preds == self.y_test))


class ForeroRun:
    """Builds the paper's setup from our data and runs Algorithm 3."""

    def __init__(self, cfg, worker_data):
        self.cfg = cfg
        self.worker_data = worker_data
        self.nodes = []
        self.info = {}

    # ---- setup ----------------------------------------------------------------
    def _kernel(self):
        from src.nonlinear_sdca.kernel import make_kernel
        return make_kernel(self.cfg)

    def _data(self):
        """Stack our training shards, re-partition per the paper, densify."""
        cfg = self.cfg
        X_tr = np.vstack([_dense(w["X_csr"]) for w in self.worker_data])
        y_tr = np.concatenate([np.asarray(w["y"], dtype=np.float64) for w in self.worker_data])
        X_te = _dense(self.worker_data[0]["X_test"])
        y_te = np.asarray(self.worker_data[0]["y_test"], dtype=np.float64)
        if cfg.get("NDSVM_FEATURES", "raw") == "dsrr":
            from src.nonlinear_sdca.projection import project_shard
            X_tr, X_te, _ = project_shard(X_tr, X_te, cfg["SEED"], cfg["PROJECTION_DIM"],
                                          cfg["PROJECTION_DIST"])
        mode = cfg.get("FORERO_PARTITION", "balanced")
        J = cfg.get("FORERO_J") or len(self.worker_data)
        if mode == "given":
            if J != len(self.worker_data):
                raise ValueError("FORERO_PARTITION='given' needs FORERO_J == NUM_WORKERS")
            sizes = np.cumsum([0] + [w["n_local"] for w in self.worker_data])
            blocks = [np.arange(sizes[j], sizes[j + 1]) for j in range(J)]
        else:
            blocks = partition(X_tr, y_tr, J, mode, cfg["SEED"])
        return [(X_tr[b], y_tr[b]) for b in blocks], X_tr, y_tr, X_te, y_te

    def _references(self, shards, X_tr, y_tr, X_te, y_te, C):
        """The paper's benchmarks: centralised SVM and the average local-only SVM."""
        if not self.cfg.get("FORERO_REFERENCES", True):
            return {}
        kw = dict(kernel=self.cfg.get("KERNEL", "rbf"), C=C, tol=1e-6)
        if kw["kernel"] == "rbf":
            kw["gamma"] = self.cfg["RBF_GAMMA"]
        else:                                    # same polynomial as make_kernel
            kw.update(gamma=self.cfg.get("POLY_GAMMA") or "auto",
                      degree=self.cfg.get("POLY_DEGREE", 3), coef0=self.cfg.get("POLY_COEF0", 1.0))
        central = SVC(**kw).fit(X_tr, y_tr)
        local = [float(np.mean(SVC(**kw).fit(Xj, yj).predict(X_te) == y_te))
                 if len(np.unique(yj)) > 1 else None for Xj, yj in shards]
        ok = [a for a in local if a is not None]
        return {"central_svm_test_risk": 1.0 - float(np.mean(central.predict(X_te) == y_te)),
                "local_svm_test_risk": (1.0 - float(np.mean(ok))) if ok else None,
                "local_svm_single_class_nodes": local.count(None)}

    def build(self):
        cfg = self.cfg
        shards, X_tr, y_tr, X_te, y_te = self._data()
        J = len(shards)
        adj = make_network(cfg.get("FORERO_NETWORK", "random"), J, cfg["SEED"],
                           cfg.get("FORERO_RADIUS"))
        diameter = check_graph(adj)
        n_global = len(y_tr)
        C = 1.0 / (cfg["LAMBDA"] * n_global)          # same regularised problem as our models
        JC = J * C
        kernel = self._kernel()
        L, eta = int(cfg["NDSVM_L"]), float(cfg["NDSVM_ETA"])

        # Sec. 4.1 random design: max/min consensus for diameter rounds + shared seed
        ext = [local_extremes(X) for X, _ in shards]
        mins, maxs, sent = flood_extremes(adj, [e[0] for e in ext], [e[1] for e in ext],
                                          rounds=diameter)
        Gamma = draw_anchors(mins[0], maxs[0], L, cfg["SEED"])
        if not all(np.array_equal(draw_anchors(mins[j], maxs[j], L, cfg["SEED"]), Gamma)
                   for j in range(1, J)):
            raise RuntimeError("anchor draws differ across nodes")
        K_GG = kernel(Gamma, Gamma)
        p_dim = X_tr.shape[1]
        self.nodes = [ForeroNode(j, X, y, X_te, y_te, adj[j], Gamma, K_GG, kernel, eta, JC,
                                 cfg["LAMBDA"], cfg["NDSVM_QP_TOL"], cfg["NDSVM_QP_MAX_SWEEPS"],
                                 np.random.default_rng([cfg["SEED"], 1, j]))
                      for j, (X, y) in enumerate(shards)]
        for j, nd in enumerate(self.nodes):
            nd.comm_bytes_p2p += sent[j]
            nd.comm_bytes += diameter * 2 * p_dim * 8              # one broadcast per round
        self._broadcast([(nd.w_tilde.copy(), nd.b) for nd in self.nodes])
        self.adj, self.X_all, self.y_all = adj, X_tr, y_tr
        self.K_allG, self.K_testG = kernel(X_tr, Gamma), kernel(X_te, Gamma)
        self.info = {
            "forero_J": J, "forero_network": cfg.get("FORERO_NETWORK", "random"),
            "forero_partition": cfg.get("FORERO_PARTITION", "balanced"),
            "algebraic_connectivity": algebraic_connectivity(adj),
            "average_degree": float(np.mean([len(a) for a in adj])), "diameter": diameter,
            "ndsvm_C": C, "ndsvm_JC": JC, "comm_model": cfg.get("FORERO_COMM_MODEL", "broadcast"),
            **self._references(shards, X_tr, y_tr, X_te, y_te, C),
        }
        logger.info("forero", f"J={J} network={self.info['forero_network']} "
                              f"a.c.={self.info['algebraic_connectivity']:.4f} "
                              f"avg deg={self.info['average_degree']:.3f} L={L} eta={eta} JC={JC:.4g}")
        return self

    def _broadcast(self, msgs):
        """Line 8: every node's (w~, b) reaches its whole neighbourhood."""
        for j, (w, b) in enumerate(msgs):
            nd = self.nodes[j]
            size = (len(w) + 1) * 8
            nd.comm_bytes_p2p += len(nd.neighbours) * size
            nd.comm_bytes += size if self.cfg.get("FORERO_COMM_MODEL", "broadcast") == "broadcast" \
                else len(nd.neighbours) * size
            for i in nd.neighbours:
                self.nodes[i].inbox[j] = (w, b)

    # ---- the loop -----------------------------------------------------------------
    def run(self, cycles=None):
        cfg = self.cfg
        if not self.nodes:
            self.build()
        cycles = cycles if cycles is not None else cfg["ROUNDS"]
        self.budget = cycles
        every = max(1, int(cfg.get("EVAL_EVERY", 1)))
        best, stall = float("inf"), 0
        for t in range(cycles):
            pending = [nd.phase_a() for nd in self.nodes]            # lines 2-6
            self._broadcast(pending)                                  # line 8
            for nd, (w, b) in zip(self.nodes, pending):
                nd._prev_w, nd._prev_b = nd.w_tilde, nd.b
                nd.w_tilde, nd.b = w, b
            for nd in self.nodes:                                     # lines 11-12
                nd.phase_c()
            last = t == cycles - 1
            if t % every and not last:
                continue
            latched, objs = self._evaluate(t)
            mean_obj = float(np.mean(objs))
            if mean_obj < best * (1.0 - cfg.get("PLATEAU_REL_EPS", 1e-3)):
                best, stall = mean_obj, 0
            else:
                stall += 1
            stop = (cfg.get("STOP_ON_THRESHOLD", False) and all(latched)) or \
                   (cfg.get("STOP_ON_PLATEAU", False) and stall >= cfg.get("PLATEAU_PATIENCE", 20))
            if stop or last:
                for nd in self.nodes:
                    nd.stopped, nd.stop_cycle = stop, t
                    nd.metrics[-1]["stopped"] = stop
                if stop:
                    logger.info("forero", f"stop at iteration {t + 1}")
                    break
        return self.nodes

    def _evaluate(self, t):
        tol = self.cfg["NDSVM_RESIDUAL_TOL"]
        anchor_vals = np.stack([nd.w_tilde + nd.b for nd in self.nodes])
        anchor_mean = anchor_vals.mean(axis=0)
        coverage = float(self.K_allG.max(axis=1).mean())
        latched, objs = [], []
        for j, nd in enumerate(self.nodes):
            t0 = time.perf_counter()
            scores = nd.decision_function(self.X_all, self.K_allG)
            hinge = float(np.maximum(0.0, 1.0 - self.y_all * scores).mean())
            acc = nd.accuracy(self.K_testG)
            obj = 0.5 * nd.lambda_reg * nd.model.norm_sq(nd.lam, nd.Qlam, nd.a, nd.c) + hinge
            primal_res = max(max(float(np.linalg.norm(nd.w_tilde - nd.inbox[i][0])),
                                 abs(nd.b - nd.inbox[i][1])) for i in nd.neighbours)
            dual_res = nd.eta * float(np.sqrt(np.sum((nd.w_tilde - nd._prev_w) ** 2)
                                              + (nd.b - nd._prev_b) ** 2))
            is_latched = primal_res <= tol and dual_res <= tol
            nd.eval_time += time.perf_counter() - t0
            nd.metrics.append({
                "round": t + 1, "objective": obj, "hinge_loss": hinge, "accuracy": acc,
                "test_risk": 1.0 - acc,                     # the paper's R_emp(t), per node
                "comm_bytes": nd.comm_bytes, "comm_bytes_p2p": nd.comm_bytes_p2p,
                "wall_time": time.time() - nd.start,
                "primal_residual": primal_res, "dual_residual": dual_res,
                "anchor_disagreement": float(np.abs(anchor_vals[j] - anchor_mean).max()),
                "anchor_coverage": coverage,
                "n_support_vectors": int(np.count_nonzero(nd.lam)),
                "qp_sweeps": nd.qp.last_passes, "qp_converged": bool(nd.qp.converged),
                "local_gap": nd.model.local_gap(nd.lam, nd.Qlam, nd._f, nd._h),
                "compute_time": nd.compute_time, "eval_time": nd.eval_time,
                "latched": is_latched, "stopped": False,
            })
            latched.append(is_latched)
            objs.append(obj)
        logger.info("forero", f"iter={t + 1} mean_obj={np.mean(objs):.4e} "
                              f"mean_test_risk={np.mean([1 - nd.metrics[-1]['accuracy'] for nd in self.nodes]):.4f} "
                              f"latched={sum(latched)}/{len(latched)}")
        return latched, objs

    def extra_json(self):
        return self.info
