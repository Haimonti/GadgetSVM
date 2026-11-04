"""Fed-KSVM (Zhou & Wang, IEEE TNNLS 2024) — standalone, in the paper's own setup.

One central server and M devices, synchronous rounds (the paper simulates this
with MPI worker processes + one coordinator; here a plain loop). Devices talk
only to the server. Algorithm 2:

    server: broadcast w(t-1)
    device m (in parallel):
        w_bar_m = w(t-1) - u_m                     u_m = A_m alpha_m, its current part of w
        w*_m    = BlockBoosting(w_bar_m, u_m)      Algorithm 1, solves (8) == subproblem (5)
        Delta_m = w*_m - u_m                       = A_m Delta alpha_m
        upload Delta_m  (+ secure-aggregation mask, optional)
    server: w(t) = w(t-1) + beta_M * (1/M) sum_m Delta_m          (7)
    device: u_m  <- u_m + (beta_M / M) Delta_m                     keeps w = sum_m u_m

The paper writes the update as (7) on the server; the matching device-side step
(the same beta_M/M on its own dual block, which keeps the primal-dual identity
w = A alpha) is the CoCoA rule it builds on (its ref. [10]) and is stated here
explicitly. Model, data and regularisation are shared with our models: the
user's shards from `load_shards` (SHARDING=iid is the paper's i.i.d. split,
SHARDING=kmeans its non-i.i.d. protocol), C = 1/(LAMBDA * n_global), and the RBF
kernel with gamma = RBF_GAMMA approximated by D random Fourier features.

    devices = FedKSVMRun(cfg, worker_data).run(cycles)
"""

import time

import numpy as np

from src.peersim_python.logger import logger
from src.baselines.fed_ksvm.rff import RandomFourierFeatures
from src.baselines.fed_ksvm.block_boosting import BlockBoosting


class Device:
    """One edge device; exposes the per-node result interface the picker reads."""

    def __init__(self, m, F, y, F_test, y_test, solver):
        self.node_id = m
        self.F, self.y = F, np.asarray(y, dtype=np.float64)
        self.F_test, self.y_test = F_test, np.asarray(y_test, dtype=np.float64)
        self.solver = solver
        self.u = np.zeros(F.shape[1])
        self.w = np.zeros(F.shape[1])            # global model as last received
        self.metrics, self.comm_bytes = [], 0
        self.stopped, self.stop_cycle = False, None
        self.compute_time = 0.0
        self.start = time.time()

    def local_round(self, w_global):
        t0 = time.perf_counter()
        self.w = w_global
        w_star = self.solver.solve(w_global - self.u, self.u)
        self.compute_time += time.perf_counter() - t0
        return w_star - self.u

    def accuracy(self):
        return float(np.mean(np.where(self.F_test @ self.w >= 0.0, 1.0, -1.0) == self.y_test))


class FedKSVMRun:

    def __init__(self, cfg, worker_data):
        if cfg.get("KERNEL", "rbf") != "rbf":
            raise ValueError("Fed-KSVM uses random Fourier features: RBF kernel only")
        self.cfg = cfg
        self.worker_data = worker_data
        self.devices = []
        self.info = {}

    def build(self):
        cfg = self.cfg
        D = int(cfg["FEDKSVM_D"])
        d = self.worker_data[0]["X_csr"].shape[1]
        rff = RandomFourierFeatures(d, D, cfg["RBF_GAMMA"], cfg["SEED"])
        n_global = sum(w["n_local"] for w in self.worker_data)
        C = 1.0 / (cfg["LAMBDA"] * n_global)
        F_test = rff.transform(self.worker_data[0]["X_test"])
        L = int(cfg["FEDKSVM_L"])
        Q = int(cfg.get("FEDKSVM_Q") or L)
        for m, w in enumerate(self.worker_data):
            F = rff.transform(w["X_csr"])
            solver = BlockBoosting(F, w["y"], C, L, Q, cfg["FEDKSVM_H"],
                                   np.random.default_rng([cfg["SEED"], m, 8]))
            self.devices.append(Device(m, F, w["y"], F_test, w["y_test"], solver))
        self.F_all = np.vstack([dv.F for dv in self.devices])
        self.y_all = np.concatenate([dv.y for dv in self.devices])
        self.w = np.zeros(D)
        self.C = C
        self.info = {"fedksvm_D": D, "fedksvm_L": L, "fedksvm_Q": Q, "fedksvm_H": cfg["FEDKSVM_H"],
                     "fedksvm_beta": cfg["FEDKSVM_BETA"], "fedksvm_C": C,
                     "secure_aggregation": bool(cfg.get("FEDKSVM_SECURE_AGG", False)),
                     "setup": "1 server + M devices, synchronous rounds"}
        logger.info("fedksvm", f"M={len(self.devices)} D={D} L={L} Q={Q} H={cfg['FEDKSVM_H']} "
                               f"beta={cfg['FEDKSVM_BETA']} C={C:.4g}")
        return self

    def _masks(self, t, M, D):
        """Secure aggregation (Sec. IV-C, (12)-(13)): pairwise masks n_ij added by i,
        subtracted by j; the shared seed stands in for the Diffie-Hellman key."""
        masks = [np.zeros(D) for _ in range(M)]
        for i in range(M):
            for j in range(i + 1, M):
                n_ij = np.random.default_rng([self.cfg["SEED"], t, i, j]).normal(0, 1, D)
                masks[i] += n_ij
                masks[j] -= n_ij
        return masks

    def run(self, cycles=None):
        cfg = self.cfg
        if not self.devices:
            self.build()
        cycles = cycles if cycles is not None else cfg["ROUNDS"]
        self.budget = cycles
        M, D = len(self.devices), self.w.size
        beta = float(cfg["FEDKSVM_BETA"])
        every = max(1, int(cfg.get("EVAL_EVERY", 1)))
        best, stall = float("inf"), 0
        secure = cfg.get("FEDKSVM_SECURE_AGG", False)
        for t in range(cycles):
            deltas = [dv.local_round(self.w.copy()) for dv in self.devices]   # broadcast + local
            uploads = deltas
            if secure:
                masks = self._masks(t, M, D)
                uploads = [dl + mk for dl, mk in zip(deltas, masks)]
            self.w = self.w + beta * np.sum(uploads, axis=0) / M              # (7)
            for dv, dl in zip(self.devices, deltas):
                dv.u = dv.u + (beta / M) * dl
                dv.w = self.w
                dv.comm_bytes += 2 * D * 8                                    # down + up
            last = t == cycles - 1
            if t % every and not last:
                continue
            obj = self._evaluate(t)
            if obj < best * (1.0 - cfg.get("PLATEAU_REL_EPS", 1e-3)):
                best, stall = obj, 0
            else:
                stall += 1
            if cfg.get("STOP_ON_PLATEAU", False) and stall >= cfg.get("PLATEAU_PATIENCE", 20):
                for dv in self.devices:
                    dv.stopped, dv.stop_cycle = True, t
                    dv.metrics[-1]["stopped"] = True
                logger.info("fedksvm", f"plateau stop at round {t + 1}")
                break
        return self.devices

    def _evaluate(self, t):
        hinge = float(np.maximum(0.0, 1.0 - self.y_all * (self.F_all @ self.w)).mean())
        obj = hinge + 0.5 * self.cfg["LAMBDA"] * float(self.w @ self.w)
        consensus = float(np.linalg.norm(self.w - np.sum([dv.u for dv in self.devices], axis=0)))
        acc = self.devices[0].accuracy()
        for dv in self.devices:
            dv.metrics.append({
                "round": t + 1, "objective": obj, "primal": obj, "hinge_loss": hinge,
                "accuracy": acc, "comm_bytes": dv.comm_bytes,
                "wall_time": time.time() - dv.start, "compute_time": dv.compute_time,
                "w_minus_sum_u": consensus, "stopped": False,
            })
        logger.info("fedksvm", f"round={t + 1} objective={obj:.4e} acc={acc:.4f}")
        return obj

    def extra_json(self):
        return self.info
