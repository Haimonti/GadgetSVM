"""KernelSDCAProtocol — RBF kernel SDCA over DSRR-projected support vectors.

A tenant of the generic `GossipProtocol`, the kernel counterpart of
`src/peersim_python/sdca_protocol.py`. The engine, topology and per-cycle loop are
inherited unchanged; only the *learner* differs.

Per node:
  - `set_data` projects the local shard with the shared DSRR matrix (x_hat = A x)
    and precomputes the local Gram K_local for fast coordinate sweeps.
  - `local_update` runs a kernel SDCA sweep that updates the signed dual `alpha`
    and an incrementally-maintained score cache, then publishes this node's own
    support-vector block (only the points with alpha != 0), versioned.
  - the gossip layer exchanges per-origin support-vector blocks; `set_state`
    folds the accumulated blocks back into the score cache (the kernel analogue of
    rebuilding `w`).

Dual convention mirrors the linear tenant: ``alpha[i]`` is the SIGNED dual
``a_i = y_i * unsigned_i`` with ``unsigned_i in [0, 1]``, and ``lam_n = lambda *
n_global``. The score of a point is ``f = (1/lam_n) * sum_j alpha_j K(x_hat_j, .)``
summed over every known support vector (local + accumulated), so ``margin = y*f``.
"""

import time

import numpy as np

from src.peersim_python.core.common_state import CommonState
from src.peersim_python.gossip_protocol import GossipProtocol
from src.nonlinear_sdca.aggregator import SupportVectorAccumulator
from src.nonlinear_sdca.kernel import rbf_gram, kernel_scores


class KernelSDCAProtocol(GossipProtocol):
    """Kernel (RBF) SDCA learner; gossips projected support vectors, accumulates."""

    def __init__(self):
        super().__init__(gossip_k=1, aggregator=SupportVectorAccumulator())
        self.data_ready = False
        self.node_id = None
        self.n_global = None
        self.n_workers = 1
        self.local_steps = 1
        self.step_scale = 1.0
        self.version = 0
        self.contributions = {}     # origin -> (version, X_hat_sv, alpha_sv, y_sv)
        self.metrics = []
        self.start = None
        # kernel / projection params (set in set_data)
        self.gamma = 0.1
        self.m = None
        self.dist = "sparse"
        self.tau = 0.0
        # versioned delta-send bookkeeping: per-neighbour last-seen version per
        # origin, so a block is transmitted to a neighbour at most once per version
        # instead of re-sending the whole accumulated table every cycle.
        self._sent = {}
        self._last_sweep_changed = True

    def clone(self):
        """Fresh learner carrying only the fanout (the engine clones the prototype
        before any data is assigned, exactly as SDCA/Pegasos do)."""
        twin = KernelSDCAProtocol()
        twin.gossip_k = self.gossip_k
        return twin

    # ---- setup (called once per node by KernelDataInitializer) ---------------
    def set_data(self, X_csr, y, X_test, y_test, lambda_reg,
                 seed, m, dist, gamma, tau):
        """Project the shard with the shared DSRR matrix and cache the local Gram."""
        from src.nonlinear_sdca.projection import project_shard

        self.y = np.asarray(y, dtype=np.float64)
        self.y_test = np.asarray(y_test, dtype=np.float64)
        self.lambda_reg = float(lambda_reg)
        self.gamma = float(gamma)
        self.m = int(m)
        self.dist = dist
        self.tau = float(tau)

        self.Xhat, self.Xhat_test, _ = project_shard(
            X_csr, X_test, seed, self.m, self.dist
        )
        self.n = self.Xhat.shape[0]
        self.alpha = np.zeros(self.n, dtype=np.float64)
        # local Gram K_local[i, j] = K(x_hat_i, x_hat_j); diag is 1 for RBF.
        self.K_local = rbf_gram(self.Xhat, self.Xhat, self.gamma)
        # score cache split into the local-dual part (maintained incrementally
        # during a sweep) and the external part (recomputed on every merge).
        self.f_local = np.zeros(self.n, dtype=np.float64)
        self.f_ext = np.zeros(self.n, dtype=np.float64)
        self.start = time.time()
        self.data_ready = True

    def configure_network(self, node_id, n_global, n_workers,
                          local_steps=None, step_scale=None):
        self.node_id = int(node_id)
        self.n_global = int(n_global)
        self.n_workers = int(n_workers)
        self.local_steps = int(local_steps) if local_steps else self.n
        scale = 1.0 / self.n_workers if step_scale is None else float(step_scale)
        if not 0.0 < scale <= 1.0:
            raise ValueError("step_scale must be in (0, 1]")
        self.step_scale = scale
        self.lam_n = self.lambda_reg * self.n_global
        self._publish_contribution(bump=False)   # seed the version-0 (empty) block

    # ---- GossipProtocol hooks ------------------------------------------------
    def ready(self):
        return self.data_ready and self.node_id is not None

    def current_state(self):
        return self.contributions

    def set_state(self, merged):
        self.contributions = merged
        self._recompute_external_scores()

    def outgoing_payload(self):
        return dict(self.contributions)   # relay every known origin's block

    def payload_nbytes(self):
        # each support vector is an m-dim float64 plus its alpha + label; one int
        # version tag per origin block.
        total = 0
        for _, (_, Xsv, alpha_sv, _) in self.contributions.items():
            total += len(alpha_sv) * (self.m * 8 + 16) + 8
        return total

    def local_update(self):
        self._run_local_steps(self.local_steps, random_order=True)
        self._publish_contribution()

    # ---- kernel SDCA ---------------------------------------------------------
    def _run_local_steps(self, count, random_order):
        self._last_sweep_changed = False
        if self.n == 0:
            return
        y, alpha, K = self.y, self.alpha, self.K_local
        lam_n = self.lam_n
        if lam_n <= 0:
            raise ValueError("lambda and the global sample count must be positive")
        margin_target = 1.0 - self.tau            # DSRR reduced margin (1-tau)
        if random_order:
            order = [CommonState.r.randrange(self.n) for _ in range(count)]
        else:
            order = [i % self.n for i in range(count)]
        for i in order:
            kii = float(K[i, i])
            if kii < 1e-12:
                continue
            yi = float(y[i])
            fi = self.f_local[i] + self.f_ext[i]
            margin = yi * fi
            unsigned_old = alpha[i] * yi
            unsigned_new = min(1.0, max(0.0, unsigned_old
                                        + (margin_target - margin) * lam_n / kii))
            delta = self.step_scale * (yi * unsigned_new - alpha[i])
            if delta == 0.0:
                continue
            alpha[i] += delta
            self.f_local += (delta / lam_n) * K[:, i]   # keep the local score current
            self._last_sweep_changed = True

    def _recompute_external_scores(self):
        """Rebuild the external part of the score cache from accumulated blocks.

        The kernel analogue of SDCA's `_rebuild_w`: only the OTHER origins' support
        vectors go here (our own dual lives in `f_local`), so the two parts never
        double-count this node's contribution.
        """
        self.f_ext = np.zeros(self.n, dtype=np.float64)
        for origin, (_, Xsv, alpha_sv, _) in self.contributions.items():
            if origin == self.node_id or len(alpha_sv) == 0:
                continue
            self.f_ext += kernel_scores(Xsv, alpha_sv, self.Xhat, self.gamma) / self.lam_n

    def _publish_contribution(self, bump=True):
        # Only advance the version (and re-snapshot the block) when this node's
        # dual actually changed this sweep; an unchanged block keeps its version so
        # the versioned delta-send will not re-transmit identical content.
        if bump and not self._last_sweep_changed and self.node_id in self.contributions:
            return
        if bump:
            self.version += 1
        sv = np.nonzero(self.alpha)[0]
        self.contributions[self.node_id] = (
            self.version,
            self.Xhat[sv].copy(),
            self.alpha[sv].copy(),
            self.y[sv].copy(),
        )

    # ---- model readout -------------------------------------------------------
    def all_support_vectors(self):
        """Concatenate this node's own SVs (from its live alpha) with every
        accumulated external block -> (X_hat, alpha, y) for scoring / evaluation."""
        blocks_x, blocks_a, blocks_y = [], [], []
        sv = np.nonzero(self.alpha)[0]
        if len(sv):
            blocks_x.append(self.Xhat[sv])
            blocks_a.append(self.alpha[sv])
            blocks_y.append(self.y[sv])
        for origin, (_, Xsv, alpha_sv, y_sv) in self.contributions.items():
            if origin == self.node_id or len(alpha_sv) == 0:
                continue
            blocks_x.append(Xsv)
            blocks_a.append(alpha_sv)
            blocks_y.append(y_sv)
        if not blocks_x:
            return (np.zeros((0, self.m)), np.zeros(0), np.zeros(0))
        return (np.vstack(blocks_x),
                np.concatenate(blocks_a),
                np.concatenate(blocks_y))

    def accuracy(self):
        Xsv, alpha_sv, _ = self.all_support_vectors()
        scores = kernel_scores(Xsv, alpha_sv, self.Xhat_test, self.gamma)
        preds = np.where(scores >= 0.0, 1.0, -1.0)
        return float(np.mean(preds == self.y_test))

    def _gossip_push(self, node, pid):
        """Versioned delta-send (overrides the generic full-table push).

        Transmit (and charge comm for) only the per-origin support-vector blocks a
        neighbour has not yet seen at the current version. Relaying is preserved —
        a node forwards ANY origin's freshest block, not just its own — so the
        accumulate-not-average consensus still reaches the whole mesh; but a block
        goes to a given neighbour at most once per version instead of re-sending the
        entire (growing) table every cycle. This keeps the communication-cost metric
        honest for the SDCA-vs-baseline comparison.
        """
        link = node.getProtocol(self.LINKABLE_PID)
        deg = link.degree()
        if deg == 0:
            return
        for peer_index in CommonState.r.sample(range(deg), min(self.gossip_k, deg)):
            peer_proto = link.getNeighbor(peer_index).getProtocol(pid)
            if peer_proto.stopped:
                continue
            seen = self._sent.setdefault(peer_proto.node_id, {})
            delta, nbytes = {}, 0
            for origin, entry in self.contributions.items():
                if entry[0] > seen.get(origin, -1):
                    delta[origin] = entry
                    nbytes += len(entry[2]) * (self.m * 8 + 16) + 8
            if not delta:
                continue
            peer_proto.inbox.append({o: e for o, e in delta.items()})
            self.comm_bytes += nbytes
            for origin, entry in delta.items():
                seen[origin] = entry[0]
