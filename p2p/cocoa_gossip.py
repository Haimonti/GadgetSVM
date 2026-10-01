"""P2P-CoCoA and P2P-CoCoA+ on the refactored gossip layer.

Port of `cocoa/cocoa.py` and `cocoa/cocoa_plus.py` onto `GossipProtocol`, the
same way `p2p/bdsvm_gossip.py` ported BDSVM: one harness, one transport, one
aggregator, so a CoCoA curve and an SDCA curve differ only in the algorithm.

Why it sits on SDCAProtocol
---------------------------
CoCoA *is* distributed SDCA with a different commit rule, so it needs exactly
the bookkeeping `SDCAProtocol` already has: worker k owns a disjoint dual block
alpha_k (signed, alpha_i = y_i * a_i), publishes the versioned contribution
X_k.T @ alpha_k / (lambda * n_global), and holds w as the sum of the freshest
contribution it knows from every origin. That keeps w and alpha describing the
same model under duplicate and out-of-order delivery, and lets the unchanged
`GlobalEvaluator` measure a real global duality gap. The old gossip port
(`p2p/cocoa_protocol.py`) added raw increments with an IncrementAggregator,
which is the failure mode `src/peersim_python/aggregator.py` documents.

What CoCoA changes relative to SDCAProtocol is only the local round:

    server code                          here
    w broadcast to all K workers    ->   w = sum of the contributions this
                                         node has heard (possibly stale)
    H local SDCA steps on a copy    ->   same, on a copy of that w
    alpha_k += scaling * dalpha_k   ->   same, then republish c_k (version+1)
    w += scaling * sum_k dw_k       ->   follows from the contributions

    CoCoA   (averaging)  scaling = beta / K,  local w moves by every step
    CoCoA+  (adding)     scaling = gamma,     local w frozen, sigma = K * gamma

K is the global worker count (`n_workers` from DataInitializer), not the node
degree: with versioned contributions every node's w is a sum over all K
origins, so the server's K is the right one. The local update rules are the
same as `cocoa/cocoa.py::_local_sdca` (projected dual step, alpha in [0, 1],
qii = ||x||^2, times sigma for CoCoA+), written in closed form on CSR rows.

`SDCA_LOCAL_STEPS` sets H, so a CoCoA round does the same coordinate work as an
SDCA round. `SDCA_STEP_SCALE` is SDCA's damping and is not read: CoCoA's own
scaling takes its place.
"""

import numpy as np

from src.peersim_python.core.common_state import CommonState
from src.peersim_python.sdca_protocol import SDCAProtocol


class CoCoAGossipProtocol(SDCAProtocol):
    """CoCoA — averaging: alpha_k += (beta / K) * dalpha_k."""

    PLUS = False

    def __init__(self, beta=1.0, gamma=1.0):
        super().__init__()
        self.beta = beta        # CoCoA averaging parameter (1 in the paper)
        self.gamma = gamma      # CoCoA+ additive parameter (1 in the paper)
        self.scaling = 1.0
        self.sigma = 1.0

    def clone(self):
        c = type(self)(beta=self.beta, gamma=self.gamma)
        c.gossip_k = self.gossip_k
        return c

    def configure_network(self, node_id, n_global, n_workers,
                          local_steps=None, step_scale=None):
        """Global constants, as for SDCA; `step_scale` is accepted and ignored."""
        self.node_id = int(node_id)
        self.n_global = int(n_global)
        self.n_workers = int(n_workers)
        self.local_steps = max(1, int(local_steps or self.n or 1))
        k = max(self.n_workers, 1)
        if self.PLUS:
            self.scaling = self.gamma
            self.sigma = k * self.gamma
        else:
            self.scaling = self.beta / k
            self.sigma = 1.0
        self._publish_contribution(bump=False)

    def warm_start(self):
        """No-op. SDCA's warm start is a full-strength local pass whose results
        are then summed across nodes; CoCoA's scaling exists precisely so that
        local solutions are not summed at full strength, and the reference
        implementation starts from alpha = 0."""
        return None

    def local_update(self):
        dalpha = self._local_sdca()
        if dalpha is not None:
            self.alpha += self.scaling * dalpha
        self._publish_contribution()

    def _local_sdca(self):
        """H steps of local SDCA against the w this node currently knows.

        Returns the signed dual change dalpha; alpha itself is not touched, so
        the caller commits only `scaling * dalpha`.
        """
        if self.n == 0:
            return None
        X, y, alpha = self.X, self.y, self.alpha
        indptr, indices, data = X.indptr, X.indices, X.data
        lam_n = self.lambda_reg * self.n_global
        if lam_n <= 0:
            raise ValueError("lambda and the global sample count must be positive")
        sigma = self.sigma
        # CoCoA:  w_loc tracks w + dw (sigma = 1).
        # CoCoA+: w_loc tracks w + sigma * dw, i.e. the frozen w plus the
        #         sigma-weighted local progress of the reference solver.
        w_loc = self.w.copy()
        dalpha = np.zeros(self.n, dtype=np.float64)

        for _ in range(self.local_steps):
            i = CommonState.r.randrange(self.n)
            start, end = indptr[i], indptr[i + 1]
            cols, vals = indices[start:end], data[start:end]
            norm_sq = float(np.dot(vals, vals))
            if norm_sq < 1e-12:
                continue
            yi = float(y[i])
            a = (alpha[i] + dalpha[i]) * yi            # unsigned, in [0, 1]
            margin = yi * float(np.dot(w_loc[cols], vals))
            a_new = min(1.0, max(0.0, a + (1.0 - margin) * lam_n / (sigma * norm_sq)))
            if a_new == a:
                continue
            delta = yi * (a_new - a)                   # signed
            dalpha[i] += delta
            w_loc[cols] += (sigma * delta / lam_n) * vals
        return dalpha


class CoCoAPlusGossipProtocol(CoCoAGossipProtocol):
    """CoCoA+ — adding: alpha_k += gamma * dalpha_k, subproblem sigma = K * gamma."""

    PLUS = True
