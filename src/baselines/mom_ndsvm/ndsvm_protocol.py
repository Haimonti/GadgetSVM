"""NDSVMProtocol — one node of MoM-NDSVM (Forero et al., JMLR 2010, Algorithm 3).

A cycle-driven protocol on the shared PeerSim engine. One engine cycle is one
iteration ``t`` of Algorithm 3, split across the engine's two hooks:

* ``nextCycle`` (lines 2-6, phase A): from the node's own state and the copies of
  its neighbours' last broadcast held in its inbox, form ``f~_j(t)``, ``h_j(t)``,
  solve (35) for ``lambda_j(t+1)``, then (36)/(37) for ``w~_j(t+1)``, ``b_j(t+1)``.
  The readout ``(a, c, b)`` of ``g_j`` at ``t+1`` is taken here (E2 pairing). The new
  ``(w~, b)`` is only *pending*: no other node can see it yet, so the activation
  order of the pass cannot leak ``t+1`` values into another node's ``t`` step.
* ``deliver`` / ``commit`` / ``finish_round`` (lines 7-14, phases B and C), called by
  ``SynchronousRoundBarrier`` once every node has finished phase A: send ``(w~, b)``
  to **all** neighbours, then the dual ascent (38)/(39).

This is a ``CDProtocol`` rather than a ``GossipProtocol`` tenant: gossip pushes to a
random subset of neighbours in the middle of the pass, whereas Algorithm 3 is a
synchronous broadcast to every neighbour.
"""

import time

import numpy as np

from src.peersim_python.cdsim.cd_protocol import CDProtocol
from src.peersim_python.cdsim.cd_state import CDState
from src.baselines.mom_ndsvm.core import LocalProblem, f_tilde, h_scalar, dual_ascent
from src.baselines.mom_ndsvm.qp import BoxQPSolver


class NDSVMProtocol(CDProtocol):
    """MoM-NDSVM learner for one node; all arithmetic lives in ``core.LocalProblem``."""

    def __init__(self):
        self.ready = False
        self.node_id = None
        self.neighbours = []
        self.inbox = {}                 # neighbour id -> (w~_i, b_i) as last received
        self.metrics = []
        self.comm_bytes = 0
        self.stopped = False
        self.stop_cycle = None
        self.start = None
        self.compute_time = 0.0
        self.eval_time = 0.0
        self._pending = None
        self._last_cycle = None
        self._f = None
        self._h = None

    def clone(self):
        return NDSVMProtocol()

    # ---- setup -------------------------------------------------------------
    def set_data(self, X, y, X_test, y_test, lambda_reg):
        """Dense features of this node's shard (raw, or DSRR-projected by flag)."""
        self.X = np.asarray(X, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)
        self.X_test = X_test
        self.y_test = np.asarray(y_test, dtype=np.float64)
        self.lambda_reg = float(lambda_reg)

    def configure_network(self, node_id, n_global, n_workers, neighbours, Gamma, K_GG,
                          kernel, eta, JC, qp_tol, qp_max_passes, init_rng):
        """Precompute node j's fixed part of Prop. 3 and draw Algorithm 3's random
        ``w~_j(0)``, ``b_j(0)`` (``alpha_j(0) = 0``, ``beta_j(0) = 0``)."""
        self.node_id = int(node_id)
        self.n_global = int(n_global)
        self.n_workers = int(n_workers)
        self.neighbours = list(neighbours)
        self.Gamma = Gamma
        self.kernel = kernel
        self.eta = float(eta)
        self.JC = float(JC)
        self.model = LocalProblem(self.X, self.y, Gamma, K_GG, kernel, self.eta,
                                  len(self.neighbours), self.JC)
        self.qp = BoxQPSolver(self.model.Q, self.JC, tol=qp_tol, max_passes=qp_max_passes)
        L = Gamma.shape[0]
        self.alpha = np.zeros(L)
        self.beta = 0.0
        self.w_tilde = init_rng.standard_normal(L)
        self.b = float(init_rng.standard_normal())
        self._prev_w = self.w_tilde.copy()
        self._prev_b = self.b
        self.lam = np.zeros(self.model.N)
        self.Qlam = np.zeros(self.model.N)
        self.a = np.zeros(self.model.N)
        self.c = np.zeros(L)
        self.b_read = self.b
        self.ready = True
        self.start = time.time()        # after this node's local precompute

    # ---- messaging -----------------------------------------------------------
    def receive(self, sender, w_tilde, b):
        self.inbox[sender] = (w_tilde, b)

    def broadcast_initial(self, protos):
        """Send ``(w~_j(0), b_j(0))`` to every neighbour so ``f~_j(0)``, ``h_j(0)`` exist."""
        for i in self.neighbours:
            protos[i].receive(self.node_id, self.w_tilde.copy(), self.b)
        self.comm_bytes += self._message_bytes()

    def _message_bytes(self):
        """One ``(L+1)``-float64 message to each neighbour."""
        return len(self.neighbours) * (self.w_tilde.shape[0] + 1) * 8

    # ---- Algorithm 3, lines 2-6 (phase A) --------------------------------------
    def nextCycle(self, node, protocolID):
        if not self.ready or self.stopped:
            return
        cycle = CDState.getCycle()
        if self._last_cycle == cycle:
            raise RuntimeError("MoM-NDSVM is synchronous: a node may run once per cycle")
        self._last_cycle = cycle
        t0 = time.perf_counter()
        w_nb = [self.inbox[i][0] for i in self.neighbours]
        b_nb = [self.inbox[i][1] for i in self.neighbours]
        f = f_tilde(self.alpha, self.w_tilde, w_nb, self.eta)          # f~_j(t)
        h = h_scalar(self.beta, self.b, b_nb, self.eta)                # h_j(t)
        lam = self.qp.solve(self.model.lin_term(f, h))                 # (35), E1-corrected
        self.lam = lam.copy()
        self.Qlam = self.qp.Qlam.copy()
        a, c, w_new, b_new = self.model.iterate(self.lam, f, h)        # (32),(33),(36),(37)
        self.a, self.c, self.b_read = a, c, b_new                      # g_j at t+1 (E2)
        self._f, self._h = f, h
        self._pending = (w_new, b_new)
        self.compute_time += time.perf_counter() - t0

    # ---- Algorithm 3, lines 7-14 (phases B and C), driven by the barrier --------
    def deliver(self, protos):
        """Line 8: broadcast ``(w~_j(t+1), b_j(t+1))`` to all neighbours."""
        w_new, b_new = self._pending
        for i in self.neighbours:
            protos[i].receive(self.node_id, w_new.copy(), b_new)
        self.comm_bytes += self._message_bytes()

    def commit(self):
        """Adopt the node's own ``t+1`` iterate once every message is out."""
        self._prev_w, self._prev_b = self.w_tilde, self.b
        self.w_tilde, self.b = self._pending
        self._pending = None

    def finish_round(self):
        """Lines 11-12: (38) and (39) with every neighbour's ``t+1`` values."""
        t0 = time.perf_counter()
        w_nb = [self.inbox[i][0] for i in self.neighbours]
        b_nb = [self.inbox[i][1] for i in self.neighbours]
        self.alpha, self.beta = dual_ascent(self.alpha, self.beta, self.w_tilde, self.b,
                                            w_nb, b_nb, self.eta)
        self.compute_time += time.perf_counter() - t0

    # ---- model readout ---------------------------------------------------------
    def decision_function(self, Z, K_ZG=None, chunk=4096):
        """(31): ``g_j(z) = sum_n a_n K(z, x_n) + sum_l c_l K(z, chi_l) + b``, summed over
        the support vectors only (``a_n != 0``), in row chunks."""
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
