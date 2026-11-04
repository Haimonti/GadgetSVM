"""COLA node — `cola/algo.py::cola` mapped onto synchronous engine rounds.

Per round t (official loop body, one PeerSim cycle):

    averaged_v = sum_{j in N(k) u {k}} W_kj v_j          # comm.local_average
    dx, dv     = localsolver.solve(averaged_v, Akxk, xk)
    xk        += gamma * dx
    Akxk      += gamma * dv
    v_k        = averaged_v + gamma * dv * K              # local_vs[rank]
    send v_k to every neighbour                           # p2p_communicate_neighborhood_tensors

Phase A (`compute_round`) runs the first five lines from the neighbour values
received at the end of round t-1; the barrier then delivers every v_k and
commits. Before round 1 every node holds v = 0 for itself and its neighbours and
exchanges those zeros once (the official initial communication).

The node's model is w_k = C * v_k (COLA's `grad_f(v)`), i.e. its estimate of the
global primal; the true global primal C * sum_k Akxk is what COLA's own global
monitor certifies.
"""

import numpy as np

from src.peersim_python.sync_protocol import SyncGossipProtocol
from src.baselines.cola.local_solver import ColaLinearSVMSolver


class ColaProtocol(SyncGossipProtocol):

    def clone(self):
        return ColaProtocol()

    def set_data(self, X, y, X_test, y_test):
        self.X = X.tocsr()
        self.y = np.asarray(y, dtype=np.float64)
        self.X_test = X_test.tocsr()
        self.y_test = np.asarray(y_test, dtype=np.float64)
        self.n, self.d = self.X.shape

    def configure(self, node_id, neighbours, weights, n_nodes, C, gamma, local_iters,
                  theta, random_state, compressor):
        """`weights` maps every j in N(k) u {k} to W_kj."""
        self.node_id = int(node_id)
        self.neighbours = list(neighbours)
        self.weights = dict(weights)
        self.K = int(n_nodes)
        self.C = float(C)
        self.gamma = float(gamma)
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError(f"COLA gamma must be in (0, 1], got {gamma}")
        self.solver = ColaLinearSVMSolver(self.X, self.y, self.C, sigma=self.gamma * self.K,
                                          local_iters=local_iters, theta=theta,
                                          random_state=random_state)
        self.compressor = compressor
        self.xk = np.zeros(self.n)
        self.Akxk = np.zeros(self.d)
        self.v = np.zeros(self.d)                         # local_vs[rank]
        self.inbox = {j: np.zeros(self.d) for j in self.neighbours}
        self.ready = True

    def initial_exchange(self):
        """The official zero-tensor exchange before round 1 (counted in bytes)."""
        self.comm_bytes += len(self.neighbours) * self.message_nbytes(self.v)

    @property
    def w(self):
        return self.C * self.v

    # ---- SyncGossipProtocol hooks ---------------------------------------------
    def compute_round(self):
        averaged_v = self.weights[self.node_id] * self.v
        for j in self.neighbours:
            averaged_v = averaged_v + self.weights[j] * self.inbox[j]
        delta_x, delta_v = self.solver.solve(averaged_v, self.Akxk, self.xk)
        self.xk += self.gamma * delta_x
        self.Akxk += self.gamma * delta_v
        v_new = averaged_v + self.gamma * delta_v * self.K
        decoded, nbytes = self.compressor(v_new)
        self._nbytes = nbytes
        self._v_new = v_new
        return decoded

    def message_nbytes(self, msg):
        return self._nbytes if msg is not self.v else self.compressor(msg)[1]

    def apply_round(self, msg):
        self.v = self._v_new

    def accuracy(self):
        preds = np.where(self.X_test.dot(self.w) >= 0.0, 1.0, -1.0)
        return float(np.mean(preds == self.y_test))
