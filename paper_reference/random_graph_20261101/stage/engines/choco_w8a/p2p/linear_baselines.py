"""Source-aligned CoLA, pairwise GADGET and CHOCO hinge-loss adaptations.

CoLA and CHOCO require barriers, supplied by LinearBaselineSimulation; they
must not use the asynchronous inbox-draining GossipProtocol loop.
"""
import numpy as np
import os
from src.peersim_python.cdsim import CDProtocol, CDState
from src.peersim_python.core import Network, CommonState
from src.peersim_python.communication import numeric_payload_nbytes, selected_neighbors


def metropolis_matrix(adjacency):
    """Symmetric nonnegative doubly stochastic matrix, with positive diagonal."""
    n = len(adjacency)
    W = np.zeros((n, n))
    for i, neighbors in enumerate(adjacency):
        for j in neighbors:
            if i not in adjacency[j]:
                raise ValueError('CoLA/CHOCO require an undirected graph')
            W[i, j] = 1. / (1 + max(len(neighbors), len(adjacency[j])))
        W[i, i] = 1 - W[i].sum()
    if n > 1 and sorted(np.linalg.eigvalsh(W))[-2] >= 1 - 1e-12:
        raise ValueError('Disconnected mixing graph')
    return W


def signed_coordinate_solve(X, y, alpha, b, curvature, steps, rng):
    """CoLA author fast_cd/svm.pyx objective, fixed coordinate budget.

    min curvature/2 ||X.T alpha||² - (y+b).T alpha, y*alpha in [0,1].
    Reproduce the author's xorshift sampler (one RandomState draw per solve).
    """
    if os.environ.get('SVM_COMPILED_LOCAL') == '1':
        from p2p._linear_fast import cola_steps
        return cola_steps(X.indptr,X.indices,X.data,y,alpha,b,curvature,steps,
                          int(rng.randint(0,0x7fffffff)),X.shape[1])
    a = alpha.copy()
    v = np.asarray(X.T @ a).ravel()
    norms = np.asarray(X.multiply(X).sum(axis=1)).ravel()
    state = int(rng.randint(0, 0x7fffffff))
    for _ in range(steps):
        state ^= (state << 13) & 0xffffffff
        state ^= state >> 17
        state ^= (state << 5) & 0xffffffff
        state &= 0xffffffff
        i = (state % 0x80000000) % len(y)
        if norms[i] == 0:
            coefficient = y[i] * (y[i] + b[i])
            if coefficient > 0:
                a[i] = y[i]
            elif coefficient < 0:
                a[i] = 0.
            continue
        st, end = X.indptr[i:i+2]
        ids, vals = X.indices[st:end], X.data[st:end]
        change = (y[i] + b[i] - curvature * (vals @ v[ids])) / (curvature * norms[i])
        new = y[i] * np.clip(y[i] * (a[i] + change), 0., 1.)
        v[ids] += (new - a[i]) * vals
        a[i] = new
    return a


def gadget_local_step(w, x, y, lam, t):
    """SPegasosGadget.updateClassifier: pre-shrink margin and projection."""
    old_margin = y * (x @ w)
    out = (1 - 1. / t) * w
    if old_margin < 1:
        out = out + y * x / (lam * t)
    norm = np.linalg.norm(out)
    return out * min(1., 1. / (np.sqrt(lam) * norm)) if norm else out


class LinearBaselineProtocol(CDProtocol):
    def __init__(self, method, local_steps=4096, choco_fraction=.2,
                 choco_gamma=.1, choco_lr=.1, choco_batch=128,
                 choco_estimate='final'):
        self.method, self.local_steps = method, local_steps
        self.fraction, self.gamma, self.lr, self.batch = choco_fraction, choco_gamma, choco_lr, choco_batch
        if choco_estimate not in ('final', 'mean', 'linear', 'quadratic'):
            raise ValueError('Unknown CHOCO output estimator')
        self.estimate = choco_estimate
        self.stopped = False
        self.comm_bytes = self.messages_sent = 0
        self.communication_trace, self.metrics = [], []
        self.inbox = []

    def clone(self):
        return type(self)(self.method, self.local_steps, self.fraction, self.gamma,
                          self.lr, self.batch, self.estimate)

    def set_data(self, X, y, Xt, yt, lambda_reg):
        self.X, self.y = X.tocsr().astype(np.float64), np.asarray(y, dtype=float)
        self.X_test, self.y_test = Xt, yt
        self.n, self.d = self.X.shape
        self.lambda_reg = lambda_reg
        self.w = np.zeros(self.d)
        self.w_estimate = np.zeros(self.d)
        self.v = np.zeros(self.d)
        self.alpha = np.zeros(self.n)
        self.local_product = np.zeros(self.d)

    def configure_network(self, node_id, n_global, n_workers, local_steps=None, step_scale=None):
        self.node_id, self.n_global, self.n_workers = node_id, n_global, n_workers
        self.C = 1. / (self.lambda_reg * n_global)
        self.rng = np.random.RandomState(CommonState.experiment_seed + 1009 * node_id)
        if local_steps is not None:
            self.local_steps = int(local_steps)
        self.hats = {i: np.zeros(self.d) for i in range(n_workers)}

    def set_mixing(self, W):
        self.W = W[self.node_id].copy()

    def send(self, peer, payload):
        p = peer.getProtocol(1)
        p.inbox.append((self.node_id, payload))
        size = numeric_payload_nbytes(payload)
        self.comm_bytes += size
        self.messages_sent += 1
        self.communication_trace.append((CDState.getCycle(), (peer.getIndex(),), size))

    def neighbors(self, node):
        link = node.getProtocol(0)
        return [link.getNeighbor(j) for j in range(link.degree())]

    def prepare_cycle(self, node):
        if self.method == 'cola':
            for peer in self.neighbors(node):
                self.send(peer, self.v.copy())

    def nextCycle(self, node, pid):
        if self.stopped:
            return
        if self.method == 'cola':
            mixed = self.W[self.node_id] * self.v
            for sender, value in self.inbox:
                mixed += self.W[sender] * value
            self.inbox.clear()
            sigma_C = self.n_workers * self.C
            b = np.asarray(self.X @ (sigma_C * self.local_product - self.C * mixed)).ravel()
            new = signed_coordinate_solve(self.X, self.y, self.alpha, b,
                                          sigma_C, self.local_steps, self.rng)
            delta = np.asarray(self.X.T @ (new - self.alpha)).ravel()
            self.alpha = new
            self.local_product += delta
            self.v = mixed + self.n_workers * delta
            self.w = self.C * self.v
        elif self.method == 'choco':
            ids = self.rng.randint(self.n, size=self.batch)
            Xb, yb = self.X[ids], self.y[ids]
            active = yb * (Xb @ self.w) < 1
            grad = self.lambda_reg * self.w - np.asarray(Xb.T @ (yb * active)).ravel() / self.batch
            eta = self.lr / (1 + self.lr * self.lambda_reg * CDState.getCycle())
            correction = sum(self.W[j] * (hat - self.hats[self.node_id])
                             for j, hat in self.hats.items())
            self.w = self.w - eta * grad + self.gamma * correction
            residual = self.w - self.hats[self.node_id]
            count = max(1, int(np.ceil(self.fraction * self.d)))
            selected = np.argsort(np.abs(residual), kind='stable')[-count:]
            self.outgoing = {'indices': selected.astype(np.uint32), 'values': residual[selected].copy()}
        elif self.method == 'gadget':
            # Historical Java trains one random local example per activation,
            # sets m_t=cycle+2, then immediately updates BOTH endpoints.
            i = int(self.rng.randint(self.n))
            x = self.X.getrow(i).toarray().ravel()
            self.w = gadget_local_step(self.w, x, self.y[i], self.lambda_reg, CDState.getCycle()+2)
            peers = selected_neighbors(node, CommonState.experiment_seed, CDState.getCycle(), 1)
            if peers:
                peer = peers[0]; other = peer.getProtocol(pid)
                before = self.w.copy(); theirs = other.w.copy()
                # Two directed model transmissions for the pairwise exchange.
                self.send(peer, before)
                other.send(node, theirs)
                merged = (before + theirs) / 2
                self.w, other.w = merged.copy(), merged.copy()
                self.inbox.clear(); other.inbox.clear()
        else:
            raise ValueError(self.method)

    def finish_cycle(self, node):
        if self.method == 'choco':
            for peer in self.neighbors(node):
                self.send(peer, {k: v.copy() for k, v in self.outgoing.items()})
            self.inbox.append((self.node_id, self.outgoing))

    def complete_cycle(self):
        if self.method == 'choco':
            for sender, payload in self.inbox:
                self.hats[sender][payload['indices']] += payload['values']
            self.inbox.clear()
            # Author BaseLogistic.update_estimate, tau=1. Output averaging does
            # not feed back into gradients, compressed messages, or raw w.
            t = CDState.getCycle()
            rho = (1. / (t + 1) if self.estimate == 'mean' else
                   2. / (t + 2) if self.estimate == 'linear' else
                   6. * (t + 1)**2 / ((t + 1) * (6 + 7*t + 2*t*t))
                   if self.estimate == 'quadratic' else 1.)
            self.w_estimate = (1. - rho) * self.w_estimate + rho * self.w

    def evaluation_w(self):
        return self.w_estimate if self.method == 'choco' and self.estimate != 'final' else self.w

    def accuracy(self):
        return float(np.mean(np.where(self.X_test @ self.evaluation_w() >= 0, 1, -1) == self.y_test))
