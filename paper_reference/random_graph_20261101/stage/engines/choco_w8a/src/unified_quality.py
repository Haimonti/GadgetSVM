"""One training objective and an offline, certified quality gate for all methods.

The evaluator never reads a test label. Its communication is offline observer
work, excluded from payload counts and included in training runtime.
"""
import time
import numpy as np
from src.peersim_python.core import Control, Network
from src.peersim_python.cdsim import CDState
from p2p.bdsvm_gossip import BDSVMGossipProtocol


def objectives(X, y, weights, lam):
    weights = np.atleast_2d(np.asarray(weights, dtype=np.float64))
    hinge = np.maximum(0., 1. - np.asarray(y)[:, None] * (X @ weights.T)).mean(axis=0)
    return hinge + .5 * lam * np.einsum('ij,ij->i', weights, weights), hinge


def dual_lower_bound(X, signed_alpha, lam):
    """Feasible signed alpha_i=y_i*a_i, a_i in [0,1], no bias constraint."""
    w = np.asarray(X.T @ signed_alpha).ravel() / (lam * X.shape[0])
    return w


def subspace_excess_bound(reference_w, basis, lam, reference_gap):
    """Conservative strong-convexity error bound for a fixed row span."""
    gram = basis @ basis.T
    coef = np.linalg.solve(gram, basis @ reference_w)
    residual = reference_w - basis.T @ coef
    smallest = float(np.linalg.eigvalsh(gram)[0])
    if smallest <= 0:
        raise ValueError('Budget basis lost row rank')
    projection_error_sq = float(np.linalg.norm(basis @ residual)**2 / smallest)
    distance = max(0., float(np.sqrt(max(0., residual @ residual - projection_error_sq)))
                   - 1e-7 * (1. + float(np.linalg.norm(reference_w))))
    radius = float(np.sqrt(2. * max(0., reference_gap) / lam))
    excess = .5 * lam * max(0., distance - radius)**2
    return distance, radius, excess


class CommonQualityEvaluator(Control):
    def __init__(self, X, y, lam, lower_bound, target, cap, started,
                 pid=1, every=10):
        self.X, self.y, self.lam = X, y, lam
        self.lower_bound, self.target = float(lower_bound), float(target)
        self.cap, self.started, self.pid, self.every = cap, started, pid, every
        self.history, self.reached = [], False
        self.final_objectives = self.final_hinges = None

    @staticmethod
    def weights(p):
        if hasattr(p, 'kernel'):
            if p.kernel != 'linear' or len(p.beta) != p.P:
                raise ValueError('Common linear objective requires no-bias linear BDSVM')
            return np.asarray(p.p.T @ p.beta).ravel()
        return p.evaluation_w() if hasattr(p, 'evaluation_w') else p.w

    def execute(self):
        number = CDState.getCycle() + 1
        if number != 1 and number % self.every and number != self.cap:
            return False
        ps = [Network.get(i).getProtocol(self.pid) for i in range(Network.size())]
        W = np.stack([self.weights(p) for p in ps])
        primal, hinge = objectives(self.X, self.y, W, self.lam)
        gaps = primal - self.lower_bound
        if not np.all(np.isfinite(gaps)):
            raise FloatingPointError('Nonfinite training objective')
        if np.min(gaps) < -1e-7:
            raise ValueError('Primal below feasible dual lower bound: invalid reference')
        self.reached = bool(np.all(gaps <= self.target))
        self.final_objectives, self.final_hinges = primal, hinge
        self.history.append({'round': number, 'train_hinge': float(hinge.mean()),
            'train_objective': float(primal.mean()), 'objective_per_peer': primal.tolist(),
            'quality_gap_per_peer': gaps.tolist(), 'max_quality_gap': float(gaps.max()),
            'comm_bytes': sum(p.comm_bytes for p in ps),
            'seconds': time.perf_counter() - self.started})
        if number % 100 == 0 or self.reached:
            print(f'quality round={number} max_gap={gaps.max():.8g} '
                  f'target={self.target} reached={self.reached}', flush=True)
        if self.reached:
            for p in ps:
                p.stopped = True
                p.stop_cycle = number - 1
        return self.reached


class LinearNoBiasBDSVM(BDSVMGossipProtocol):
    """Same IRWLS sufficient-statistic update on the shared b=0 problem.

For its native objective, C=1/(2*lambda*N) makes the regularizer exactly
lambda*||w||^2/2 after division by 2*C*N. The pre-image budget is retained.
"""
    def clone(self):
        return type(self)(gossip_k=self.gossip_k, P=self.P, C=self.C,
            rho=self.rho, gamma=self.gamma, gamma_mult=self.gamma_mult,
            preimage=self.preimage, arch_seed=self.arch_seed,
            ey_floor=self.ey_floor, kernel=self.kernel)

    def set_data(self, *args):
        super().set_data(*args)
        self.beta = np.zeros(self.P)

    def _initialize_kernel(self):
        if self.kernel != 'linear':
            raise ValueError('This experiment is linear only')
        # Invertible change of coordinates: preserve the exact budget row
        # span while making its Gram matrix well conditioned. This avoids a
        # trace-scaled ridge that would change the shared objective.
        gram = self.p @ self.p.T
        chol = np.linalg.cholesky(gram)
        self.p = np.linalg.solve(chol, self.p)
        self.Kpp = self.p @ self.p.T
        self.Km = np.asarray(self._kernel(self.X, self.p))
        self.kernel_ready = True
        self._publish()

    def local_update(self):
        if len(self.contributions) < self.n_workers:
            return
        A = np.zeros((self.P, self.P)); rhs = np.zeros(self.P)
        for _, (C_k, d_k), _ in self.contributions.values():
            A += C_k; rhs += d_k
        A += self.Kpp
        try:
            new = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            new = np.linalg.lstsq(A, rhs, rcond=None)[0]
        previous = self.beta
        self.beta = self.rho * previous + (1. - self.rho) * new
        self.rel_change = float(np.linalg.norm(self.beta - previous)) / max(
            float(np.linalg.norm(previous)), 1e-30)
        self._publish()

    def accuracy(self):
        w = self.p.T @ self.beta
        return float(np.mean(np.where(self.X_test @ w >= 0., 1., -1.) == self.y_test))
