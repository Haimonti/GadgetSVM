"""FDR-SVM Eq. (9), Algorithm 2, arXiv:2410.03877v3.

The author UCI code calls the *dual norm* pnorm=1. Here it is named
dual_norm explicitly. Clarabel replaces Gurobi, without changing the convex
problem. No Pegasos, squared-norm Wasserstein surrogate or test-based stop.
"""
from dataclasses import dataclass
import numpy as np
import cvxpy as cp
from .transport import ExactAggregation


def norm_epigraph(w, radius, dual_norm):
    if dual_norm not in (1, 2, np.inf):
        raise ValueError('dual_norm must be 1, 2 or np.inf')
    return cp.norm(w, dual_norm) <= radius


def robust_terms(X, y, w, epsilon, kappa, dual_norm):
    radius = cp.Variable(nonneg=True)
    slack = cp.Variable(len(y), nonneg=True)
    margin = cp.multiply(y, X @ w)
    constraints = [slack >= 1 - margin,
                   slack >= 1 + margin - kappa * radius,
                   norm_epigraph(w, radius, dual_norm)]
    return epsilon * radius + cp.sum(slack) / len(y), constraints


def solve_checked(problem, solver='CLARABEL'):
    problem.solve(solver=solver)
    if problem.status != cp.OPTIMAL or not np.isfinite(problem.value):
        raise RuntimeError(f'Local convex solve failed: {problem.status}')
    violation = max((float(np.max(c.violation())) for c in problem.constraints), default=0.)
    if violation > 2e-6:
        raise RuntimeError(f'Convex solution infeasible: {violation}')
    return float(problem.value)


class LocalProblem:
    def __init__(self, X, y, epsilon, kappa, rho, tau=0., dual_norm=1):
        y = np.asarray(y, dtype=float).reshape(-1)
        if X.ndim != 2 or X.shape[0] != len(y) or not len(y):
            raise ValueError('Nonempty aligned X/y required')
        if not np.isin(y, [-1., 1.]).all():
            raise ValueError('FDR labels must be -1/+1')
        if not all(np.isfinite(v) for v in (epsilon, kappa, rho, tau)) or min(epsilon, tau) < 0 or min(kappa, rho) <= 0:
            raise ValueError('Invalid FDR hyperparameters')
        self.w = cp.Variable(X.shape[1])
        self.center = cp.Parameter(X.shape[1])
        loss, constraints = robust_terms(X, y, self.w, epsilon, kappa, dual_norm)
        self.problem = cp.Problem(cp.Minimize(loss + tau * cp.sum_squares(self.w)
                     + rho / 2 * cp.sum_squares(self.w - self.center)), constraints)

    def solve(self, global_w, mu):
        self.center.value = global_w - mu
        value = solve_checked(self.problem)
        return np.asarray(self.w.value).copy(), value


@dataclass
class FDRResult:
    weight: np.ndarray
    local_weights: np.ndarray
    duals: np.ndarray
    history: list
    communication_bytes: int
    link_rounds: int
    stop_reason: str = 'paper_fixed_round_budget'


def train(shards, rounds=100, rho=.01, epsilon=None, kappa=1., tau=0.,
          dual_norm=1, alpha=None, adjacency=None, initial_dual=1.):
    """Synchronous reference; adjacency enables exact P2P dissemination.

    Paper UCI defaults: equal alpha, eps_g=1/(10*N_g), kappa=1,
    w0=0, mu0=1. T/rho require training-only selection. ADMM-SC tau
    must satisfy Theorem 3. A fixed round budget is not a convergence claim.
    """
    g = len(shards)
    if not g or not isinstance(rounds, int) or rounds <= 0:
        raise ValueError('Positive rounds and nonempty clients required')
    d = shards[0][0].shape[1]
    if any(X.shape[1] != d or len(y) == 0 for X, y in shards):
        raise ValueError('Clients require a common feature dimension and nonempty data')
    alpha = np.full(g, 1 / g) if alpha is None else np.asarray(alpha, dtype=float)
    epsilon = [1 / (10 * len(y)) for X, y in shards] if epsilon is None else np.broadcast_to(epsilon, (g,))
    kappas = np.broadcast_to(kappa, (g,))
    taus = np.broadcast_to(tau, (g,))
    if alpha.shape != (g,) or (alpha <= 0).any() or not np.isfinite(alpha).all() or not np.isclose(sum(alpha), 1):
        raise ValueError('Positive alpha summing to one required')
    if np.any(taus > 0) and g > 1:
        denominators = [i * (2*g+1-i) for i in range(1, g)] + [(g-1)*(g+2)]
        if rho > np.min(4 * alpha * taus / denominators) * (1 + 1e-12):
            raise ValueError('ADMM-SC parameters violate Theorem 3 rho bound')
    problems = [LocalProblem(X, y, e, k, rho, t, dual_norm)
                for (X, y), e, k, t in zip(shards, epsilon, kappas, taus)]
    z = np.zeros(d)
    mu = np.full((g, d), float(initial_dual))
    transport = ExactAggregation(adjacency)
    history = []
    transport.broadcast(z.nbytes, g)  # initial w0 distribution
    for t in range(rounds):
        old = z.copy()
        local = np.stack([p.solve(z, u)[0] for p, u in zip(problems, mu)])
        z_all = transport.weighted(list(local + mu), alpha)
        z = z_all[0]
        transport.broadcast(z.nbytes, g)
        mu += local - z
        history.append(dict(round=t+1, weight=z.tolist(),
            primal_residual=float(np.sqrt(np.sum(alpha[:, None] * (local-z)**2))),
            dual_residual=float(rho * np.linalg.norm(z-old)),
            weighted_dual_norm=float(np.linalg.norm(alpha @ mu))))
    return FDRResult(z, local, mu, history, transport.bytes, transport.link_rounds)


def centralized(shards, epsilon=None, kappa=1., tau=0., dual_norm=1, alpha=None):
    """Independent direct solve of the *same robust objective*, without ADMM."""
    g = len(shards)
    alpha = np.full(g, 1/g) if alpha is None else np.asarray(alpha)
    epsilon = [1/(10*len(y)) for X, y in shards] if epsilon is None else np.broadcast_to(epsilon, (g,))
    w = cp.Variable(shards[0][0].shape[1])
    constraints, objective = [], 0
    for (X, y), a, e, k, t in zip(shards, alpha, epsilon, np.broadcast_to(kappa, (g,)), np.broadcast_to(tau, (g,))):
        loss, cons = robust_terms(X, np.asarray(y), w, e, k, dual_norm)
        objective += a * (loss + t * cp.sum_squares(w))
        constraints.extend(cons)
    problem = cp.Problem(cp.Minimize(objective), constraints)
    value = solve_checked(problem)
    return np.asarray(w.value), value
