"""Train-only monitoring and global patience stop for comparison experiments."""
from collections import deque
import time

import numpy as np
import scipy.sparse as sp

from src.peersim_python.core import Control, Network
from src.peersim_python.cdsim import CDState


class ObjectivePatience:
    """Patience counts training rounds, not the number of plot samples.

The guard against deterioration prevents an objective diverging away from its
best value from being labelled converged merely for failing to improve.
"""
    def __init__(self, n_peers, window=10, patience=50, rtol=1e-3, atol=1e-8):
        if min(n_peers, window, patience) < 1 or rtol < 0 or atol < 0:
            raise ValueError('Invalid convergence parameters')
        self.window, self.patience, self.rtol, self.atol = window, patience, rtol, atol
        self.values = deque(maxlen=window)
        self.best = np.full(n_peers, np.inf)
        self.last_improved = np.zeros(n_peers, dtype=int)
        self.smooth = None

    def update(self, objectives, round_number, ready=True):
        objectives = np.asarray(objectives, dtype=float)
        if not ready or not np.all(np.isfinite(objectives)):
            self.values.clear()
            self.best.fill(np.inf)
            self.last_improved.fill(round_number)
            self.smooth = None
            return False
        self.values.append(objectives.copy())
        if len(self.values) < self.window:
            return False
        self.smooth = np.mean(self.values, axis=0)
        finite = np.isfinite(self.best)
        tolerance = self.atol + self.rtol * np.where(finite, np.abs(self.best), 0.)
        improved = ~finite | (self.smooth < self.best - tolerance)
        self.best[improved] = self.smooth[improved]
        self.last_improved[improved] = round_number
        tolerance = self.atol + self.rtol * np.abs(self.best)
        stable = np.abs(self.smooth - self.best) <= tolerance
        return bool(np.all(stable & (round_number - self.last_improved >= self.patience)))


class WindowStability:
    """Stop after two stable adjacent-window comparisons for every peer.

    Means over complete, non-overlapping windows damp stochastic SGD noise.
    No historical minimum is used as a stationary noisy sequence can remain
    above its luckiest observation indefinitely. This is a stationarity rule,
    not a proof of optimality.
    """
    def __init__(self, n_peers, window=50, checks=2, rtol=1e-3, atol=1e-8):
        if min(n_peers, window, checks) < 1 or min(rtol, atol) < 0:
            raise ValueError('Invalid stability parameters')
        self.n_peers, self.window, self.checks = n_peers, window, checks
        self.rtol, self.atol = rtol, atol
        self.values = deque(maxlen=2 * window)
        self.samples = self.stable_checks = 0

    def update(self, objectives, round_number, ready=True):
        values = np.asarray(objectives, dtype=float)
        if values.shape != (self.n_peers,):
            raise ValueError('One objective per peer required')
        if not ready or not np.all(np.isfinite(values)):
            self.values.clear()
            self.samples = self.stable_checks = 0
            return False
        self.values.append(values.copy())
        self.samples += 1
        if len(self.values) < 2 * self.window or self.samples % self.window:
            return False
        history = np.array(self.values)
        previous = history[:self.window].mean(axis=0)
        current = history[self.window:].mean(axis=0)
        tolerance = self.atol + self.rtol * np.abs(previous)
        stable = np.all(np.abs(current - previous) <= tolerance)
        self.stable_checks = self.stable_checks + 1 if stable else 0
        return self.stable_checks >= self.checks


class TrainConvergenceEvaluator(Control):
    """No test access. All peers remain active until a global stop decision.

Global train objectives are measured offline on stacked shards; this is a
simulation observer, not a new distributed evaluation protocol. Its CPU time
is included, and its hypothetical deployment traffic is excluded explicitly.
"""
    def __init__(self, pid, config, cycles, started):
        self.pid, self.total_cycles, self.started = pid, cycles, started
        self.plot_every = int(config.get('EVAL_EVERY', 10))
        self.rule = ObjectivePatience(config['NUM_WORKERS'],
            window=config.get('CONVERGENCE_WINDOW', 10),
            patience=config.get('CONVERGENCE_PATIENCE', 50),
            rtol=config.get('CONVERGENCE_RTOL', 1e-3))
        if config.get('CONVERGENCE_RULE') == 'adjacent_windows':
            self.rule = WindowStability(config['NUM_WORKERS'],
                window=config.get('CONVERGENCE_STABILITY_WINDOW', 50),
                checks=2, rtol=config.get('CONVERGENCE_RTOL', 1e-3))
        self.X = self.y = None
        self.kernel_cache = {}
        self.converged = False
        self.history = []

    def execute(self):
        ps = [Network.get(i).getProtocol(self.pid) for i in range(Network.size())]
        if self.X is None:
            self.X = sp.vstack([p.X for p in ps]).tocsr()
            self.y = np.concatenate([p.y for p in ps])
        round_number = CDState.getCycle() + 1
        hinges, objectives = [], []
        ready = True
        # Each peer's full-train loss. Vectorize the common linear case.
        if not hasattr(ps[0], 'kernel') and not hasattr(ps[0], 'phase'):
            W = np.stack([p.w for p in ps])
            H = np.maximum(0., 1 - self.y[:, None] * (self.X @ W.T)).mean(axis=0)
        else:
            H = None
        for i, p in enumerate(ps):
            if hasattr(p, 'phase') and p.phase == 1:
                hinge = objective = float('nan')
                ready = False
            elif hasattr(p, 'kernel'):
                if not getattr(p,'kernel_ready',True):
                    ready=False
                    hinges.append(float('nan'))
                    objectives.append(float('nan'))
                    continue
                # RBF gamma is shared after training-only P2P calibration;
                # do not incorrectly reuse a kernel across different gammas.
                key = (p.kernel, p.gamma, p.P)
                if p.kernel == 'linear':
                    scores = self.X @ (p.p.T @ p.beta[:p.P]) + p.beta[p.P]
                else:
                    if key not in self.kernel_cache:
                        self.kernel_cache[key] = p._kernel(self.X, p.p)
                    scores = self.kernel_cache[key] @ p.beta[:p.P] + p.beta[p.P]
                hinge = float(np.maximum(0., 1 - self.y * scores).mean())
                # IRWLS uses a_i = 2C / margin_residual, corresponding to
                # 1/2 beta'K beta + 2C sum hinge. Scale by 2CN.
                objective = hinge + float(p.beta @ p.Kpp @ p.beta) / (4 * p.C * len(self.y))
                ready &= len(p.contributions) == len(ps)
            else:
                if hasattr(p, 'phase'):
                    ready &= p.convergence_ready
                    scores = self.X @ (p.V.T @ p.w)
                    hinge = float(np.maximum(0., 1 - self.y * scores).mean())
                else:
                    hinge = float(H[i])
                    if hasattr(p, 'contributions'):
                        ready &= len(p.contributions) == len(ps)
                curvature = p.lambda_reg
                if hasattr(p, 'eps'):
                    curvature += sum(peer.n * peer.eps for peer in ps) / len(self.y)
                objective = hinge + .5 * curvature * float(p.w @ p.w)
            hinges.append(hinge)
            objectives.append(objective)
        self.converged = self.rule.update(objectives, round_number, ready)
        if round_number % 100 == 0:
            print(f'train round={round_number} objective={np.mean(objectives):.7g} '
                  f'payload_MB={sum(p.comm_bytes for p in ps)/1e6:.3f} '
                  f'ready={ready} plateau={self.converged}', flush=True)
        if self.converged:
            for p in ps:
                p.stopped = True
                p.stop_cycle = round_number - 1
                p.stop_reason = 'train_objective_plateau'
        if round_number == 1 or round_number % self.plot_every == 0 or self.converged or round_number == self.total_cycles:
            elapsed = time.perf_counter() - self.started
            for p, hinge, objective in zip(ps, hinges, objectives):
                p.metrics.append({'round': round_number,
                    'hinge_loss': hinge if np.isfinite(hinge) else None,
                    'primal': objective if np.isfinite(objective) else None,
                    'comm_bytes': p.comm_bytes,
                    'stopped': self.converged, 'wall_time': elapsed})
            finite_hinges = [v for v in hinges if np.isfinite(v)]
            self.history.append({'round': round_number,
                'train_hinge': float(np.mean(finite_hinges)) if finite_hinges else None,
                'train_objective': float(np.mean(objectives)) if np.all(np.isfinite(objectives)) else None,
                'comm_bytes': sum(p.comm_bytes for p in ps), 'seconds': elapsed})
        return self.converged
