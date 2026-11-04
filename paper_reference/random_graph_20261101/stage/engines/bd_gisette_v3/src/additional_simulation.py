"""Existing FedSSL/FDR local solvers on the current PeerSim cycle engine.

Evaluation uses the full training set, just as the existing comparison runs.
Stop only when all peers are stable for three evaluation windows, after SSL's
phase transition. No duality gap is claimed for these primal methods.
"""
import time
import numpy as np
import scipy.sparse as sp

from p2p.fedssl_protocol import FedSSLAMCProtocol
from p2p.fdr_svm_protocol import FDRSVMProtocol
from src.fedavg_simulation import FedAvgSimulation
from src.peersim_python.core import Network, GeneralNode, CommonState, Control
from src.peersim_python.idle_protocol import IdleProtocol
from src.peersim_python.observers import DataInitializer
from src.peersim_python.cdsim import CDSimulator, CDState
from src.peersim_python.logger import logger


class ObservedSSL(FedSSLAMCProtocol):
    def _record(self):
        pass  # one global evaluator, rather than a full local pass each cycle


class ObservedFDR(FDRSVMProtocol):
    def _setup_extra(self):
        super()._setup_extra()
        self.gram = None
        if (0 < self.n <= 2000 and self.d > 512 and self.X.nnz/self.n > 512
                and np.all(np.diff(self.X.indptr) > 0)):
            # Equivalent row-space recursion for small, feature-dense shards.
            dense = np.asarray(self.X.toarray(), dtype=np.float64)
            self.gram = np.ascontiguousarray(dense @ dense.T)

    def _prox(self, X, y, v, A):
        if self.gram is not None:
            from p2p._prox_gram import prox_pegasos_gram
            return prox_pegasos_gram(X, self.gram, y, v, A, self.rho,
                                     self.n_local_steps, self.rng)
        from p2p._prox_fast import prox_pegasos_fast
        return prox_pegasos_fast(X, y, v, A, self.rho, self.n_local_steps, self.rng)

    def _record(self):
        pass


class AdditionalEvaluator(Control):
    def __init__(self, cfg, shards, cycles, start):
        self.cfg, self.cycles, self.start = cfg, cycles, start
        self.X = sp.vstack([s['X_csr'] for s in shards]).tocsr()
        self.y = np.concatenate([s['y'] for s in shards])
        self.prev = None
        self.stable = 0

    def execute(self):
        cycle = CDState.getCycle()
        if cycle % self.cfg['EVAL_EVERY'] and cycle != self.cycles - 1:
            return False
        ps = [Network.get(i).getProtocol(1) for i in range(Network.size())]
        ready = all(getattr(p, 'phase', 2) == 2 for p in ps)
        W = np.stack([p.w for p in ps])
        bias = np.array([getattr(p, 'bias', 0.) for p in ps])
        # Bias is included in the movement/consensus measurements as well.
        state = np.column_stack([W, bias])
        change = np.full(len(ps), np.inf)
        if ready and self.prev is not None:
            change = np.linalg.norm(state-self.prev, axis=1) / np.maximum(
                np.linalg.norm(self.prev, axis=1), 1e-12)
        self.prev = state.copy() if ready else None
        self.stable = self.stable + 1 if ready and np.all(change < 1e-3) else 0
        stop = self.cfg['STOP_ON_THRESHOLD'] and self.stable >= 3
        hinges = (np.maximum(0., 1.-self.y[:, None]*(self.X.dot(W.T)+bias)).mean(axis=0)
                  if ready else np.full(len(ps), np.nan))
        for i, p in enumerate(ps):
            p.stopped = bool(stop)
            if stop:
                p.stop_cycle, p.stop_reason = cycle, 'all peers stable for 3 evaluation windows'
            p.metrics.append(dict(
                round=cycle+1, accuracy=p.accuracy(), hinge_loss=float(hinges[i]),
                primal=float(hinges[i]+p.lambda_reg/2*np.dot(p.w, p.w)),
                dual=float('nan'), duality_gap=float('nan'), rel_change=float(change[i]),
                consensus_error=(float(np.linalg.norm(state[i]-state.mean(axis=0)))
                                 if ready else float('nan')),
                phase=getattr(p, 'phase', 2), stopped=bool(stop),
                wall_time=time.perf_counter()-self.start, comm_bytes=p.comm_bytes,
                primal_residual=float(np.linalg.norm(p.w-p.z)) if hasattr(p, 'z') else float('nan')))
        acc = np.mean([p.metrics[-1]['accuracy'] for p in ps]) if ready else float('nan')
        logger.info('observer', f'cycle={cycle} accuracy={acc:.5f} max_change={change.max():.3e} stable_windows={self.stable}')
        return stop


def run_additional(config, shards, method, cycles=None):
    cycles = cycles if cycles is not None else config['ROUNDS']
    start = time.perf_counter()
    CommonState.initializeRandom(config['SEED'])
    proto = ObservedSSL() if method == 'fedssl_amc' else ObservedFDR()
    # Keep the established protocol defaults: 64 components, C=1, ten SSL cycles;
    # FDR rho=1, eps_scale=1. Local solver work follows the shared configuration.
    Network.reset(config['NUM_WORKERS'], GeneralNode([IdleProtocol(), proto]))
    for i in range(Network.size()):
        p = Network.get(i).getProtocol(1)
        p.stopped, p.stop_cycle, p.stop_reason = False, None, None
    sim = CDSimulator(cycles, initializers=[
        FedAvgSimulation(config, shards)._topology(config['TOPOLOGY']),
        DataInitializer(1, shards, config['LAMBDA'], config['GOSSIP_K'],
                        local_steps=config.get('SDCA_LOCAL_STEPS'))],
        controls=[AdditionalEvaluator(config, shards, cycles, start)],
        activation=config['ACTIVATION'])
    stopped_at = sim.nextExperiment()
    from run_compare import _collect
    out = _collect([Network.get(i).getProtocol(1) for i in range(Network.size())],
                   method, stopped_at, cycles)
    out['engine'] = 'src.peersim_python.cdsim.CDSimulator'
    out['method_params'] = ({'d_enc': proto.d_enc, 'svm_C': proto.svm_C,
                             'encoder_cycles': proto.encoder_cycles,
                             'classifier_gossip': 'original feature space including intercept',
                             'aggregation_weight': 'fixed local sample count'}
                            if method == 'fedssl_amc' else
                            {'rho': proto.rho, 'eps_scale': proto.eps_scale,
                             'local_steps': config['SDCA_LOCAL_STEPS'],
                             'robust_surrogate': 'existing lambda + eps quadratic regularizer'})
    out['stop_rule'] = 'all peers relative parameter change < 1e-3 for 3 evaluation windows'
    out['elapsed_seconds'] = time.perf_counter()-start
    return out
