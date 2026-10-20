import unittest
import time

import numpy as np
import scipy.sparse as sp

from src.train_convergence import ObjectivePatience, WindowStability, TrainConvergenceEvaluator
from legacy_approximations.ssl_svd_gossip import FedSSLGossipProtocol, FedSSLTableAggregator
from src.peersim_python.core import CommonState, GeneralNode, Network
from src.peersim_python.idle_protocol import IdleProtocol
from src.peersim_python.observers import DataInitializer
from src.peersim_python.dynamics import WireFull
from src.peersim_python.cdsim import CDSimulator


class PatienceTests(unittest.TestCase):
    def test_fifty_rounds_means_fifty_training_rounds_after_smoothing(self):
        rule = ObjectivePatience(2)
        for round_number in range(1, 60):
            self.assertFalse(rule.update([.4, .5], round_number))
        self.assertTrue(rule.update([.4, .5], 60))

    def test_improving_peer_prevents_global_stop(self):
        rule = ObjectivePatience(2, window=1, patience=5)
        for r in range(1, 30):
            self.assertFalse(rule.update([.5, 1 / r], r))

    def test_deterioration_is_not_convergence(self):
        rule = ObjectivePatience(1, window=1, patience=5)
        self.assertFalse(rule.update([.2], 1))
        for r in range(2, 30):
            self.assertFalse(rule.update([.5 + r / 10], r))

    def test_incomplete_or_nonfinite_history_cannot_trigger_stop(self):
        rule = ObjectivePatience(1, window=2, patience=5)
        for r in range(1, 20):
            self.assertFalse(rule.update([.2], r, ready=False))
        self.assertFalse(rule.update([np.nan], 20))
        for r in range(21, 27):
            self.assertFalse(rule.update([.2], r))
        self.assertTrue(rule.update([.2], 27))


class WindowStabilityTests(unittest.TestCase):
    def test_two_stable_comparisons_require_three_full_windows(self):
        rule = WindowStability(2)
        for r in range(1, 150):
            self.assertFalse(rule.update([.35, .36], r))
        self.assertTrue(rule.update([.35, .36], 150))

    def test_stationary_noise_does_not_require_reaching_historical_minimum(self):
        rule = WindowStability(2, window=10)
        for r in range(1, 30):
            noise = (-1) ** r * .001
            self.assertFalse(rule.update([.35 + noise, .36 - noise], r))
        self.assertTrue(rule.update([.351, .359], 30))

    def test_one_moving_peer_prevents_stop_in_either_direction(self):
        for sign in (-1, 1):
            rule = WindowStability(2, window=5)
            for r in range(1, 60):
                self.assertFalse(rule.update([.35, 2 + sign * r * .01], r))

    def test_nonfinite_and_unready_inputs_reset_windows(self):
        rule = WindowStability(1, window=2)
        for r in range(1, 5):
            self.assertFalse(rule.update([.35], r))
        self.assertFalse(rule.update([np.nan], 5))
        self.assertFalse(rule.update([.35], 6, ready=False))
        for r in range(7, 12):
            self.assertFalse(rule.update([.35], r))
        self.assertTrue(rule.update([.35], 12))


class LegacySVDIntegrationTests(unittest.TestCase):
    def test_common_basis_and_final_only_test_evaluation(self):
        rng = np.random.default_rng(7)
        X = sp.csr_matrix(rng.normal(size=(120, 8)))
        y = np.where(X.toarray()[:, 0] >= 0, 1., -1.)
        shards = [dict(X_csr=X[ix], y=y[ix], X_test=X[:10], y_test=y[:10],
                       n_local=len(ix)) for ix in np.array_split(np.arange(120), 4)]
        CommonState.initializeRandom(42)
        Network.reset(4, GeneralNode([IdleProtocol(),
                      FedSSLGossipProtocol(gossip_k=3, d_enc=4, seed=42)]))
        init = DataInitializer(1, shards, .01, 3)
        init.execute()
        protos = [Network.get(i).getProtocol(1) for i in range(4)]
        for p in protos:
            p.accuracy = lambda: (_ for _ in ()).throw(AssertionError('Test evaluated during training'))
        observer = TrainConvergenceEvaluator(1, dict(NUM_WORKERS=4,
            CONVERGENCE_PATIENCE=5, CONVERGENCE_WINDOW=2, EVAL_EVERY=2), 80,
            time.perf_counter())
        sim = CDSimulator(80, [WireFull(0, undir=True)], [observer])
        rounds = sim.nextExperiment()
        self.assertLess(rounds, 80)
        self.assertTrue(observer.converged)
        for p in protos:
            self.assertTrue(p.convergence_ready)
            np.testing.assert_allclose(p.V, protos[0].V, atol=1e-12)
            np.testing.assert_allclose(p.w, protos[0].w, atol=1e-12)
            self.assertGreater(p.comm_bytes, 0)
            self.assertTrue(all('accuracy' not in row for row in p.metrics))
        self.assertTrue(np.isfinite(FedSSLGossipProtocol.accuracy(protos[0])))

    def test_message_merge_is_idempotent(self):
        current = {'encoders': {0: (np.eye(2), 3)}, 'classifiers': {}}
        message = {'encoders': {1: (np.eye(2), 4)}, 'classifiers': {}}
        agg = FedSSLTableAggregator()
        first = agg.aggregate(current, [message])
        twice = agg.aggregate(first, [message, message])
        self.assertEqual(len(twice['encoders']), 2)
        self.assertEqual(sum(n for _, n in twice['encoders'].values()), 7)


class AllMethodObserverTests(unittest.TestCase):
    def test_training_observer_handles_all_existing_protocols_without_test_calls(self):
        from src.peersim_python.simulation import Simulation
        from src.bdsvm_simulation import BDSVMSimulation
        from src.fedavg_simulation import FedAvgSimulation
        from src.cocoa_simulation import CoCoASimulation
        rng = np.random.default_rng(17)
        X = sp.csr_matrix(rng.normal(size=(80, 8)))
        y = np.tile([-1., 1.], 40)
        shards = [dict(X_csr=X[ix], y=y[ix], X_test=X[:8], y_test=y[:8],
                       n_local=len(ix)) for ix in np.array_split(np.arange(80), 4)]
        config = dict(NUM_WORKERS=4, SEED=42, LAMBDA=.01, GOSSIP_K=3,
                      TOPOLOGY='full', SDCA_LOCAL_STEPS=8, SDCA_STEP_SCALE=1.,
                      FDR_LOCAL_STEPS=5, GAP_THRESHOLD=1e-4, EVAL_EVERY=1,
                      STOP_ON_THRESHOLD=True, ACTIVATION='shuffle', WARM_START=False)
        factories = [lambda: Simulation(config, shards),
            lambda: FedAvgSimulation(config, shards),
            lambda: CoCoASimulation(config, shards, variant='cocoa'),
            lambda: CoCoASimulation(config, shards, variant='cocoa_plus'),
            lambda: BDSVMSimulation(config, shards, params=dict(
                kernel='linear', P=4, C=1., rho=.5, preimage='unit', ey_floor=.01))]
        for factory in factories:
            sim = factory().build().sim
            for initializer in sim.initializers:
                initializer.execute()
            sim.initializers = []
            for i in range(4):
                proto = Network.get(i).getProtocol(1)
                proto.accuracy = lambda: (_ for _ in ()).throw(AssertionError('test call'))
            observer = TrainConvergenceEvaluator(1, config, 3, time.perf_counter())
            sim.controls = [observer]
            sim.cycles = 3
            self.assertEqual(sim.nextExperiment(), 3)
            self.assertFalse(observer.converged)
            self.assertTrue(np.isfinite(observer.history[-1]['train_hinge']))
            self.assertEqual(observer.history[-1]['comm_bytes'],
                sum(Network.get(i).getProtocol(1).comm_bytes for i in range(4)))


if __name__ == '__main__':
    unittest.main()
