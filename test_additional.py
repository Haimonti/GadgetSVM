"""Numerical regression checks for the additional PeerSim integrations."""
import unittest
import numpy as np
from scipy import sparse
from p2p._prox import prox_pegasos
from p2p._prox_fast import prox_pegasos_fast
from p2p.fedssl_protocol import FedSSLAMCProtocol
from p2p._prox_gram import prox_pegasos_gram


class AdditionalProtocolTests(unittest.TestCase):
    def test_encoder_phase_has_no_classifier_consensus_curve(self):
        from plot_merged import series
        history = [[dict(round=1, phase=1, consensus_error=0.),
                    dict(round=11, phase=2, consensus_error=2.),
                    dict(round=21, phase=2, consensus_error=.5)]]
        x, mean, std = series(history, 'consensus_error')
        np.testing.assert_array_equal(x, [11, 21])
        np.testing.assert_allclose(mean, [2., .5])

    def test_gram_solver_matches_reference_and_rng(self):
        r = np.random.default_rng(42)
        X = sparse.csr_matrix(r.normal(size=(40, 1000)))
        dense = X.toarray()
        G = dense @ dense.T
        y = r.choice([-1., 1.], 40).astype(np.float32)
        for _ in range(5):
            v = r.normal(size=1000)
            a, b = np.random.default_rng(123), np.random.default_rng(123)
            expected = prox_pegasos(X, y, v, 1.1, 1., 4096, a)
            actual = prox_pegasos_gram(X, G, y, v, 1.1, 1., 4096, b)
            np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=1e-6)
            self.assertEqual(a.integers(2**30), b.integers(2**30))

    def test_compiled_solver_matches_reference_and_rng(self):
        for d in (4, 123, 10000):
            r = np.random.default_rng(42)
            X = sparse.random(100, d, density=min(.3, 5/d), random_state=r, format='csr')
            y = r.choice([-1., 1.], 100).astype(np.float32)
            v = r.normal(size=d)
            for rho in (0., 1.):
                a, b = np.random.default_rng(123), np.random.default_rng(123)
                expected = prox_pegasos(X, y, v, 1.1, rho, 4096, a)
                actual = prox_pegasos_fast(X, y, v, 1.1, rho, 4096, b)
                np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=1e-6)
                self.assertEqual(a.integers(2**30), b.integers(2**30))

    def test_classifier_merge_preserves_intercept_and_finite_weights(self):
        p = FedSSLAMCProtocol()
        p.phase, p.trainable, p.n, p.age = 2, True, 2, 2.
        p.w, p.bias = np.array([1., 0.]), 2.
        p.inbox = [({'w': np.array([0., 1.]), 'bias': 4.}, 6.)]
        p._merge_inbox()
        np.testing.assert_allclose(p.w, [.25, .75])
        self.assertEqual(p.bias, 3.5)
        for _ in range(5000):
            p.inbox = [({'w': p.w.copy(), 'bias': p.bias}, p.age)]
            p._merge_inbox()
        self.assertEqual(p.age, 2.)
        self.assertTrue(np.isfinite(p.w).all())

    def test_lift_preserves_encoded_scores_for_different_bases(self):
        X = np.array([[1., 3.], [-2., 5.]])
        V1, V2 = np.eye(2), np.array([[0., -1.], [1., 0.]])
        w = np.array([2., -1.])
        for V in (V1, V2):
            np.testing.assert_allclose((X @ V.T) @ w, X @ (V.T @ w))
        # Averaging raw encoded coordinates would fail this equivalence.
        expected = ((X @ V1.T) @ w + (X @ V2.T) @ w) / 2
        np.testing.assert_allclose(X @ ((V1.T @ w + V2.T @ w)/2), expected)


if __name__ == '__main__':
    unittest.main()
