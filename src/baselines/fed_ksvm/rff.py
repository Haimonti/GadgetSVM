"""Random Fourier features (Rahimi & Recht) — Fed-KSVM Sec. III-B.

For the Gaussian kernel exp(-gamma ||x - x'||^2) the spectral density is
N(0, 2 gamma I). Draw D frequencies w_d and phases b_d ~ U[0, 2pi) from a seed
shared by every device, and map

    a(x) = [sqrt(2) cos(w_1^T x + b_1), ..., sqrt(2) cos(w_D^T x + b_D)]

so E[(1/D) a(x)^T a(x')] = K(x, x'). The model is f(x) = (1/sqrt(D)) w^T a(x), so
the features used everywhere below are a~(x) = a(x) / sqrt(D).
"""

import numpy as np


class RandomFourierFeatures:

    def __init__(self, n_features, D, gamma, seed):
        rng = np.random.default_rng([seed, 9])
        self.D = int(D)
        self.W = rng.normal(0.0, np.sqrt(2.0 * gamma), size=(n_features, self.D))
        self.b = rng.uniform(0.0, 2.0 * np.pi, size=self.D)

    def transform(self, X, chunk=4096):
        """a~(X) = sqrt(2/D) cos(X W + b), dense (n, D)."""
        out = np.empty((X.shape[0], self.D))
        for s in range(0, X.shape[0], chunk):
            Z = X[s:s + chunk] @ self.W
            out[s:s + chunk] = np.sqrt(2.0 / self.D) * np.cos(np.asarray(Z) + self.b)
        return out
