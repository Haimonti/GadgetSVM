"""Message compression operators Q(x) with exact byte accounting (learner-agnostic).

Ported from CHOCO-SGD's convex code (epfml/ChocoSGD `convex_code/logistic.py`,
`__quantize` + `qsgd_quantize`), so CHOCO uses its own operators verbatim and the
quantization ablation can apply the same operators to every other model's
message. Every operator returns `(decoded, nbytes)`: the vector the receiver
reconstructs and the exact size of its encoding.

Spec strings (CONFIG["CHOCO_COMPRESSOR"] / CONFIG["COMPRESSOR"]):

    full                 lossless, dense float64               8·d bytes
    lossless_sparse      lossless, (int32 index, float64) pairs, or dense if smaller
    top_k:<frac>         keep the k = ceil(frac·d) largest |x_i|        (biased)
    random_k:<frac>      keep k random coordinates                       (biased)
    random_k_unbiased:<frac>   same, rescaled by d/k                     (unbiased)
    qsgd:<s>             QSGD with s levels, CHOCO's biased scaling      (biased)
    qsgd_unbiased:<s>    QSGD with s levels                              (unbiased)

Sparse operators cost k·(4 + 8) bytes (index + value). QSGD costs one float64
norm plus d·(1 + ceil(log2(s + 1))) bits for sign + level, packed. "8-bit QSGD"
is s = 127, "4-bit" is s = 7.
"""

import math

import numpy as np


def _parse(spec):
    name, _, arg = spec.partition(":")
    return name, (float(arg) if arg else None)


def qsgd_quantize(x, s, is_biased, rng):
    """CHOCO's `qsgd_quantize`: stochastic rounding of |x|/||x|| onto s levels."""
    norm = np.sqrt(np.sum(np.square(x)))
    if norm == 0.0:
        return np.zeros_like(x)
    level_float = s * np.abs(x) / norm
    previous_level = np.floor(level_float)
    is_next_level = rng.random(x.shape) < (level_float - previous_level)
    new_level = previous_level + is_next_level
    scale = 1.0
    if is_biased:
        d = x.size
        scale = 1.0 / (min(d / s ** 2, math.sqrt(d) / s) + 1.0)
    return scale * np.sign(x) * norm * new_level / s


class Compressor:
    """Q(x) for one spec; `rng` (np.random.Generator) drives the randomised ones."""

    def __init__(self, spec="full", rng=None):
        self.spec = spec
        self.name, self.arg = _parse(spec)
        self.rng = rng if rng is not None else np.random.default_rng(0)
        if self.name not in ("full", "lossless_sparse", "top_k", "random_k",
                             "random_k_unbiased", "qsgd", "qsgd_unbiased"):
            raise ValueError(f"Unknown compressor '{spec}'")

    @property
    def lossless(self):
        return self.name in ("full", "lossless_sparse")

    def _k(self, d):
        return max(1, min(d, math.ceil(self.arg * d)))

    def __call__(self, x):
        x = np.asarray(x, dtype=np.float64)
        d = x.size
        if self.name == "full":
            return x.copy(), 8 * d
        if self.name == "lossless_sparse":
            return x.copy(), min(8 * d, 12 * int(np.count_nonzero(x)))
        if self.name == "top_k":
            k = self._k(d)
            q = np.zeros_like(x)
            idx = np.argsort(np.abs(x))[::-1][:k]
            q[idx] = x[idx]
            return q, 12 * k
        if self.name in ("random_k", "random_k_unbiased"):
            k = self._k(d)
            q = np.zeros_like(x)
            idx = self.rng.choice(d, size=k, replace=False)
            q[idx] = x[idx] * (d / k if self.name == "random_k_unbiased" else 1.0)
            return q, 12 * k
        s = int(self.arg)
        q = qsgd_quantize(x, s, self.name == "qsgd", self.rng)
        bits = d * (1 + math.ceil(math.log2(s + 1)))
        return q, 8 + math.ceil(bits / 8)

    def nbytes_of(self, decoded):
        """Encoded size of an already-decoded vector (for relaying it unchanged)."""
        d = decoded.size
        if self.name == "full":
            return 8 * d
        if self.name == "lossless_sparse":
            return min(8 * d, 12 * int(np.count_nonzero(decoded)))
        if self.name in ("top_k", "random_k", "random_k_unbiased"):
            return 12 * self._k(d)
        s = int(self.arg)
        return 8 + math.ceil(d * (1 + math.ceil(math.log2(s + 1))) / 8)
