"""Block boosting — Fed-KSVM Algorithm 1 (the device's memory-efficient local solver).

Device m solves its local primal (8) for w_m given the fixed w_bar_m:

    min_w  C sum_{i in I_m} max{0, 1 - y_i (w_bar + w)^T a~_i} + 1/2 ||w||^2

whose dual is the CoCoA subproblem (5). The D random-feature dimensions are cut
into L equal blocks; each of Q iterations picks a block l at random, fixes the
other blocks, forms

    B_i = y_i (w_bar^T a~_i + sum_{j != l} w[j]^T a~_i[j])                    (10)

and solves the block dual (11) by SDCA,

    max_alpha  -1/2 ||A[l] alpha||^2 + sum_i (1 - B_i) alpha_i,  0 <= alpha_i <= C,
    A[l] = [y_i a~_i[l]]_i,

then sets w[l] = A[l] alpha* (only block l of the features is touched — the
memory saving the paper is about). Each block keeps its own dual vector between
visits (warm start). Theorem 1: the result converges to the optimum of (8).

SDCA coordinate step on (11):  alpha_i <- clip(alpha_i + (1 - B_i - y_i a~_i[l]^T v) / ||a~_i[l]||^2, 0, C),
with v = A[l] alpha maintained incrementally.
"""

import numpy as np


class BlockBoosting:

    def __init__(self, F, y, C, L, Q, H, rng):
        """F: device's random features a~ (n_m x D); H = SDCA passes per block solve."""
        self.F, self.y, self.C = F, np.asarray(y, dtype=np.float64), float(C)
        self.n, self.D = F.shape
        self.blocks = np.array_split(np.arange(self.D), int(L))
        self.Q, self.H, self.rng = int(Q), float(H), rng
        self.duals = [np.zeros(self.n) for _ in self.blocks]
        self.norms = [np.einsum("ij,ij->i", F[:, blk], F[:, blk]) for blk in self.blocks]

    def solve(self, w_bar, w):
        """Q block iterations from the current local vector w; returns the new w."""
        w = w.copy()
        margin_all = self.F @ (w_bar + w)                    # (w_bar + w)^T a~_i
        for _ in range(self.Q):
            l = int(self.rng.integers(len(self.blocks)))
            blk = self.blocks[l]
            Fl = self.F[:, blk]
            B = self.y * (margin_all - Fl @ w[blk])          # (10): everything but block l
            alpha = self.duals[l]
            v = (self.y * alpha) @ Fl                          # A[l] alpha
            nrm = self.norms[l]
            for i in self.rng.integers(0, self.n, size=max(1, int(self.H * self.n))):
                if nrm[i] <= 0.0:
                    continue
                g = 1.0 - B[i] - self.y[i] * float(Fl[i] @ v)
                new = min(self.C, max(0.0, alpha[i] + g / nrm[i]))
                if new != alpha[i]:
                    v += (new - alpha[i]) * self.y[i] * Fl[i]
                    alpha[i] = new
            margin_all += Fl @ (v - w[blk])
            w[blk] = v                                         # w[l] = A[l] alpha*
        return w
