"""Engine-free math of MoM-NDSVM (Forero, Cano & Giannakis, JMLR 2010, Sec. 4).

Everything node ``j`` computes in Proposition 3 / Algorithm 3, written against the
paper's own definitions so each line can be checked against an equation number.
The PeerSim tenant (``ndsvm_protocol.py``) and the validator
(``validate_ndsvm.py``) both drive this module, so the numbers the simulator
produces are the numbers the validator certifies.

Notation (paper, Prop. 2/3), for node ``j`` with ``|B_j|`` neighbours::

    c0          = 2 eta |B_j|
    U~_j        = I_L + c0 K(G,G)                       (Prop. 2)
    K~(Z, Z')   = c0 K(Z,G) U~_j^{-1} K(G,Z')            (Prop. 3)
    f~_j(t)     = 2 alpha_j(t) - eta sum_{i in B_j} [w~_j(t) + w~_i(t)]
    h_j(t)      = 2 beta_j(t)  - eta sum_{i in B_j} [b_j(t)  + b_i(t)]

``G`` stands for the anchor set Gamma = [chi_1 .. chi_L]^T.

Three typesetting errors in the paper are corrected here (derivation in the
package docstring, numerical proof in ``validate_ndsvm.py``):

* **E1** (35)/(82): the bracket ``(f~^T (K(G,X) - K~(G,X)) + h 1^T / c0) Y lambda``
  enters the dual with a **plus** sign, not the printed minus. Only then is
  ``lambda_j(t+1)`` the KKT multiplier of the iterate (36)/(37) builds from it.
* **E2** (32)-(34)/(81): the discriminant of the iterate ``(w_j(t+1), b_j(t+1))``
  pairs ``lambda_j(t+1)`` with ``f~_j(t)`` and ``h_j(t)``.
* **E3** (79): the sum inside ``b_j(t+1)`` runs over the biases ``b``, i.e. (37).
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular


class LocalProblem:
    """The parts of node j's Prop. 3 iteration that are fixed for the whole run.

    Built once from the node's data ``X`` (N x p), labels ``y`` in {-1,+1}, the
    common anchors ``Gamma`` (L x p) and their Gram ``K_GG`` (identical on every
    node), the kernel, the ADMM penalty ``eta``, the node degree ``|B_j|`` and the
    box bound ``JC``. Only ``f~`` and ``h`` change between iterations, so the
    (N x N) quadratic of (35) is assembled exactly once.
    """

    def __init__(self, X, y, Gamma, K_GG, kernel, eta, degree, JC):
        if degree < 1:
            raise ValueError("MoM-NDSVM needs every node to have >= 1 neighbour")
        self.y = np.asarray(y, dtype=np.float64)
        self.N = self.y.shape[0]
        self.L = Gamma.shape[0]
        self.eta = float(eta)
        self.degree = int(degree)
        self.JC = float(JC)
        self.c0 = 2.0 * self.eta * self.degree                 # 2 eta |B_j|

        self.K_XG = kernel(X, Gamma)                           # (29)  N x L
        self.K_GG = np.asarray(K_GG, dtype=np.float64)         # (30)  L x L
        U = np.eye(self.L) + self.c0 * self.K_GG               # U~_j
        self._chol = cho_factor(U, lower=True)

        K_GX = self.K_XG.T
        Uinv_KGX = cho_solve(self._chol, K_GX)                 # U~^{-1} K(G,X)
        Uinv_KGG = cho_solve(self._chol, self.K_GG)            # U~^{-1} K(G,G)
        Kt_GX = self.c0 * self.K_GG @ Uinv_KGX                 # K~(G,X)
        Kt_GG = self.c0 * self.K_GG @ Uinv_KGG                 # K~(G,G)
        self.A_GX = K_GX - Kt_GX       # K(G,X) - K~(G,X); its transpose is K(X,G) - K~(X,G)
        self.B_GG = self.K_GG - Kt_GG  # K(G,G) - K~(G,G)

        # (35) quadratic: Y (K(X,X) - K~(X,X) + 1 1^T / c0) Y, assembled in place.
        # K~(X,X) = c0 K(X,G) U~^{-1} K(G,X) = c0 M^T M with M = C^{-1} K(G,X),
        # C the lower Cholesky factor of U~, so K(X,X) is never kept.
        C = np.linalg.cholesky(U)
        M = solve_triangular(C, K_GX, lower=True)              # L x N
        Q = kernel(X, X)                                       # (28)
        Q -= self.c0 * (M.T @ M)
        Q += 1.0 / self.c0
        Q *= self.y[:, None]
        Q *= self.y[None, :]
        Q += Q.T
        Q *= 0.5                                               # exact symmetry
        self.Q = Q

    # ---- one ADMM iteration --------------------------------------------------
    def lin_term(self, f, h):
        """Linear coefficient ``p`` of (35), with the E1-corrected sign:
        ``p = 1 + Y [ (K(X,G) - K~(X,G)) f~ + (h / c0) 1 ]``."""
        return 1.0 + self.y * (self.A_GX.T @ f + h / self.c0)

    def iterate(self, lam, f, h):
        """(32), (33), (36), (37) for ``lambda = lambda_j(t+1)``, ``f = f~_j(t)``,
        ``h = h_j(t)`` (E2 pairing). Returns ``(a, c, w_tilde, b)``: the readout
        coefficients and bias of ``g_j`` at ``t+1``, and the vector node j
        broadcasts."""
        a = self.y * lam                                       # (32)
        w_tilde = self.A_GX @ a - self.B_GG @ f                # (36)
        b = (a.sum() - h) / self.c0                            # (37)
        c = self.c0 * cho_solve(self._chol, self.K_GG @ f - self.K_XG.T @ a) - f   # (33)
        return a, c, w_tilde, b

    # ---- quantities derived from an iterate ----------------------------------
    def kxx_times(self, Qlam, a):
        """``K(X,X) a`` recovered from the cached ``Q lambda`` without K(X,X):
        ``Y Q lambda = (K - K~ + 1 1^T / c0) a``."""
        u = self.K_XG.T @ a
        return (self.y * Qlam
                + self.c0 * (self.K_XG @ cho_solve(self._chol, u))
                - a.sum() / self.c0)

    def own_scores(self, Qlam, a, c, b):
        """``g_j`` on the node's own training points."""
        return self.kxx_times(Qlam, a) + self.K_XG @ c + b

    def norm_sq(self, lam, Qlam, a, c):
        """``||w_j||^2 = a^T K(X,X) a + 2 a^T K(X,G) c + c^T K(G,G) c`` with
        ``a^T K(X,X) a = lam^T Q lam + c0 u^T U~^{-1} u - (1^T a)^2 / c0``,
        ``u = K(G,X) a``."""
        u = self.K_XG.T @ a
        aKa = float(lam @ Qlam + self.c0 * (u @ cho_solve(self._chol, u))
                    - a.sum() ** 2 / self.c0)
        return aKa + 2.0 * float(a @ (self.K_XG @ c)) + float(c @ self.K_GG @ c)

    def local_gap(self, lam, Qlam, f, h):
        """Primal-dual gap of node j's subproblem (25) at the given multipliers.

        Primal: ``1/2||w||^2 + (c0/2)||Gw||^2 + f~^T Gw + (c0/2) b^2 + h b
        + JC sum hinge`` (the constant of the augmented term dropped). Dual:
        ``-1/2 lam^T Q lam + p^T lam - 1/2 f~^T (K(G,G) - K~(G,G)) f~ - h^2/(2 c0)``.
        Zero (to solver tolerance) exactly when lambda solves (35) *and* the readout
        follows E2; the printed-sign variant leaves a large gap.
        """
        a, c, w_tilde, b = self.iterate(lam, f, h)
        g_own = self.own_scores(Qlam, a, c, b)
        hinge = np.maximum(0.0, 1.0 - self.y * g_own).sum()
        primal = (0.5 * self.norm_sq(lam, Qlam, a, c)
                  + 0.5 * self.c0 * float(w_tilde @ w_tilde) + float(f @ w_tilde)
                  + 0.5 * self.c0 * b * b + h * b + self.JC * hinge)
        dual = (-0.5 * float(lam @ Qlam) + float(self.lin_term(f, h) @ lam)
                - 0.5 * float(f @ (self.B_GG @ f)) - h * h / (2.0 * self.c0))
        return primal - dual


def f_tilde(alpha, w_own, w_neighbours, eta):
    """``f~_j(t) = 2 alpha_j - eta sum_{i in B_j} [w~_j + w~_i]``."""
    return 2.0 * alpha - eta * (len(w_neighbours) * w_own + np.sum(w_neighbours, axis=0))


def h_scalar(beta, b_own, b_neighbours, eta):
    """``h_j(t) = 2 beta_j - eta sum_{i in B_j} [b_j + b_i]``."""
    return 2.0 * beta - eta * (len(b_neighbours) * b_own + float(np.sum(b_neighbours)))


def dual_ascent(alpha, beta, w_own, b_own, w_neighbours, b_neighbours, eta):
    """(38)/(39): ``alpha += eta/2 sum (w~_j - w~_i)``, ``beta += eta/2 sum (b_j - b_i)``,
    all at ``t+1``."""
    k = len(w_neighbours)
    alpha = alpha + 0.5 * eta * (k * w_own - np.sum(w_neighbours, axis=0))
    beta = beta + 0.5 * eta * (k * b_own - float(np.sum(b_neighbours)))
    return alpha, beta
