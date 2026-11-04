"""Retired quadratic-penalty FDR approximation (old engine layout).

Its local objective drops the paper's label-flip epigraph constraints and is
not a Wasserstein FDR-SVM reproduction. Current code: paper_reference.fdr.
"""
import numpy as np

from p2p.admm_protocol import ConsensusADMMProtocol


class FDRSVMProtocol(ConsensusADMMProtocol):

    def __init__(self):
        super().__init__()
        self.eps_scale = 1.0

    def _clone_extra(self, c):
        c.eps_scale = self.eps_scale

    def _setup_extra(self):
        # eps_i = eps_scale / sqrt(n_i): less local data -> wider ambiguity ball
        # -> stronger regularisation. Purely local, unlike the server version.
        self.eps = self.eps_scale / max(np.sqrt(self.n), 1.0)

    def _solve_local(self, v):
        A = self.lambda_reg + self.eps + self.rho
        return self._prox(self.X, self.y, v, A)
