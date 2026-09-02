"""P2P FDR-SVM — decentralised counterpart of `methods/fdr_svm.py`.

Distributionally robust SVM over per-node Wasserstein balls. The robust local
objective is unchanged: by Kantorovich duality the worst case over an eps_i-ball
adds eps_i*||w||, so for the L2 case the local problem is ordinary SVM with
regularisation (lambda + eps_i).

This method decentralises more cleanly than it federates. In the server version
eps_k = eps_scale / sqrt(n_k) is derived from client sizes the server collects;
here every node already knows its own n_i, so eps_i is computed locally at
set-up and never communicated at all. The consensus machinery is inherited whole
from protocols/admm_protocol.py — structurally identical to BDSVM, which is why
the two server implementations share a shape too.
"""
import numpy as np

from protocols.admm_protocol import ConsensusADMMProtocol


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
