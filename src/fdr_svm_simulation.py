"""Retired quadratic/one-hop FDR assembly, preserved under legacy_approximations.

Paper Eq. (9) is now in paper_reference.fdr; run_paper_reference.py provides
server and exact P2P transport without reusing the invalid local objective.
"""
from src.fedavg_simulation import FedAvgSimulation


class FDRSVMSimulation(FedAvgSimulation):
    def build(self):
        raise RuntimeError('The former FDR quadratic/one-hop approximation is retired. '
                           'Use run_paper_reference.py fdr for Eq. (9) and exact P2P aggregation.')
