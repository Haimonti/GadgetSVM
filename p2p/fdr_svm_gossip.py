"""Retired one-hop quadratic approximation; use the exact paper transport."""


class FDRSVMGossipProtocol:
    def __init__(self, *args, **kwargs):
        raise RuntimeError('Quadratic FDR gossip is retired. Use paper_reference.fdr.train '
                           'with adjacency for exact Eq. (9) P2P aggregation.')
