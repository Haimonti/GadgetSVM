"""Gossip mixing matrix W built from the wired overlay (learner-agnostic).

Decentralised averaging baselines (COLA, CHOCO-SGD) mix neighbour values with a
symmetric, doubly-stochastic W supported on the graph. We use Metropolis-Hastings
weights,

    W_ij = 1 / (1 + max(deg_i, deg_j))   for each edge (i, j),
    W_ii = 1 - sum_{j != i} W_ij,

which reproduce COLA's `cola/graph.py` weights exactly on its topologies: ring
(1/3), complete (1/n), c-connected cycle (1/(2c+1)) and the 2-D grid (which COLA
already defines with this rule). `beta` is the second-largest eigenvalue modulus
of W (1 - spectral gap), the connectivity quantity in COLA's and CHOCO's rates.

    adj = adjacency_from_network(LINKABLE_PID)
    W, beta = metropolis_weights(adj)
"""

import numpy as np

from src.peersim_python.core import Network


def adjacency_from_network(pid):
    """Neighbour index lists of the current overlay, read from Linkable `pid`."""
    adj = []
    for i in range(Network.size()):
        link = Network.get(i).getProtocol(pid)
        adj.append(sorted(link.getNeighbor(k).getIndex() for k in range(link.degree())))
    return adj


def metropolis_weights(adj):
    """Return (W, beta) for an undirected adjacency list."""
    n = len(adj)
    deg = np.array([len(a) for a in adj])
    W = np.zeros((n, n))
    for i, nbrs in enumerate(adj):
        for j in nbrs:
            if i not in adj[j]:
                raise ValueError(f"adjacency is not symmetric at edge ({i}, {j})")
            W[i, j] = 1.0 / (1.0 + max(deg[i], deg[j]))
    W[np.diag_indices(n)] = 1.0 - W.sum(axis=1)
    assert np.allclose(W, W.T) and np.allclose(W.sum(axis=1), 1.0)
    eig = np.sort(np.abs(np.linalg.eigvalsh(W)))[::-1]
    beta = float(eig[1]) if n > 1 else 0.0
    return W, beta
