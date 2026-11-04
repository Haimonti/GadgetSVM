"""Common anchor set Gamma = {chi_l} — the paper's Sec. 4.1 "random design".

    "MoM-NDSVM starts by performing a consensus step on the entry-wise maxima and
    minima of the local training vectors {x_jn}. The set {chi_l} is formed by drawing
    elements chi_l randomly from a uniform p-dimensional distribution with extreme
    points per entry given by x_k^min and x_k^max. To agree on the set, all nodes in
    the network are assumed to share a common seed."  (Forero et al., Sec. 4.1)

Global max/min consensus is exact after at most diameter(G) synchronous rounds
(the paper's footnote 3), so that is exactly how many rounds are run. Every
message a node sends (its current min and max vectors, ``2p`` float64, to each
neighbour) is charged to that node. The uniform draw needs no messages: the same
bounds and the same seed give the same Gamma on every node.
"""

from collections import deque

import numpy as np


def adjacency(linkable_protocols):
    """Neighbour index lists read from each node's Linkable (IdleProtocol)."""
    return [[peer.getIndex() for peer in link.neighbors] for link in linkable_protocols]


def check_graph(adj):
    """Assert what Algorithm 3 needs (undirected, connected, every |B_j| >= 1) and
    return the diameter, which bounds the extremes consensus."""
    n = len(adj)
    for j, nbrs in enumerate(adj):
        if not nbrs:
            raise ValueError(f"node {j} has no neighbours; MoM-NDSVM needs |B_j| >= 1")
        if j in nbrs or len(set(nbrs)) != len(nbrs):
            raise ValueError(f"node {j} has a self-loop or a duplicate edge")
        for i in nbrs:
            if j not in adj[i]:
                raise ValueError(f"edge {j}->{i} has no reverse; the graph must be undirected")
    diameter = 0
    for s in range(n):
        dist = [-1] * n
        dist[s] = 0
        queue = deque([s])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[v] < 0:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        if min(dist) < 0:
            raise ValueError("the graph is disconnected; Lemma 1 requires a connected graph")
        diameter = max(diameter, max(dist))
    return diameter


def local_extremes(X):
    """Per-feature min and max of one node's (dense) training block."""
    return X.min(axis=0), X.max(axis=0)


def flood_extremes(adj, mins, maxs, rounds):
    """Synchronous max/min consensus: each round every node sends its current
    (min, max) to every neighbour, then keeps the entry-wise min/max of what it holds
    and received. Returns per-node results and per-node bytes sent."""
    mins = [m.astype(np.float64, copy=True) for m in mins]
    maxs = [m.astype(np.float64, copy=True) for m in maxs]
    p = mins[0].shape[0]
    sent = [0] * len(adj)
    for _ in range(rounds):
        out = [(mn.copy(), mx.copy()) for mn, mx in zip(mins, maxs)]   # messages of this round
        for j, nbrs in enumerate(adj):
            sent[j] += len(nbrs) * 2 * p * 8
            for i in nbrs:
                np.minimum(mins[j], out[i][0], out=mins[j])
                np.maximum(maxs[j], out[i][1], out=maxs[j])
    return mins, maxs, sent


def draw_anchors(x_min, x_max, L, seed):
    """Gamma: L points uniform in the box [x_min, x_max] from the shared seed."""
    rng = np.random.default_rng(seed)
    return x_min + (x_max - x_min) * rng.random((int(L), x_min.shape[0]))
