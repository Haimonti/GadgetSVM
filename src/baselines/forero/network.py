"""The paper's own experimental setup (Forero et al., JMLR 2010, Sec. 5) — no PeerSim.

Network. The paper uses "a randomly generated network with J nodes", reporting
only that it is connected, its algebraic connectivity and its average degree
(e.g. J = 30: 0.0448, 3.267); it also runs a ring and a fully connected network.
It does not name the generator, so `random` is the standard choice for this
(sensor-network) setting: a random geometric graph — J points uniform in the
unit square, linked when closer than r — with r the smallest radius at which the
seeded draw is connected (or FORERO_RADIUS). Algebraic connectivity and average
degree are logged so a run can be matched to the paper's numbers.

Partition. "The training set is equally partitioned across nodes", in the
paper's two extremes: `balanced` (every node holds the same number of examples
of each class) and `biased` (each node holds data of a single class where the
class sizes allow it). `given` keeps our shards as they are.

Communication. A node *broadcasts* its message once per iteration to its whole
one-hop neighbourhood (wireless sensor model): the paper counts "the number of
scalars communicated per node", e.g. 3J per iteration for J nodes in 2-D. So
`comm_bytes` charges one message per node per iteration (FORERO_COMM_MODEL
"broadcast"); "p2p" charges one copy per neighbour, as our gossip tenants do.
"""

import numpy as np


def random_geometric(J, seed, radius=None):
    rng = np.random.default_rng([seed, 5])
    pts = rng.random((J, 2))
    dist = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
    if radius is None:
        for r in np.unique(dist[np.triu_indices(J, 1)]):
            adj = _from_mask(dist <= r)
            if _connected(adj):
                return adj
        raise RuntimeError("could not build a connected random geometric graph")
    adj = _from_mask(dist <= radius)
    if not _connected(adj):
        raise ValueError(f"FORERO_RADIUS={radius} gives a disconnected graph for J={J}")
    return adj


def ring(J):
    return [sorted({(j - 1) % J, (j + 1) % J} - {j}) for j in range(J)]


def full(J):
    return [[i for i in range(J) if i != j] for j in range(J)]


def make_network(name, J, seed, radius=None):
    if name == "random":
        return random_geometric(J, seed, radius)
    if name == "ring":
        return ring(J)
    if name == "full":
        return full(J)
    raise ValueError(f"Unknown FORERO_NETWORK '{name}'. Choose: random | ring | full")


def _from_mask(mask):
    J = mask.shape[0]
    return [[i for i in range(J) if mask[j, i] and i != j] for j in range(J)]


def _connected(adj):
    seen, stack = {0}, [0]
    while stack:
        for i in adj[stack.pop()]:
            if i not in seen:
                seen.add(i)
                stack.append(i)
    return len(seen) == len(adj)


def algebraic_connectivity(adj):
    J = len(adj)
    Lap = np.zeros((J, J))
    for j, nbrs in enumerate(adj):
        Lap[j, j] = len(nbrs)
        for i in nbrs:
            Lap[j, i] = -1.0
    return float(np.sort(np.linalg.eigvalsh(Lap))[1]) if J > 1 else 0.0


def partition(X, y, J, mode, seed):
    """Row index blocks for J nodes (see module docstring)."""
    rng = np.random.default_rng([seed, 6])
    if mode == "balanced":
        blocks = [[] for _ in range(J)]
        for c in np.unique(y):
            idx = rng.permutation(np.flatnonzero(y == c))
            for j, part in enumerate(np.array_split(idx, J)):
                blocks[j].extend(part.tolist())
        return [np.sort(np.array(b, dtype=int)) for b in blocks]
    if mode == "biased":
        order = np.argsort(y, kind="stable")
        return [np.sort(b) for b in np.array_split(order, J)]
    raise ValueError(f"Unknown FORERO_PARTITION '{mode}'. Choose: balanced | biased | given")
