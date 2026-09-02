"""peersim.core.Network — the one global Node[] array.

The whole "network" is a single Python list living in this class. Every node is
an entry; only the first size() entries are live. This is the clearest sign
that PeerSim is a single-process simulator: there is exactly one array, in one
process, that *is* the network.
"""

from src.peersim_python.core.common_state import CommonState


class Network:
    """peersim.core.Network — the one global Node[] array."""
    node: list = []
    len = 0
    prototype = None

    @classmethod
    def reset(cls, size, prototype):
        """Build the network by cloning the prototype `size` times."""
        cls.prototype = prototype
        prototype.setIndex(-1)
        cls.node = []
        for _ in range(size):
            cls.node.append(prototype.clone())
        cls.len = size
        for i, nd in enumerate(cls.node):
            nd.setIndex(i)

    @classmethod
    def size(cls):
        return cls.len

    @classmethod
    def get(cls, index):
        return cls.node[index]

    @classmethod
    def add(cls, n):
        cls.node.append(n)
        n.setIndex(cls.len)
        cls.len += 1

    @classmethod
    def shuffle(cls):
        """Fisher-Yates over the live nodes using the shared RNG."""
        for i in range(cls.len, 1, -1):
            cls.swap(i - 1, CommonState.r.randint(0, i - 1))

    @classmethod
    def swap(cls, i, j):
        cls.node[i], cls.node[j] = cls.node[j], cls.node[i]
        cls.node[i].setIndex(i)
        cls.node[j].setIndex(j)
