"""peersim.core.IdleProtocol — the reference Linkable implementation.

Stores a node's neighbour list and does nothing else. This is the protocol the
Wire* controls populate and the one gossip protocols read their peers from.
"""

from src.network_layer.peersim_python.core import Protocol, Linkable


class IdleProtocol(Protocol, Linkable):
    def __init__(self, capacity: int = 10):
        self.neighbors: list = []

    def clone(self):
        c = IdleProtocol()
        # Copy neighbour *references* (as PeerSim's arraycopy does). At prototype
        # clone time the list is empty; Wire* fills it afterwards per node.
        c.neighbors = list(self.neighbors)
        return c

    def contains(self, n):
        return any(x is n for x in self.neighbors)

    def addNeighbor(self, neighbour):
        if self.contains(neighbour):
            return False
        self.neighbors.append(neighbour)
        return True

    def getNeighbor(self, i):
        return self.neighbors[i]

    def degree(self):
        return len(self.neighbors)

    def pack(self):
        pass

    def onKill(self):
        self.neighbors = []
