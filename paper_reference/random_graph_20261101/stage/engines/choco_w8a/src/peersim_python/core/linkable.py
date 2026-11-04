"""peersim.core.Linkable — a node's neighbour view (the overlay edges).

Behaves as an ordered set of neighbours: unique elements, random access, no
removal. Gossip protocols pick a peer through this interface. This is the
*topology* layer — it answers "who are my neighbours?" and nothing else; the
actual message-passing lives in the protocols that use it.
"""


class Linkable:
    """peersim.core.Linkable — a node's neighbour view (the overlay edges)."""
    def degree(self):
        raise NotImplementedError

    def getNeighbor(self, i):
        raise NotImplementedError

    def addNeighbor(self, neighbour):
        raise NotImplementedError

    def contains(self, neighbor):
        raise NotImplementedError

    def pack(self):
        pass

    def onKill(self):
        pass
