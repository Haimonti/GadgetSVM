from src.peersim_python.core import Control, CommonState
from src.peersim_python.graph import OverlayGraph, wireKOut
"""peersim.dynamics — topology wiring controls.

WireGraph is the base (wires the live overlay over a Linkable protocol). WireKOut
is the faithful k-random-out wiring (PeerSim's default, == this project's
`random_kout`). The ring/full/star/mesh variants mirror the topology names in
`own_network/network_topology.py` so both runtimes accept the same CONFIG.
"""



class WireGraph(Control):
    """peersim.dynamics.WireGraph — apply a wiring over Linkable protocol `pid`."""
    def __init__(self, pid, undir=False):
        self.pid = pid
        self.undir = undir

    def execute(self):
        g = OverlayGraph(self.pid, directed=not self.undir)
        if g.size() == 0:
            return False
        self.wire(g)
        return False

    def wire(self, g):
        raise NotImplementedError


class WireKOut(WireGraph):
    """peersim.dynamics.WireKOut — each node → k random out-neighbours."""
    def __init__(self, pid, k, undir=False):
        super().__init__(pid, undir)
        self.k = k

    def wire(self, g):
        wireKOut(g, self.k, CommonState.r)


class WireRing(WireGraph):
    """Each node → its successor (ring)."""
    def wire(self, g):
        n = g.size()
        for i in range(n):
            g.setEdge(i, (i + 1) % n)


class WireFull(WireGraph):
    """Fully-connected mesh — every node → every other."""
    def wire(self, g):
        n = g.size()
        for i in range(n):
            for j in range(n):
                if i != j:
                    g.setEdge(i, j)


class WireStar(WireGraph):
    """Nodes 1..N-1 → node 0 (centralised anti-pattern, for comparison)."""
    def wire(self, g):
        n = g.size()
        for i in range(1, n):
            g.setEdge(i, 0)


class WireMesh(WireGraph):
    """Each node → neighbours at distance 1 and 2."""
    def wire(self, g):
        n = g.size()
        for i in range(n):
            g.setEdge(i, (i + 1) % n)
            g.setEdge(i, (i + 2) % n)
