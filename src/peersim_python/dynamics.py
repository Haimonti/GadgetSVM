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

class WireGrid(WireGraph):
    """2-D non-periodic grid (COLA's `grid`) — n must be a perfect square.

    Node i sits at (i // s, i % s) with s = sqrt(n); it links to its right and
    lower neighbour, so interior nodes have degree 4 and corners degree 2.
    """
    def wire(self, g):
        n = g.size()
        s = int(round(n ** 0.5))
        if s * s != n:
            raise ValueError(f"grid topology needs a perfect-square node count, got {n}")
        for i in range(n):
            r, c = divmod(i, s)
            if c + 1 < s:
                g.setEdge(i, i + 1)
            if r + 1 < s:
                g.setEdge(i, i + s)


class WireConnectedCycle(WireGraph):
    """c-connected cycle (COLA's `connected_cycle`) — each node links to its c
    nearest nodes on each side of the ring (c=1 is the plain ring)."""
    def __init__(self, pid, c, undir=False):
        super().__init__(pid, undir)
        self.c = c

    def wire(self, g):
        n = g.size()
        for i in range(n):
            for d in range(1, self.c + 1):
                j = (i + d) % n
                if j != i:
                    g.setEdge(i, j)


TOPOLOGIES = ("random_kout", "ring", "full", "complete", "star", "mesh", "grid",
              "connected_cycle")


def make_topology(cfg, pid):
    """Map CONFIG["TOPOLOGY"] to its Wire* control (always undirected).

    Shared by every PeerSim tenant so all models see the same graph for the same
    (TOPOLOGY, SEED, NUM_WORKERS). `complete` is an alias of `full`;
    `connected_cycle` reads CONFIG["CYCLE_CONNECTIVITY"] (default 2).
    """
    name = cfg["TOPOLOGY"]
    if name == "random_kout":
        return WireKOut(pid, cfg["GOSSIP_K"], undir=True)
    if name == "ring":
        return WireRing(pid, undir=True)
    if name in ("full", "complete"):
        return WireFull(pid, undir=True)
    if name == "star":
        return WireStar(pid, undir=True)
    if name == "mesh":
        return WireMesh(pid, undir=True)
    if name == "grid":
        return WireGrid(pid, undir=True)
    if name == "connected_cycle":
        return WireConnectedCycle(pid, cfg.get("CYCLE_CONNECTIVITY", 2), undir=True)
    raise ValueError(f"Unknown topology '{name}'. Choose: {' | '.join(TOPOLOGIES)}")
