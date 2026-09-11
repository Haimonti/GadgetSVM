from src.peersim_python.core import Network
"""peersim.graph — minimal Graph view + GraphFactory.wireKOut.

Only the pieces the wiring controls need: an OverlayGraph that maps graph edges
onto the nodes' Linkable protocol, and the exact k-random-out wiring algorithm
from PeerSim's GraphFactory.
"""



class OverlayGraph:
    """peersim.graph.OverlayGraph — a Graph view over Linkable protocol `pid`.

    setEdge(i, j) becomes `node_i.getProtocol(pid).addNeighbor(node_j)`. When
    the graph is undirected, the reverse edge is added too. This is the bridge
    between a wiring algorithm and the nodes' actual neighbour lists.
    """
    def __init__(self, pid, directed=True):
        self.pid = pid
        self.directed = directed

    def size(self):
        return Network.size()

    def setEdge(self, i, j):
        Network.get(i).getProtocol(self.pid).addNeighbor(Network.get(j))
        if not self.directed:
            Network.get(j).getProtocol(self.pid).addNeighbor(Network.get(i))


def wireKOut(g, k, r):
    """peersim.graph.GraphFactory.wireKOut — k distinct random out-neighbours.

    Partial Fisher-Yates draw without replacement; a drawn target equal to the
    source is retried without advancing (no self-loops).
    """
    n = g.size()
    if n < 2:
        return g
    if n <= k:
        k = n - 1
    nodes = list(range(n))
    for i in range(n):
        j = 0
        while j < k:
            newedge = j + r.randint(0, n - j - 1)  # == nextInt(n-j)
            nodes[j], nodes[newedge] = nodes[newedge], nodes[j]
            if nodes[j] != i:
                g.setEdge(i, nodes[j])
                j += 1
            # else: same slot retried, resampling a non-self target
    return g
