"""peersim.cdsim.FullNextCycle — runs each up node's CDProtocol once per cycle.

`activation` chooses node visiting order within a cycle:
  - "shuffle"      : a fresh random permutation (each node fires once) — the
                     default; models asynchronous ordering while keeping one
                     update per node per round for clean convergence curves.
  - "getpair_rand" : random WITH replacement (PeerSim's getpair mode) — more
                     aggressively asynchronous; some nodes may fire twice,
                     others not at all in a cycle.
  - "ordered"      : array order (deterministic).
"""

from src.peersim_python.core.control import Control
from src.peersim_python.core.network import Network
from src.peersim_python.core.common_state import CommonState
from src.peersim_python.cdsim.cd_state import CDState
from src.peersim_python.cdsim.cd_protocol import CDProtocol


class FullNextCycle(Control):
    """peersim.cdsim.FullNextCycle — runs each up node's CDProtocol once per cycle."""
    def __init__(self, activation="shuffle"):
        self.activation = activation

    def execute(self):
        cycle = CDState.getCycle()
        size = Network.size()
        from src.peersim_python.communication import activation_order
        order=activation_order(size,getattr(CommonState,'experiment_seed',0),cycle,self.activation)
        for j,index in enumerate(order):
            node=Network.get(index)
            if not node.isUp():
                continue
            CommonState.setNode(node)
            CDState.ctime = j
            for k in range(node.protocolSize()):
                p = node.getProtocol(k)
                if isinstance(p, CDProtocol):
                    CommonState.setPid(k)
                    p.nextCycle(node, k)
                    if not node.isUp():  # node died mid-cycle
                        break
        return False
