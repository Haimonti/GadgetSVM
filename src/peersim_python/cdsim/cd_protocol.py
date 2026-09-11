"""peersim.cdsim.CDProtocol — a cycle-driven protocol.

nextCycle(node, protocolID) is invoked once per cycle for every up node. This is
the one method the cycle-driven engine calls on each protocol every round.
"""

from src.peersim_python.core.protocol import Protocol


class CDProtocol(Protocol):
    """peersim.cdsim.CDProtocol — a cycle-driven protocol."""
    def nextCycle(self, node, protocolID):
        raise NotImplementedError
