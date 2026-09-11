"""peersim.cdsim.CDState — CommonState plus the current cycle number.

Extends the global CommonState with the cycle-driven notion of "which cycle are
we in", keeping ``getTime() == cycle`` as PeerSim does.
"""

from src.peersim_python.core.common_state import CommonState


class CDState(CommonState):
    """peersim.cdsim.CDState — CommonState plus the current cycle number."""
    cycle = -1
    ctime = -1

    @classmethod
    def getCycle(cls):
        return cls.cycle

    @classmethod
    def setCycle(cls, t):
        cls.cycle = t
        cls.ctime = 0
        cls.setTime(t)  # keep getTime() == cycle, as in PeerSim
