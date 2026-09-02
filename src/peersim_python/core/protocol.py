"""peersim.core.Protocol — marker base; every protocol must be cloneable.

PeerSim builds one prototype protocol then clone()s it per node, so a node's
protocol object *is* its state. Subclasses override clone() when a plain
deep-copy is wrong (e.g. large shared data assigned later by an initializer).
"""

import copy


class Protocol:
    """peersim.core.Protocol — marker base; every protocol must be cloneable."""
    def clone(self):
        return copy.deepcopy(self)
