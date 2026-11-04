"""peersim.core.GeneralNode — one peer = a container of protocols.

A node holds an ordered list of protocol objects (accessed by protocol id).
getID() is a permanent unique id; getIndex() is the (mutable) position in the
global Network array. In PeerSim ``Node`` is the interface and ``GeneralNode``
the default impl; here ``Node`` is a module-level alias to ``GeneralNode``.
"""

import copy

from src.peersim_python.core.fallible import Fallible


class GeneralNode(Fallible):
    """peersim.core.GeneralNode — one peer = a container of protocols."""
    counterID = -1

    def __init__(self, protocols=None):
        self.protocol = list(protocols) if protocols else []
        self.index = -1
        self.failstate = Fallible.OK
        self.ID = GeneralNode._nextID()

    @classmethod
    def _nextID(cls):
        cls.counterID += 1
        return cls.counterID

    def clone(self):
        """Deep-clone every protocol and assign a fresh unique ID (PeerSim semantics)."""
        result = copy.copy(self)
        result.protocol = [p.clone() for p in self.protocol]
        result.ID = GeneralNode._nextID()
        return result

    def getProtocol(self, i):
        return self.protocol[i]

    def protocolSize(self):
        return len(self.protocol)

    def getIndex(self):
        return self.index

    def setIndex(self, index):
        self.index = index

    def getID(self):
        return self.ID

    def getFailState(self):
        return self.failstate

    def setFailState(self, state):
        if self.failstate == Fallible.DEAD and state != Fallible.DEAD:
            raise RuntimeError("cannot resurrect a DEAD node")
        if state == Fallible.DEAD:
            self.index = -1
            self.failstate = Fallible.DEAD
            for p in self.protocol:
                if hasattr(p, "onKill"):
                    p.onKill()
        else:
            self.failstate = state

    def isUp(self):
        return self.failstate == Fallible.OK

    def __hash__(self):
        return int(self.ID)


# In PeerSim `Node` is the interface; GeneralNode is the default impl.
Node = GeneralNode
