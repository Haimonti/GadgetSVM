"""peersim.core.CommonState — the global simulation context.

One process, one global state: the current time (== cycle number in
cycle-driven mode), the current node/protocol being executed, and the single
shared RNG ``r`` that every component draws from. This global singleton is the
heart of why PeerSim is centralised at the *runtime* level.
"""

import random as _random


class CommonState:
    """peersim.core.CommonState — global simulation context.

    One process, one global state: the current time (== cycle number in
    cycle-driven mode), the current node/protocol being executed, and the single
    shared RNG `r` that every component draws from. This global singleton is the
    heart of why PeerSim is centralised at the *runtime* level.
    """
    time = 0
    endtime = -1
    phase = 0
    pid = 0
    node = None
    r: "_random.Random" = None  # the shared RNG

    @classmethod
    def getTime(cls):
        return cls.time

    @classmethod
    def setTime(cls, t):
        cls.time = t

    @classmethod
    def getEndTime(cls):
        return cls.endtime

    @classmethod
    def setEndTime(cls, t):
        cls.endtime = t

    @classmethod
    def getPid(cls):
        return cls.pid

    @classmethod
    def setPid(cls, p):
        cls.pid = p

    @classmethod
    def getNode(cls):
        return cls.node

    @classmethod
    def setNode(cls, n):
        cls.node = n

    @classmethod
    def initializeRandom(cls, seed):
        """Create (once) and seed the shared RNG."""
        if cls.r is None:
            cls.r = _random.Random()
        cls.r.seed(seed)
