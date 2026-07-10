"""peersim.core — Python port of PeerSim's core building blocks.

Faithful to PeerSim 1.0.5: Fallible, Node/GeneralNode, Network, Protocol,
Linkable, CommonState, Control, Scheduler. Method names are kept in PeerSim's
camelCase so the simulation reads like the Java original.
"""

import copy
import random as _random


class Fallible:
    """peersim.core.Fallible — a node's health state."""
    OK = 0
    DEAD = 1
    DOWN = 2


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


class Protocol:
    """peersim.core.Protocol — marker base; every protocol must be cloneable.

    PeerSim builds one prototype protocol then clone()s it per node, so a node's
    protocol object *is* its state. Subclasses override clone() when a shallow
    deep-copy is wrong (e.g. large shared data assigned later by an initializer).
    """
    def clone(self):
        return copy.deepcopy(self)


class Linkable:
    """peersim.core.Linkable — a node's neighbour view (the overlay edges).

    Behaves as an ordered set of neighbours: unique elements, random access,
    no removal. Gossip protocols pick a peer through this interface.
    """
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


class Control:
    """peersim.core.Control — a network-wide hook.

    execute() returns True iff the simulation must STOP. Observers/initializers
    return False; a convergence check returns True once the threshold is met.
    """
    def execute(self):
        raise NotImplementedError


class GeneralNode(Fallible):
    """peersim.core.GeneralNode — one peer = a container of protocols.

    A node holds an ordered list of protocol objects (accessed by protocol id).
    getID() is a permanent unique id; getIndex() is the (mutable) position in
    the global Network array.
    """
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


class Network:
    """peersim.core.Network — the one global Node[] array.

    The whole "network" is a single Python list living in this class. Every node
    is an entry; only the first size() entries are live. This is the clearest
    sign that PeerSim is a single-process simulator: there is exactly one array,
    in one process, that *is* the network.
    """
    node: list = []
    len = 0
    prototype = None

    @classmethod
    def reset(cls, size, prototype):
        """Build the network by cloning the prototype `size` times."""
        cls.prototype = prototype
        prototype.setIndex(-1)
        cls.node = []
        for _ in range(size):
            cls.node.append(prototype.clone())
        cls.len = size
        for i, nd in enumerate(cls.node):
            nd.setIndex(i)

    @classmethod
    def size(cls):
        return cls.len

    @classmethod
    def get(cls, index):
        return cls.node[index]

    @classmethod
    def add(cls, n):
        cls.node.append(n)
        n.setIndex(cls.len)
        cls.len += 1

    @classmethod
    def shuffle(cls):
        """Fisher-Yates over the live nodes using the shared RNG."""
        for i in range(cls.len, 1, -1):
            cls.swap(i - 1, CommonState.r.randint(0, i - 1))

    @classmethod
    def swap(cls, i, j):
        cls.node[i], cls.node[j] = cls.node[j], cls.node[i]
        cls.node[i].setIndex(i)
        cls.node[j].setIndex(j)


class Scheduler:
    """peersim.core.Scheduler — decides at which times a component is active."""
    def __init__(self, step=1, from_=0, until=None, fin=False, at=None):
        if at is not None:
            self.from_ = at
            self.until = at + 1
            self.step = 1
        else:
            self.step = step
            self.from_ = from_
            self.until = until if until is not None else float("inf")
        self.fin = fin

    def active(self, time=None):
        if time is None:
            time = CommonState.getTime()
        return self.from_ <= time < self.until and (time - self.from_) % self.step == 0
