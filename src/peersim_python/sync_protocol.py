"""Synchronous all-neighbour rounds on the cycle-driven engine (learner-agnostic).

Gossip tenants (`GossipProtocol`) push to random peers whenever they run. The
decentralised-averaging baselines (COLA, CHOCO-SGD) instead assume synchronous
rounds: every node computes from the neighbour values of round t, then every
node sends to *all* its neighbours, then round t+1 starts. On the PeerSim engine
that is two phases per cycle:

    phase A  nextCycle (every node, any activation order)
             -> compute_round() reads only own state + inbox, returns the message
    phase B  SynchronousBarrier control (after all nextCycles)
             -> deliver() for every node, then commit() for every node

Because phase A never reads a neighbour's live state, the shuffle order cannot
leak round-(t+1) values into round t.

    class MyProto(SyncGossipProtocol):
        def compute_round(self): ...    # -> outgoing message
        def message_nbytes(self, msg): ...
        def apply_round(self, msg): ...  # adopt own update + inbox
"""

import time

from src.peersim_python.cdsim.cd_protocol import CDProtocol
from src.peersim_python.cdsim.cd_state import CDState
from src.peersim_python.core import Control, Network


class SyncGossipProtocol(CDProtocol):
    """Base for a node in synchronous all-neighbour rounds."""

    def __init__(self):
        self.ready = False
        self.node_id = None
        self.neighbours = []
        self.inbox = {}
        self.metrics = []
        self.comm_bytes = 0
        self.stopped = False
        self.stop_cycle = None
        self.start = None
        self.compute_time = 0.0
        self._pending = None
        self._last_cycle = None

    # ---- hooks a learner implements -------------------------------------------
    def compute_round(self):
        raise NotImplementedError

    def message_nbytes(self, msg):
        raise NotImplementedError

    def apply_round(self, msg):
        raise NotImplementedError

    # ---- phase A ---------------------------------------------------------------
    def nextCycle(self, node, protocolID):
        if not self.ready or self.stopped:
            return
        cycle = CDState.getCycle()
        if self._last_cycle == cycle:
            raise RuntimeError("synchronous protocol: a node may run once per cycle")
        self._last_cycle = cycle
        t0 = time.perf_counter()
        self._pending = self.compute_round()
        self.compute_time += time.perf_counter() - t0

    # ---- phase B (driven by SynchronousBarrier) ---------------------------------
    def receive(self, sender, msg):
        self.inbox[sender] = msg

    def deliver(self, protos):
        msg = self._pending
        for i in self.neighbours:
            protos[i].receive(self.node_id, msg)
        self.comm_bytes += len(self.neighbours) * self.message_nbytes(msg)

    def commit(self):
        self.apply_round(self._pending)
        self._pending = None


class SynchronousBarrier(Control):
    """Runs after every node's phase A: deliver all messages, then commit all."""

    def __init__(self, pid):
        self.pid = pid

    def execute(self):
        protos = [Network.get(i).getProtocol(self.pid) for i in range(Network.size())]
        live = [p for p in protos if p.ready and not p.stopped]
        if not live:
            return False
        if any(p._pending is None for p in live):
            raise RuntimeError("a node skipped phase A; synchronous rounds are broken")
        for p in live:
            p.deliver(protos)
        for p in live:
            p.commit()
        return False
