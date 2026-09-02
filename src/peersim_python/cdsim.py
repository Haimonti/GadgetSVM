"""peersim.cdsim — the cycle-driven simulation engine.

CDProtocol (nextCycle per node per cycle), CDState (adds the cycle counter),
FullNextCycle (runs every node's CDProtocol once per cycle), and CDSimulator
(the main experiment loop). This is the engine the gossip-SDCA simulation runs on.
"""

from src.peersim_python.core import (
    CommonState, Network, Control, Protocol,
)
from src.peersim_python.logger import logger


class CDProtocol(Protocol):
    """peersim.cdsim.CDProtocol — a cycle-driven protocol.

    nextCycle(node, protocolID) is invoked once per cycle for every up node.
    """
    def nextCycle(self, node, protocolID):
        raise NotImplementedError


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


class FullNextCycle(Control):
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
    def __init__(self, activation="shuffle"):
        self.activation = activation

    def execute(self):
        cycle = CDState.getCycle()
        size = Network.size()
        order = None
        if self.activation == "shuffle":
            order = list(range(size))
            CommonState.r.shuffle(order)

        for j in range(size):
            if self.activation == "getpair_rand":
                node = Network.get(CommonState.r.randint(0, size - 1))
            elif self.activation == "shuffle":
                node = Network.get(order[j])
            else:
                node = Network.get(j)

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


class CDSimulator:
    """peersim.cdsim.CDSimulator — the main experiment loop.

    Programmatic equivalent of PeerSim's config-file driver: instead of reading
    `init.*` / `control.*` / `simulation.cycles` from a text file, they are
    passed in directly.

    Loop order per cycle (train then observe):
      1. FullNextCycle — every node runs one nextCycle (local update + gossip).
      2. controls (observers) — measure state; if any returns True, stop.
    Note: PeerSim appends FullNextCycle *last* among controls; here it runs
    first each cycle so the convergence observer reads freshly-updated models
    before deciding to stop. Everything else matches PeerSim's control flow.
    """
    def __init__(self, cycles, initializers=None, controls=None, activation="shuffle"):
        self.cycles = cycles
        self.initializers = initializers or []
        self.controls = controls or []
        self.runner = FullNextCycle(activation)

    def nextExperiment(self):
        CDState.setEndTime(self.cycles)
        CDState.setCycle(0)

        # init.* — run once, in order (wiring, data assignment, ...)
        for init in self.initializers:
            init.execute()
        logger.info("cdsim", f"Initialised — {Network.size()} nodes, {self.cycles} cycles max")

        stopped_at = self.cycles
        for i in range(self.cycles):
            CDState.setCycle(i)
            self.runner.execute()                 # every node: local step + gossip
            stop = False
            for c in self.controls:               # observers
                if c.execute():
                    stop = True
            if stop:
                stopped_at = i + 1
                logger.info("cdsim", f"Convergence threshold met — stopping at cycle {i}")
                break

        logger.info("cdsim", f"Simulation finished after {stopped_at} cycle(s)")
        return stopped_at
