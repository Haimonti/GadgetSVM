"""peersim.cdsim.CDSimulator — the main experiment loop.

Programmatic equivalent of PeerSim's config-file driver: instead of reading
`init.*` / `control.*` / `simulation.cycles` from a text file, they are passed
in directly (the `Simulation` orchestrator assembles them).

Loop order per cycle (train then observe):
  1. FullNextCycle — every node runs one nextCycle (local update + gossip).
  2. controls (observers) — measure state; if any returns True, stop.
Note: PeerSim appends FullNextCycle *last* among controls; here it runs first
each cycle so the convergence observer reads freshly-updated models before
deciding to stop. Everything else matches PeerSim's control flow.
"""

from src.peersim_python.cdsim.cd_state import CDState
from src.peersim_python.cdsim.full_next_cycle import FullNextCycle
from src.peersim_python.core.network import Network
from src.peersim_python.logger import logger


class CDSimulator:
    """peersim.cdsim.CDSimulator — the main experiment loop."""
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
