"""SDCA's measuring Control with a second way to stop: the iterate has stalled.

`GlobalEvaluator` stops a node when its duality gap drops under GAP_THRESHOLD.
On covtype that fires around cycle 160. On gisette, real-sim and rcv1 it never
fires for some or all nodes — not because they are still moving, but because
the gap of the gossip fixed point sits just above the line (gisette 1.2e-4,
rcv1 1.02e-4, real-sim 1.38e-3) and stays there, to four digits, for thousands
of cycles with consensus error at 1e-10. The iterate has converged; the
certificate is simply not as tight as the constant asked for.

So this subclass adds the criterion BDSVM already uses (Algorithm 2 step 9):
the relative movement of the iterate between evaluations,

    ||w - w_prev|| / ||w_prev||  <  stall_tol

and a node stops on whichever fires first, gap or stall — still only once it
has heard from every origin, as before. The gap rule is untouched, so a run
that stopped on the gap before stops there now too; the stall rule only ever
adds a stop, never removes one.

`stall_tol` is compared across one `eval_every` window (10 cycles by default),
not one cycle. It started at 1e-4 — a converged gossip iterate moves ~1e-10 per
window, a still-converging one 1e-3 to 1e-2. a9a showed the gap in that
reasoning: once six nodes stop and go silent, the four still running are left
gossiping frozen contributions and creep along a flat direction at ~2.5e-4 per
window, decaying so slowly that 1e-4 is never reached. At 1e-3 they stop at
cycle ~1200 with test accuracy already equal to BDSVM's. The cost is that the
other stall-stopped runs stop earlier too (gisette 900 -> 390, real-sim
100 -> 60) at unchanged accuracy. The measured value is written into every
metrics row as `rel_change` so the margin is visible rather than assumed.
"""

import numpy as np

from src.peersim_python.cdsim import CDState
from src.peersim_python.core import Network
from src.peersim_python.observers import GlobalEvaluator
from src.peersim_python.simulation import Simulation
from src.peersim_python.logger import logger


class StallAwareGapEvaluator(GlobalEvaluator):
    """GlobalEvaluator plus a stop on the iterate's relative movement."""

    def __init__(self, *args, stall_tol=1e-3, **kwargs):
        super().__init__(*args, **kwargs)
        self.stall_tol = float(stall_tol)

    def _is_eval_cycle(self, cycle):
        is_last = self.total_cycles is not None and cycle >= self.total_cycles - 1
        return not (cycle % self.eval_every and not is_last)

    def execute(self):
        cycle = CDState.getCycle()
        if not self._is_eval_cycle(cycle):
            return False

        # The gap rule, the metrics row, the log line — all as before.
        gap_says_stop = super().execute()

        protos = [Network.get(i).getProtocol(self.pid) for i in range(Network.size())]
        newly = 0
        for p in protos:
            w_prev = getattr(p, "_w_at_last_eval", None)
            if w_prev is None:
                rel = float("inf")
            else:
                denom = float(np.linalg.norm(w_prev))
                rel = (float(np.linalg.norm(p.w - w_prev)) / denom
                       if denom > 0.0 else float("inf"))
            p._w_at_last_eval = p.w.copy()
            if p.metrics:
                p.metrics[-1]["rel_change"] = rel
            is_complete = len(p.contributions) == len(protos)
            if (self.stop_on_threshold and not p.stopped
                    and is_complete and rel < self.stall_tol):
                p.stopped = True
                p.stop_cycle = cycle
                p.stop_reason = "stall"
                newly += 1
            elif p.stopped and not hasattr(p, "stop_reason"):
                p.stop_reason = "gap"
        if newly:
            logger.info("observer",
                        f"cycle={cycle}  {newly} node(s) stopped on stall "
                        f"(rel_change < {self.stall_tol:g})")
        if not self.stop_on_threshold:
            return False
        return gap_says_stop or all(p.stopped for p in protos)


class StallAwareSimulation(Simulation):
    """`Simulation` with `StallAwareGapEvaluator` in place of `GlobalEvaluator`.

    Everything else — prototype, wiring, DataInitializer, cycle loop — is the
    parent's, so the only thing that changes is when a node may stop.
    """

    def __init__(self, config, worker_data, stall_tol=1e-3):
        super().__init__(config, worker_data)
        self.stall_tol = stall_tol

    def build(self):
        super().build()
        old = self.sim.controls[0]
        assert isinstance(old, GlobalEvaluator)
        self.sim.controls[0] = StallAwareGapEvaluator(
            self.SDCA_PID, old.gap_threshold,
            eval_every=old.eval_every,
            stop_on_threshold=old.stop_on_threshold,
            stall_tol=self.stall_tol,
        )
        return self
