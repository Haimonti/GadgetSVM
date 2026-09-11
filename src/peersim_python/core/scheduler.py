"""peersim.core.Scheduler — decides at which times a component is active.

A pure predicate over the cycle number: ``active(time)`` is True when ``time``
falls on the schedule (``from``, ``from+step``, … below ``until``, or a one-shot
``at``). NOTE: currently orphaned — the engine does not yet consult it; wiring it
into control/protocol execution is a planned follow-up (see the package README /
plan seams).
"""

from src.peersim_python.core.common_state import CommonState


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
