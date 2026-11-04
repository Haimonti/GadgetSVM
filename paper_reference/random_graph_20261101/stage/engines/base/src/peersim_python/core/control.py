"""peersim.core.Control — a network-wide hook.

execute() returns True iff the simulation must STOP. Observers and initializers
return False; a convergence check returns True once its threshold is met. Every
component that is not a per-node protocol (initializers, observers, topology
wirers) is a Control.
"""


class Control:
    """peersim.core.Control — a network-wide hook (True from execute() = stop)."""
    def execute(self):
        raise NotImplementedError
