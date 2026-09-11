"""peersim.core.Fallible — a node's health state.

A node is in exactly one of three states. `OK` nodes participate in the
simulation; `DOWN`/`DEAD` nodes are skipped by the engine. Split out of the old
monolithic ``core.py`` so each PeerSim component lives in its own file.
"""


class Fallible:
    """peersim.core.Fallible — a node's health state."""
    OK = 0
    DEAD = 1
    DOWN = 2
