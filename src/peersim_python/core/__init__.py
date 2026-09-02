"""peersim.core — Python port of PeerSim's core building blocks.

Faithful to PeerSim 1.0.5: Fallible, CommonState, Protocol, Linkable, Control,
Node/GeneralNode, Network, Scheduler — one class per module. This package
``__init__`` re-exports them as the ``peersim.core`` namespace (the legitimate
public surface, analogous to Java's ``import peersim.core.Network``), so
consumers keep writing ``from src.peersim_python.core import Network``.

Internal rule: submodules import their siblings by full path (e.g.
``from src.peersim_python.core.common_state import CommonState``), never through
this ``__init__``, to avoid partial-initialisation import cycles.
"""

from src.peersim_python.core.fallible import Fallible
from src.peersim_python.core.common_state import CommonState
from src.peersim_python.core.protocol import Protocol
from src.peersim_python.core.linkable import Linkable
from src.peersim_python.core.control import Control
from src.peersim_python.core.general_node import GeneralNode, Node
from src.peersim_python.core.network import Network
from src.peersim_python.core.scheduler import Scheduler

__all__ = [
    "Fallible", "CommonState", "Protocol", "Linkable", "Control",
    "GeneralNode", "Node", "Network", "Scheduler",
]
