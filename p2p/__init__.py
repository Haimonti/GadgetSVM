"""Decentralised (P2PFL) protocols for the FL algorithms in `methods/` and `cocoa/`.

Everything P2P-specific lives here; `methods/`, `cocoa/` and `run_benchmark.py`
are the untouched server-based baselines, and the PeerSim engine stays at
`src/network_layer/peersim_python/` where Sreekar's branch expects it, so the
two halves merge without conflict.

Run with `python -m p2p.run_peersim` from the repository root.

Each protocol here is the P2P counterpart of one server-based algorithm, written
against the PeerSim engine in `src/network_layer/peersim_python/`. The server
implementations are left untouched — they are the baselines these are compared
against. `methods/centralized.py` has no counterpart by construction: it is the
single-machine upper bound both settings are measured against.

Every protocol follows the shape of `peersim_python/sdca_protocol.py` — the same
four-step nextCycle, the same set_data signature, the same metrics keys — so one
DataInitializer and one set of controls drive all of them.

Each local solver is verified numerically identical to its server original, so
the only thing that differs between a run_benchmark.py result and a PeerSim
result is the aggregation rule. That is the entire point of the port.
"""
# The gossip layer was refactored on branch `sreekar` (d72eebd): the flat
# src/network_layer/peersim_python/ this package was written against is gone,
# replaced by src/peersim_python/ with a GossipProtocol base class and pluggable
# aggregators. BDSVM has been ported (p2p/bdsvm_gossip.py); the other five still
# target the old layout and are kept out of the import path until they follow.
# The stale engine is parked at .stale/peersim_old for reference.
from p2p.bdsvm_gossip import BDSVMGossipProtocol

PROTOCOLS = {
    "bdsvm": BDSVMGossipProtocol,
}
