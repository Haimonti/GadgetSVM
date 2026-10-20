"""PeerSim protocol registry for the current engine.

Only the BDSVM port is registered here. Historical old-engine protocols are
retained for inspection. FDR references live in paper_reference. I/Q SSL has a separate PeerSim
assembly in src.fedssl_simulation because its data contract differs from
the tabular SVM protocols in this registry.
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
