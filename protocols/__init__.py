"""Decentralised (P2PFL) protocols for the FL algorithms in `methods/` and `cocoa/`.

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
from protocols.fedavg_protocol import FedAvgProtocol
from protocols.cocoa_protocol import CoCoAProtocol, CoCoAPlusProtocol
from protocols.bdsvm_protocol import BDSVMProtocol
from protocols.fdr_svm_protocol import FDRSVMProtocol
from protocols.fedssl_protocol import FedSSLAMCProtocol

PROTOCOLS = {
    "fedavg_svm": FedAvgProtocol,
    "cocoa":      CoCoAProtocol,
    "cocoa_plus": CoCoAPlusProtocol,
    "bdsvm":      BDSVMProtocol,
    "fdr_svm":    FDRSVMProtocol,
    "fedssl_amc": FedSSLAMCProtocol,
}
