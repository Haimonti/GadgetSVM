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
from p2p.fedavg_protocol import FedAvgProtocol
from p2p.cocoa_protocol import CoCoAProtocol, CoCoAPlusProtocol
from p2p.bdsvm_protocol import BDSVMProtocol
from p2p.fdr_svm_protocol import FDRSVMProtocol
from p2p.fedssl_protocol import FedSSLAMCProtocol

PROTOCOLS = {
    "fedavg_svm": FedAvgProtocol,
    "cocoa":      CoCoAProtocol,
    "cocoa_plus": CoCoAPlusProtocol,
    "bdsvm":      BDSVMProtocol,
    "fdr_svm":    FDRSVMProtocol,
    "fedssl_amc": FedSSLAMCProtocol,
}
