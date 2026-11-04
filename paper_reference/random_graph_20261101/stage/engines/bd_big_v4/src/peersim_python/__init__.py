"""PeerSim 1.0.5 core — Python port.

A faithful re-implementation of PeerSim's cycle-driven simulation engine in
Python, structured to mirror the Java class/method names (Node, Network,
Protocol, Linkable, CommonState, Control, Scheduler, CDProtocol, CDSimulator,
WireKOut, ...). It is a *simulator*: all nodes live in one process and take
turns in a single loop, exactly like real PeerSim — the decentralisation is in
the protocol logic, not the runtime.

This package lets the project's gossip-SDCA setup run as a PeerSim-style P2P
network simulation (driven by `src/peersim_run.py` via the `Simulation`
orchestrator), independently of the p2pfl runtime used by `src/own_network/`.

Layout: `core/` and `cdsim/` are the ported PeerSim engine (one class per file);
`gossip_protocol.py` + `aggregator.py` are the reusable, learner-agnostic gossip
substrate; `sdca_protocol.py` is one tenant of it; `simulation.py` is the
orchestrator that wires everything together.
"""
