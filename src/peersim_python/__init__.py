"""PeerSim 1.0.5 core — Python port.

A faithful re-implementation of PeerSim's cycle-driven simulation engine in
Python, structured to mirror the Java class/method names (Node, Network,
Protocol, Linkable, CommonState, Control, Scheduler, CDProtocol, CDSimulator,
WireKOut, ...). It is a *simulator*: all nodes live in one process and take
turns in a single loop, exactly like real PeerSim — the decentralisation is in
the protocol logic, not the runtime.

This package is the shared, learner-agnostic distribution engine. The models
that ride on it are separate tenant packages — `src/linear_sdca/`,
`src/baselines/gadget/`, `src/nonlinear_sdca/` and `src/baselines/mom_ndsvm/` — each with its own protocol, observers
and `Simulation`, all run from the shared entry point `src/scripts/main.py`.

Layout: `core/` and `cdsim/` are the ported PeerSim engine (one class per file);
`gossip_protocol.py` + `aggregator.py` are the reusable gossip substrate (the
base `Aggregator` and the concrete merge strategies every tenant draws from);
`idle_protocol.py`, `dynamics.py`, `graph.py` and `logger.py` round out the
engine. The learner protocols and their orchestrators live in the tenant
packages, not here.
"""
