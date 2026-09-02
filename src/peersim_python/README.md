# peersim_python

A faithful Python port of PeerSim 1.0.5's cycle-driven engine, with a reusable
gossip layer and the gossip-SDCA learner as one tenant on top. Everything runs
in one process: all nodes take turns in a single loop (a *simulator*), so the
decentralisation lives in the protocol logic, not the runtime. Driven by
`src/peersim_run.py` via the `Simulation` orchestrator.

## Layout

The ported PeerSim engine is split one-class-per-file into two subpackages that
mirror the Java packages `peersim.core` and `peersim.cdsim`. The reusable gossip
substrate and the SDCA tenant sit at the package root.

### `core/` — PeerSim core (== `peersim.core`)
| File | Class | Use case |
|------|-------|----------|
| `fallible.py` | `Fallible` | Node health-state constants (OK/DEAD/DOWN). |
| `common_state.py` | `CommonState` | Global singleton: time/cycle, current node/pid, the one shared RNG `r`. |
| `protocol.py` | `Protocol` | Marker base; every protocol is cloneable (prototype → clone per node). |
| `linkable.py` | `Linkable` | Neighbour-view interface (the overlay edges / topology layer). |
| `control.py` | `Control` | Network-wide hook; `execute()→True` stops the sim. |
| `general_node.py` | `GeneralNode` (+ `Node` alias) | One peer = an ordered list of protocol objects. |
| `network.py` | `Network` | The one global `Node[]` array that *is* the network. |
| `scheduler.py` | `Scheduler` | `active(time)` predicate — **orphaned**, not yet wired into the engine. |
| `__init__.py` | — | Re-exports the above as the `core` namespace. |

### `cdsim/` — cycle-driven engine (== `peersim.cdsim`)
| File | Class | Use case |
|------|-------|----------|
| `cd_protocol.py` | `CDProtocol` | Adds `nextCycle(node, pid)` — called once per node per cycle. |
| `cd_state.py` | `CDState` | `CommonState` + the current cycle counter. |
| `full_next_cycle.py` | `FullNextCycle` | Runs every up node's `nextCycle` once per cycle (shuffle/getpair/ordered). |
| `cd_simulator.py` | `CDSimulator` | The main loop: initializers once → per cycle run + observe → stop. |
| `__init__.py` | — | Re-exports the four engine classes. |

### Package root
| File | Use case |
|------|----------|
| `simulation.py` | **`Simulation`** — the orchestrator. Seeds RNG, builds the network, wires topology, assigns shards, sets up observers, and drives `CDSimulator`. Public API: `Simulation(config, worker_data).run(cycles)`. |
| `gossip_protocol.py` | **`GossipProtocol`** — the reusable, learner-agnostic gossip loop: inbox + send (via `Linkable`) + merge (via an `Aggregator`). A tenant supplies `get_payload`/`set_payload`/`local_update`/`record`/`payload_nbytes`. |
| `aggregator.py` | `Aggregator` interface + `PlainAverageAggregator` (equal-weight mean). Swappable merge rule. |
| `sdca_protocol.py` | `SDCAProtocol(GossipProtocol)` — the learner tenant. Payload = primal weight `w`; local step = one closed-form SDCA epoch; `alpha` stays local. Includes `sgd_init()` warm start. |
| `idle_protocol.py` | `IdleProtocol` — the reference `Linkable`: stores a node's neighbour list (protocol id 0). |
| `graph.py` | `OverlayGraph` + `wireKOut` (PeerSim's exact k-random-out wiring). |
| `dynamics.py` | Topology-wiring controls: `WireKOut` (`random_kout`), `WireRing`, `WireFull`, `WireStar`, `WireMesh`. |
| `observers.py` | `DataInitializer` (hands each node its shard) + `ConvergenceObserver` (stops when every node's gap < threshold). |
| `logger.py` | Tiny p2pfl-free stdout logger. |
| `__init__.py` | Package docstring / design contract. |

## Per-cycle flow

```
wire topology → assign shards → [ drain inbox + aggregate → local update → gossip push → record ] × cycles → converge/stop
```

## Communication model

- **Topology (universal):** `Linkable`/`IdleProtocol` hold each node's neighbour
  list. Config `TOPOLOGY="random_kout"`, `GOSSIP_K=3` → a random mesh where each
  node gossips to 3 neighbours per round.
- **Transport + aggregation (reusable):** `GossipProtocol` owns the inbox, the
  push, and the merge — any learner can reuse it. The merge rule is a pluggable
  `Aggregator` (default `PlainAverageAggregator`).
- **Payload:** only the weight vector `w` travels; the SDCA dual variables
  `alpha` stay local.

> **Note:** the default `PlainAverageAggregator` overwrites `w` without touching
> `alpha`, so on long runs the primal–dual invariant can drift (plateau, not
> convergence). A CoCoA-style increment aggregator (share the per-round change,
> not the absolute weight) would be the drop-in fix — add it as another
> `Aggregator` without touching the protocol.
