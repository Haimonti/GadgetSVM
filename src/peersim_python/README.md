# peersim_python

A faithful Python port of PeerSim 1.0.5's cycle-driven engine, plus a reusable
gossip layer. This package is **the shared distribution engine only** — it is
learner-agnostic. Each model is a separate tenant package that rides on this
engine: `src/linear_sdca/` (gossip-SDCA), `src/baselines/gadget/` (the GADGET SVM
baseline), `src/nonlinear_sdca/` (RBF-kernel SDCA) and `src/baselines/mom_ndsvm/`
(Forero et al.'s MoM-NDSVM, a synchronous ADMM `CDProtocol` rather than a gossip
tenant). Each tenant owns its own
protocol, observers and `Simulation`. All of them are run from the shared entry
point `src/scripts/main.py` (`python src/scripts/main.py -m <model> -d <dataset>`).
Everything runs in one process: all nodes take turns in a single loop (a
*simulator*), so the decentralisation lives in the protocol logic, not the
runtime.

## Layout

The ported PeerSim engine is split one-class-per-file into two subpackages that
mirror the Java packages `peersim.core` and `peersim.cdsim`. The reusable gossip
substrate sits at the package root. The learner protocols and their `Simulation`
orchestrators are **not** here — they live in the tenant packages above.

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
| `gossip_protocol.py` | **`GossipProtocol`** — the reusable, learner-agnostic gossip loop: inbox + send (via `Linkable`) + merge (via an `Aggregator`). A tenant supplies `current_state`/`set_state`/`outgoing_payload`/`local_update`/`payload_nbytes`; neighbours are drawn without replacement. |
| `aggregator.py` | `Aggregator` interface + `PlainAverageAggregator`, `IncrementAggregator`, and `VersionedContributionAggregator` — the merge strategies tenants draw from (e.g. linear SDCA uses the versioned one; GADGET uses plain-average). |
| `idle_protocol.py` | `IdleProtocol` — the reference `Linkable`: stores a node's neighbour list (protocol id 0). |
| `graph.py` | `OverlayGraph` + `wireKOut` (PeerSim's exact k-random-out wiring). |
| `dynamics.py` | Topology-wiring controls: `WireKOut` (`random_kout`), `WireRing`, `WireFull`, `WireStar`, `WireMesh`. |
| `logger.py` | Tiny p2pfl-free stdout logger. |
| `__init__.py` | Package docstring / design contract. |

The learner protocol, the `DataInitializer`/evaluator observers and the
`Simulation` orchestrator for each model live in that model's tenant package
(`src/linear_sdca/`, `src/baselines/gadget/`, `src/nonlinear_sdca/`,
`src/baselines/mom_ndsvm/`), not here.

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
