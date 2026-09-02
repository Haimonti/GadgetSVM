# peersim_python

A faithful Python port of PeerSim 1.0.5's cycle-driven engine, with the
gossip-SDCA learner bolted on top as a protocol. Everything runs in one process:
all nodes take turns in a single loop (a *simulator*), so the decentralisation
lives in the protocol logic, not the runtime. Driven by the repo-root
`peersim_run.py`.

## Files

| File | Use case |
|------|----------|
| `__init__.py` | Package marker + docstring. States the design contract (single-process PeerSim-style simulator, p2pfl-free). |
| `logger.py` | Tiny stdout logger (`logger.info(tag, msg)`) — a p2pfl-free stand-in so the package runs without p2pfl installed. |
| `core.py` | PeerSim core classes: `CommonState` (global RNG/time), `Protocol`, `Linkable`, `Control`, `GeneralNode`/`Node`, `Network` (the one global node array), `Scheduler`. The engine's foundation. |
| `idle_protocol.py` | `IdleProtocol` — the reference `Linkable`: just stores a node's neighbour list. This is protocol id 0 on every node. |
| `graph.py` | `OverlayGraph` (maps graph edges onto nodes' neighbour lists) + `wireKOut` (PeerSim's exact k-random-out wiring algorithm). |
| `dynamics.py` | Topology-wiring controls: `WireKOut` (the `random_kout` default), plus `WireRing`, `WireFull`, `WireStar`, `WireMesh`. Run once at init. |
| `cdsim.py` | The cycle-driven engine: `CDProtocol` (per-cycle protocol), `CDState` (cycle counter), `FullNextCycle` (ticks every node once per cycle), `CDSimulator` (the main experiment loop with init → train → observe → stop). |
| `sdca_protocol.py` | `SDCAProtocol` — the learner. Per cycle each node: merges received weights, runs one closed-form SDCA epoch on its shard, gossips its weight to neighbours, records primal/dual/gap/accuracy. Includes `sgd_init()` (paper's SGD warm start). |
| `observers.py` | Two controls: `DataInitializer` (hands each node its data shard + hyperparameters at startup) and `ConvergenceObserver` (stops the sim once every node's duality gap is below threshold). |

## Per-cycle flow

```
wire topology → assign shards → [ merge inbox → local SDCA epoch → gossip push → record ] × cycles → converge/stop
```

> **Note:** `sdca_protocol.py`'s `_merge_inbox` uses age-weighted averaging of the
> absolute weight vector, which breaks SDCA's primal–dual consistency and diverges
> (then overflows to NaN on long runs). The fix is CoCoA-style increment
> aggregation (share the per-round change, not the absolute weight).
