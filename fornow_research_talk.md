# PeerSim & Distributed ML Simulation — Conversation Summary

## 1. Python Alternatives to PeerSim
- **PeersimGym / peersim-environment** (Frederico Metelo et al.): not a true rewrite — wraps the Java PeerSim engine in a Spring Boot REST server, exposed to Python via a PettingZoo/Gym class (`GET /state`, `POST /action`). Built and used for multi-agent RL on task-offloading.
- **PyActive / JPyActive** (URV cloudlab): a native-Python *actor library*, not a port. Benchmarked its Chord implementation against PeerSim/PlanetSim/Macedon.
- **Key pattern:** people who want PeerSim in Python tend to *wrap the JVM*, not reimplement it — because PeerSim's value is scale, which pure Python can't match. No widely-adopted pure-Python PeerSim clone exists.

## 2. What PeerSim Is
- Java-based simulator for **peer-to-peer overlay networks**, GPL-licensed.
- Purpose: test distributed *protocols/algorithms* (gossip, epidemic aggregation, overlays, churn) without deploying to a real network.
- Core tradeoff: **scale over realism** — abstracts the network to delays + drop rates; cycle engine tested to ~250k nodes.
- No GUI, no debugger — driven by a flat plain-text config file; results captured via observer Controls writing to disk.

## 3. The Two Engines
- **Cycle-driven** (`peersim.cdsim`): simplified, no real time/messages. Loop = `for each cycle → for each node → nextCycle(node, pid)`. Maximum scale. Natural fit for gossip/SDCA rounds.
- **Event-driven** (`peersim.edsim`): realistic discrete-event model. A **priority queue of timestamped events**; engine pops earliest, advances clock, delivers to `processEvent()`. Messages are real objects through a `Transport` layer (delay/drop applied). Auto-detected via the `simulation.endtime` config parameter.
- Bridge: a cycle-style protocol can run inside the event engine via `CDProtocol` + a `CDScheduler`.

## 4. Core Elements
| Element | Role |
|---|---|
| **Node** | One peer = a *stack of protocols*, accessed via `node.getProtocol(pid)`. |
| **Network** | Global Java array of all nodes (the rows of a matrix). |
| **Protocol** | Where algorithm logic lives. Interfaces: `CDProtocol` (cycle), `EDProtocol` (event). Can be non-executable (data-only). |
| **Linkable** | A node's neighbor view; how a gossip protocol picks a peer. |
| **Transport** | Communication channel abstraction (event model); applies delay/drop on `send()`. |
| **Control** | Network-wide components: **Initializers** (setup/topology), **Observers** (collect stats), **Dynamics** (churn/topology change). |
| **Scheduler** | Governs *when* protocols/controls run; default = every cycle, configurable. |

- **Mental model:** a matrix — Network = rows of Nodes, each Protocol = a column (one instance per node).
- **Cloning, not construction:** instances are created by cloning one prototype; user code must implement `clone()` correctly or shared-reference bugs leak state across nodes.

## 5. The Engine Internals
- The "engine" is **pure Java** — ordinary classes (`CDSimulator`, `EDSimulator`), no C++, no JNI, no native runtime.
- An engine = **a main loop + a scheduler**. Cycle engine = nested for-loops; event engine = a heap of pending events.
- Scale comes from **abstraction**, not a fast language. The performance ceiling is **RAM**, not CPU.

## 6. Nodes as Java State
- A node is just a **Java object on the JVM heap**; its protocol objects' instance fields *are* its state (e.g. an SDCA dual variable αᵢ is a `double` field).
- Because everything shares one heap, protocol A can reach into node B's fields directly — PeerSim uses these **object-invocation shortcuts** instead of always sending messages.
- **Consequence:** peer isolation is a *convention you maintain*, not a barrier the engine enforces. Accidentally reading cross-node/global state a real peer couldn't know → unrealistically good (buggy) results.
- "State" here = mutable data in objects, **not** a formal state-machine.

## 7. Persistence
- Node state is **not persistent** — lives only while the process runs; killing the JVM erases it.
- PeerSim is a **batch program**, not a live service: JVM starts → builds nodes → loops → writes stats → exits. Persistence is the *observer's* job (write to disk during the run), not the node's. Resume by re-running from config + an Initializer.
- **Java vs Python state:** same idea mechanically. Differences that matter: JVM objects are far leaner (fits 100k+ nodes in RAM) and JIT-compiled loops are much faster than interpreted Python. Cross-node shared-heap access behaves identically.

## 8. PeerSim Is a Tool, Not a Machine
- Just a Java **library/tool** (a JAR + config) run on your **local, single machine, single JVM**. Not HPC, not a cluster, not itself distributed.
- The distribution is **fictional** — all nodes are objects taking turns in one loop. Analogy: a flight simulator vs an airplane.
- Limit is memory, not compute. You can run many independent runs in parallel for sweeps, but each run is sequential and local.

## 9. Real-World Persistence (what replaces the simulation)
PeerSim only *mimics* these; production splits them into separate layers:
1. **Networking (real message passing):** libp2p (Gossipsub — used by Ethereum/IPFS), gRPC.
2. **Per-node durable state:** RocksDB / LevelDB (embedded KV stores), SQLite.
3. **Cluster-wide persistence + consistency:** consensus via Raft (etcd) / Paxos; distributed DBs like Cassandra, CockroachDB, etcd.
- For decentralized ML: **p2pfl deployed** is the real-world counterpart — real nodes, real transport, each persisting local model/dual state to disk.

## 10. Why Java (not C++ or Rust)?
- **Era + goal:** early-2000s, choice was Java vs C++. Java won on GC (no manual memory mgmt for huge churning node graphs), portability (run the JAR anywhere), and fast iteration. For a tool whose output is a *research paper*, that beats C++'s ~2–3× speed.
- **Ceiling is RAM anyway:** C++/Rust shrink per-object overhead but don't change the fundamental limit; JIT makes Java "fast enough" on tight loops.
- **Rust didn't exist** (1.0 = 2015; PeerSim predates it). Today Rust would be the strong pick for a fresh build — C++ speed without memory-safety landmines (why modern P2P infra like libp2p leans Rust).

## 11. Java-for-ML Was Fine — But Your Case Flips It
- Old PeerSim ML (e.g. GADGET SVM) used **deliberately tiny** ML: linear/logistic regression = a weight vector + a per-node update, a few lines of arithmetic. No NumPy needed; the *protocol* was the hard part, not the model.
- Your instinct to switch is right: the moment you want autograd, optimized linear algebra, existing model code, and ready datasets (RCV1), Python becomes the natural home.

## 12. Integrating Python ML with PeerSim — Three Options
- **A. Don't integrate; use p2pfl (best for you).** p2pfl already *is* a Python distributed-learning framework doing what PeerSim does (nodes, neighbors, gossip) but natively with real ML. For gossip SDCA on RCV1, you don't need PeerSim at all.
- **B. Wrap Java PeerSim behind Python (PeersimGym-style).** Only if you specifically need PeerSim's churn/topology/scale machinery p2pfl lacks. Cost: ugly two-process boundary. Overkill for SDCA.
- **C. Reimplement PeerSim in Python (almost never worth it).** You'd lose its whole reason for existing — scale. Weeks of work for a *worse* simulator.
- **Decision rule:** researching the *protocol at massive scale* → PeerSim/Java. Researching the *ML algorithm over a modest decentralized network* → Python/p2pfl. **Your SDCA work is the second.** Borrow PeerSim *concepts* (cycle loop, Linkable peer-view) as design inspiration only.

## 13. SDCA: Java or Python? → Python
- SDCA's per-coordinate update is cheap in either language; the hard part is the **primal-dual plumbing** — maintaining `w = Σ αᵢxᵢ`, computing the **duality gap** as convergence certificate, verifying gossip preserves the invariant while FedAvg-style averaging breaks it. Python gives fast iteration + easy inspection for exactly this.
- **Decisive factor — RCV1 is sparse:** `scipy.sparse` + NumPy give C-backed sparse dot products and updates. The hot arithmetic runs in compiled BLAS, not the interpreter — so Python is both easier *and* not slower. Java has no comparably mature sparse-ML stack.
- **Caveat + fix:** naive single-coordinate `for` loops are slow in Python → vectorize over mini-batches of dual coordinates, or JIT with Numba/Cython only if profiling demands it.
- **Decisive for your setup:** SDCA must live *inside* the p2pfl learner so the gossip aggregator can pass dual state between nodes. Keep optimizer and framework in **one language**. The old Java SVM papers used Java because their *framework* was Java — **the framework decides, not the algorithm**, and yours is Python.