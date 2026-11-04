"""Centralized SDCA baseline, on the PeerSim-Python engine as ONE node.

The linear baseline for our gossip SDCA: the same regularised hinge-loss problem
(lambda, no bias) and the same closed-form SDCA coordinate step, but with every
worker's shard held by a single node. The network has one node and no links, so
nothing is ever sent (comm_bytes = 0). It answers "what does decentralisation
cost?" and stores the reference optimum P* (exact, liblinear) that every linear
model's suboptimality P(w) - P* is measured against.

Same engine as every tenant: `Network`, `GeneralNode`, `CDSimulator` (one
`nextCycle` per cycle) and a measuring `Control`.

  * `central_protocol.py` — `CentralSDCAProtocol` (the learner, a CDProtocol)
  * `observers.py`        — `CentralDataInitializer`, `CentralEvaluator`, `reference_optimum`
  * `simulation.py`       — `CentralSDCASimulation` (assembly + run)

One cycle = `NUM_WORKERS * SDCA_LOCAL_STEPS` coordinate updates, the total work all
distributed nodes do in one cycle, so iteration axes line up with the P2P models.
Stops at gap < `CENTRAL_GAP_TOL` (default 1e-6), on a plateau, or at `ROUNDS`.

Entry point: `src/scripts/main.py -m central_sdca -d <dataset>`.
"""
