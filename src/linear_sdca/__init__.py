"""Linear gossip-SDCA — one tenant of the PeerSim distribution engine.

This package is the *model* for linear SDCA: the learner protocol, its data
initializer + evaluator, and the simulation that wires them onto the shared
engine in `src/peersim_python/`. It mirrors `src/baselines/gadget/` and
`src/nonlinear_sdca/` so all three models have the same shape. All filepath /
output wiring lives in the shared entry point `src/scripts/main.py` (run it as
`python src/scripts/main.py -m sdca -d <dataset>`); this package stays model-only.

Modules: `sdca_protocol` (the `SDCAProtocol` learner), `observers`
(`DataInitializer` + `GlobalEvaluator`, where the duality gap is measured), and
`simulation` (`Simulation`, the orchestrator). The gossip substrate, aggregators
and PeerSim core they depend on live in `src/peersim_python/`.
"""
