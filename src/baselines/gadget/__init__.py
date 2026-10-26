"""GADGET SVM baseline, ported into the PeerSim-Python engine.

This package holds the GADGET SVM model of Hensel & Dutta (*GADGET SVM: a
Gossip-bAseD sub-GradiEnT SVM solver*) --- a Pegasos sub-gradient local solver
whose nodes gossip by **averaging their weight vectors**. It is a self-contained
baseline that reuses the *same* distributed engine (`src.peersim_python.core` /
`cdsim`), the *same* config and the *same* data shards as the gossip-SDCA setup,
so the two can be compared head to head on identical data, topology and budget.

Only the model differs from SDCA:
  * local learner  : Pegasos sub-gradient  (SDCA uses dual coordinate ascent)
  * communicated   : the weight vector w   (SDCA gossips versioned dual contributions)
  * convergence    : objective change      (SDCA uses the duality gap)

Entry point: `src/scripts/main.py -m gadget -d <dataset>` via the `GadgetSimulation` orchestrator.
"""
