"""Non-linear (kernel) gossip-SDCA — a third tenant of the PeerSim engine.

This package is **model code only**. It reuses the same single-process PeerSim
engine (`src/peersim_python/`) and gossip substrate (`GossipProtocol`) that the
linear SDCA learner and the GADGET baseline run on; all file-path / dataset /
output wiring lives in the shared entry point `src/scripts/main.py` (run it as
`python src/scripts/main.py -m nonlinear -d <dataset>`).

What differs from the linear SDCA tenant:

- **Kernel, not a linear model.** The decision function is
  ``f(x) = sum_j alpha_j K(x_hat_j, x_hat)`` with an RBF kernel
  ``K(a,b) = exp(-gamma ||a-b||^2)``. There is no explicit weight vector ``w``.

- **DSRR random projection as the communication + recovery layer.** Every node
  projects its data with the SAME random matrix ``A in R^{m x d}``
  (``x_hat = A x``, ``d -> m``) generated deterministically from the shared
  ``SEED`` and never transmitted (Yang et al., *Dual-sparse Regularized
  Randomized Reduction*, ICML 2015, in ``notes/dual-sparce-reg.pdf``). Johnson-
  Lindenstrauss guarantees distances are preserved, so the RBF kernel computed on
  projected data approximates the kernel on the original data — this is what makes
  the compact, privacy-preserving exchange recoverable.

- **Accumulate, not average.** Nodes gossip only their projected support vectors
  and dual coefficients ``{(x_hat_i, alpha_i) : alpha_i != 0}``, tagged per origin
  and versioned; a node keeps the freshest block per origin and scores any point
  against the *union* of all support vectors it has heard of
  (`SupportVectorAccumulator`), rather than averaging weight vectors.

Modules: `projection` (the shared DSRR projector), `kernel` (RBF helpers),
`aggregator` (the support-vector accumulator), `kernel_sdca_protocol` (the tenant),
`observers` (data init + kernel-space duality-gap evaluator), `simulation` (the
orchestrator), and `reference_kernel_sdca` (a single-machine correctness oracle).
"""
