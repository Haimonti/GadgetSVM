"""Non-linear (kernel) gossip-SDCA — a third tenant of the PeerSim engine.

This package is **model code only**. It reuses the same single-process PeerSim
engine (`src/peersim_python/`) and gossip substrate (`GossipProtocol`) that the
linear SDCA learner and the GADGET baseline run on; all file-path / dataset /
output wiring lives in the shared entry point `src/scripts/main.py` (run it as
`python src/scripts/main.py -m nonlinear -d <dataset>`).

What differs from the linear SDCA tenant:

- **Kernel, not a linear model.** The decision function is
  ``f(x) = sum_j alpha_j K(x_hat_j, x_hat)`` with an RBF kernel
  ``K(a,b) = exp(-gamma ||a-b||^2)`` or a polynomial kernel (CONFIG["KERNEL"];
  these are the only two in scope). There is no explicit weight vector ``w``.

- **DSRR random projection as the communication + recovery layer.** Every node
  projects its data with the SAME random matrix ``A in R^{m x d}``
  (``x_hat = A x``, ``d -> m``) generated deterministically from the shared
  ``SEED`` and never transmitted (Yang et al., *Dual-sparse Regularized
  Randomized Reduction*, ICML 2015, in ``notes/research_papers/dual-sparce-reg.pdf``).
  Johnson-Lindenstrauss preserves squared distances up to ``(1 +/- eps)``, so the
  RBF error is multiplicative in the exponent: ``K_hat = exp(-gamma D^2 (1 +/- eps))``,
  i.e. ``K^(1+eps) <= K_hat <= K^(1-eps)`` (a ratio of ``exp(+/- gamma eps D^2)``,
  not ``1 +/- eps``). Training runs in the reduced space; DSRR's *recovery* step
  then scores with the learned duals and the ORIGINAL-space kernel
  (``accuracy_recovered``, by decentralised partial scores — see
  ``KernelEvaluator._recovered``).

- **Dual-sparse regulariser tau.** DSRR adds ``(tau/n)||alpha||_1`` to the reduced
  dual; for the hinge loss this is exactly a hinge with the smaller margin
  ``1 - tau`` in the primal (paper eq. 6). Hence the update target ``1 - tau`` and
  the gap pair ``P = mean max(0, (1-tau) - y f) + reg``, ``D = (1-tau) mean(a) - reg``.

- **Accumulate, not average.** Nodes gossip only their projected support vectors
  and dual coefficients ``{(x_hat_i, alpha_i) : alpha_i != 0}``, tagged per origin
  and versioned; a node keeps the freshest block per origin and scores any point
  against the *union* of all support vectors it has heard of
  (`SupportVectorAccumulator`), rather than averaging weight vectors.

Modules: `projection` (the shared DSRR projector), `kernel` (RBF / polynomial helpers),
`aggregator` (the support-vector accumulator), `kernel_sdca_protocol` (the tenant),
`observers` (data init + kernel-space duality-gap evaluator), `simulation` (the
orchestrator), and `reference_kernel_sdca` (a single-machine correctness oracle).
"""
