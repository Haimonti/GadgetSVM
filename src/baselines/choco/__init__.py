"""CHOCO-SGD with CHOCO-Gossip (Koloskova, Stich, Jaggi, ICML 2019), as a PeerSim tenant.

One implementation: CHOCO-SGD is the learner and CHOCO-Gossip (compressed
difference q = Q(x - x_hat) + public copies x_hat) is its communication step.
A faithful port of the official convex code (github.com/epfml/ChocoSGD,
`convex_code/logistic.py`, method="choco", Apache-2.0), run on the same engine,
seeded graph, shards and metrics as our linear SDCA.

| Official code                                   | Here                                       |
|-------------------------------------------------|--------------------------------------------|
| `fit` loop body, method == "choco"              | `ChocoProtocol.compute_round` + barrier     |
| `x_hat.dot(W - I)` mixing                       | sum_j W_ij (x_hat_j - x_hat_i), N(i) u {i}  |
| `__create_mixing_matrix` (ring 1/3, ...)        | `peersim_python/mixing.py` Metropolis (= 1/3 on the ring) on our graph |
| `__quantize` / `qsgd_quantize`                  | `peersim_python/compression.py` (CHOCO_COMPRESSOR) |
| `BaseLogistic.lr` (constant / decay / bottou)    | `ChocoProtocol.lr` (CHOCO_LR_TYPE, CHOCO_INITIAL_LR, CHOCO_TAU) |
| `consensus_lr`                                  | CHOCO_CONSENSUS_LR (paper-tuned: full 1.0, qsgd-8 0.34, top-k 0.04, random-k 0.01 on epsilon) |
| one sample per node per step                    | CHOCO_BATCH (default 1)                     |
| x0 ~ N(0, 0.01) tiled, x_hat = x0               | same (shared seed)                          |

Deliberate, documented differences:
  * Loss: logistic gradient -> hinge subgradient `y a [y a.x < 1] - lambda x`
    (our objective; CHOCO-SGD's theory holds for convex losses with bounded
    subgradients). This is the only change to the algorithm.
  * Graph: Metropolis weights on our wired graph instead of their fixed
    ring/torus list (identical on the ring).
  * Evaluation: each node's own x (our schema) rather than their node-average;
    no dual exists, so convergence is the primal objective, and the plot
    analyzer reports suboptimality against centralized SDCA's P*.
  * Bytes: exact compressor size x degree per round (the official code logs none).
  * One SGD step per round is the algorithm, so CHOCO needs many more rounds
    than SDCA's 4096 coordinate steps per cycle; raise ROUNDS accordingly.
"""
