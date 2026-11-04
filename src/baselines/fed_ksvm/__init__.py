"""Fed-KSVM — "A Memory-Efficient Federated Kernel Support Vector Machine for Edge
Devices", X. Zhou and X. Wang, IEEE TNNLS 35(12), 2024. Federated (server-based)
nonlinear baseline, run in the paper's own setup (standalone, not on PeerSim).

| Paper                                           | Here                                   |
|-------------------------------------------------|----------------------------------------|
| random Fourier features a(x), D dims (Sec. III-B) | `rff.RandomFourierFeatures` (W ~ N(0, 2 gamma I), shared seed) |
| primal (3), dual (4), local subproblem (5)      | `algorithm.Device.local_round`          |
| block boosting, Algorithm 1, (8)-(11), SDCA     | `block_boosting.BlockBoosting`          |
| server update (7), Algorithm 2                  | `algorithm.FedKSVMRun.run`              |
| secure aggregation (12)-(13)                    | `FEDKSVM_SECURE_AGG` (off by default)   |
| 1 coordinator + M MPI workers, synchronous      | plain synchronous loop, M = NUM_WORKERS |
| i.i.d. / k-means non-i.i.d. split               | SHARDING = iid / kmeans                 |

Shared with our models: the dataset and test set (`load_shards`), C = 1/(LAMBDA
n_global) (same regularised problem; no bias, as in the paper), RBF gamma =
RBF_GAMMA, metric definitions. Comm: D float64 down + D up per device per round.

Explicit choices where the paper is silent:
  * device-side dual step u_m += (beta_M / M) Delta_m, which keeps w = sum_m u_m
    (the CoCoA rule the paper builds on; with beta_M = 1 this is CoCoA averaging);
  * FEDKSVM_Q (block iterations per round, default L) and FEDKSVM_H (SDCA passes
    per block solve) — the paper's Table III values are not reproduced here;
  * the secure-aggregation seed stands in for the Diffie-Hellman key exchange.
RBF only: random Fourier features need a shift-invariant kernel (no polynomial).
"""
