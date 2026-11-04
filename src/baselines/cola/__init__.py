"""COLA (He, Bian, Jaggi, NeurIPS 2018) — decentralised CoCoA, as a PeerSim tenant.

A faithful port of the official code (github.com/epfml/cola, Apache-2.0) for the
hinge-loss linear SVM (`LinearSVM`, data split by samples), run on the same engine,
seeded graph, shards and metrics as our linear SDCA. Only the algorithm differs.

| Official code                                  | Here                                     |
|------------------------------------------------|------------------------------------------|
| `cola/algo.py::cola` loop body                 | `ColaProtocol.compute_round` + barrier    |
| `comm.local_average` (sum_j W_kj v_j)          | same, over N(k) u {k}                    |
| `p2p_communicate_neighborhood_tensors`         | `SynchronousBarrier` -> deliver to all    |
| `cola/graph.py` mixing weights                 | `peersim_python/mixing.py` (Metropolis — equal to COLA's on ring/complete/cycle/grid) |
| `LinearSVM.solve` + `SVMCoordSolver.fit`       | `local_solver.ColaLinearSVMSolver`        |
| `fast_cd/svm.pyx::sparse_cd` (incl. xorshift)  | `local_solver.sparse_cd`                  |
| gamma = 1 (fixed in run_cola.py), sigma = gamma*K | `COLA_GAMMA` (default 1.0)            |
| `--local_iters` (passes), `--theta` (tol)      | `COLA_LOCAL_ITERS`, `COLA_THETA`          |
| `--c`                                          | C = 1/(LAMBDA * n_global)  (same optimum as ours) |
| `Monitor` global mode: cv2, gap                | evaluator fields `cv2`, `duality_gap`     |

Deliberate, documented differences (none change the algorithm):
  * Comm bytes are counted (the official code logs none): deg * 8 d per round,
    plus the initial zero exchange.
  * The per-iteration MPI all-reduce of running time (an exit-time check) is not
    modelled; the run stops on our run-level gap / plateau / ROUNDS rule.
  * Faithful quirks kept: the solver warm-starts from its previous `dual_coef_`
    (not `xk`); `sparse_cd`'s early-exit gap omits the factor C that `dense_cd`
    applies (only matters when COLA_THETA > 0); all nodes share the solver
    `random_state` (= SEED), as `run_cola.py` passes one value to every rank.
  * Optional message compression (CONFIG["COMPRESSOR"], default "full") for the
    quantization ablation; COLA itself sends full-precision v.
"""
