# Model explanations: what each model does, as currently set up

This file describes the code as it is now. For each model it gives the idea, what happens in one round, what gets sent, when it stops, which settings matter, and what makes it slow. Default values are taken from `src/config.py`. Exact file paths are listed in `README.md`.

---

## 0. What every model shares

**The problem.** Every model trains the same SVM:

```
P(w) = λ/2 ‖w‖²  +  (1/n) Σ_i max(0, 1 − y_i · f(x_i))
```

- Linear models use `f(x) = wᵀx`.
- Kernel models use `f(x) = (1/λn) Σ_j α_j K(x_j, x)`.
- `λ = LAMBDA = 1e-4`, and `n` is the size of the whole training set across all nodes.
- The dual methods (SDCA, COLA, Fed-KSVM, nonlinear) also track the dual `D(α)`. The **duality gap** `P − D ≥ 0` measures how far the model is from the optimum, and it reaches 0 at the optimum.

**The data.**
- `src/scripts/data_sharding.py` splits the training set into `NUM_WORKERS = 10` disjoint shards. With `SHARDING = iid` the split is random; the other options are `dirichlet`, `label_sorted` and `kmeans`.
- Every node is evaluated on the **same** test set.

| dataset | train n | test | features d | per node (10 nodes) |
|---|---|---|---|---|
| gisette | 4,800 | 1,200 (20% split) | 5,000 | ~480 |
| a9a (`adult`) | 32,561 | 16,281 | 123 | ~3,256 |
| w8a | 49,749 | 14,951 | 300 | ~4,975 |
| ijcnn1 | 49,990 | 91,701 | 22 | ~4,999 |

**The two communication styles.**
- **Gossip (asynchronous style):** sdca, gadget, nonlinear.
  - Each cycle, every node does local work and then pushes its state to `GOSSIP_K = 3` random neighbours on a `random_kout` graph.
  - There is no clock everyone waits on. It's like rumours spreading through a crowd.
- **Synchronous rounds:** cola, choco.
  - Every node computes, then everyone exchanges with **all** neighbours at once, mixing with a fixed weight matrix `W` (Metropolis weights).
  - It's like a meeting where everybody speaks once per round.
- **Single node on PeerSim:** central_sdca, the whole dataset on one node with no links.
- **Standalone (their own paper's setup, no PeerSim):**
  - ndsvm (Forero): synchronous neighbours on the paper's own random graph.
  - fed_ksvm: a server plus devices.

**Stopping (what the summary's `stop_reason` means).**

| stop_reason | meaning |
|---|---|
| `converged` | Every node met the model's own criterion. For gap models that is gap < `GAP_THRESHOLD = 1e-4`; for GADGET it is a stable objective. |
| `plateau` | The global metric did not improve by at least `PLATEAU_REL_EPS = 0.1%` for `PLATEAU_PATIENCE = 20` evaluations in a row. Evaluations run every `EVAL_EVERY = 10` cycles, so that is at least 200 cycles with no progress. |
| `budget` | The run used all `ROUNDS = 5000` (or the model's own budget) without either of the above, so it **did not converge**. |
| `fixed_budget` | The run was designed to use its whole budget (CHOCO only). |

Known issue: `GAP_THRESHOLD = 1e-4` is an absolute number. On gisette, P* ≈ 2.1e-4, so 1e-4 is about 50% of the optimum value. Most SDCA runs therefore end on `plateau` rather than `converged`. See the end of `README.md`.

**Communication accounting.**
- `comm_bytes` counts every byte a node sends. Dense float64 is 8 bytes per number.
- Compression (`COMPRESSOR` / `CHOCO_COMPRESSOR`) changes this through `src/peersim_python/compression.py`.

---

## 1. Linear models

### 1.1 `sdca`: our gossip SDCA (linear)

**Idea.** Each node owns its own data's dual variables. It publishes one versioned **contribution vector**, `c_k = X_kᵀ α_k / (λ n)`, and the global model is the **sum** of all nodes' contributions: `w = Σ_k c_k`.

Think of a group building one sum together. Everyone writes their own number on a shared board, and you only ever replace your own entry. Nothing is averaged, so nobody's work is diluted.

**One cycle on node k:**
1. Merge the incoming tables, keeping the newest version per origin, and rebuild `w` from the table.
2. Run `SDCA_LOCAL_STEPS = 4096` coordinate steps. Each picks a random local point `i` and does `α_i ← clip(α_i + (1 − y_i wᵀx_i)·λn/‖x_i‖²)`, then `w += Δα_i x_i/(λn)`.
3. Bump the version and push the entries a neighbour hasn't seen yet to 3 random neighbours.

**Sends:** only the table entries a neighbour hasn't seen at their current version (delta-send), at d×8 + 16 bytes per entry.

**Stops:**
- A node latches (stops training) when its gap on the global data drops below 1e-4 (`STOP_RULE = node_global_gap`).
- A latched node keeps relaying entries.
- If its gap rises above the threshold again (hysteresis), it resumes training.
- The run also stops on a plateau.

**Settings:**

| key | value | meaning |
|---|---|---|
| `SDCA_LOCAL_STEPS` | 4096 | local work per cycle (compute vs. communication ratio) |
| `SDCA_STEP_SCALE` | 1.0 | step damping; 1/K is the safe CoCoA+ value |
| `GOSSIP_K` | 3 | peers per push |
| `WARM_START` | True | one full local pass before any gossip |
| `STOP_RULE` | node_global_gap | alternatives: `node_local_gap` (own shard only), `mean_gap` (global oracle) |

**Cost:** 10 nodes × 4096 Python-level steps per cycle, about 0.3 s per cycle. Measured: a9a 1,661 cycles in about 7 min; gisette 1,101 cycles in about 8 min.

### 1.2 `central_sdca`: centralized SDCA baseline (one PeerSim node, all data)

> **On hold.** Being redesigned and not part of any run right now. The description below is the current code in `src/baselines/central_sdca/`.

**Idea.** The linear baseline for our SDCA. It is the same SDCA coordinate rule with step scale 1, run on your PeerSim engine as a 1-node network that holds every shard and has no links, so it never communicates. This is "what if one computer had all the data". Code: `src/baselines/central_sdca/`.

- **One round** = `NUM_WORKERS × SDCA_LOCAL_STEPS` = 40,960 steps, so its x-axis lines up with the distributed models.
- **Stops:** at gap < `CENTRAL_GAP_TOL = 1e-6`, or on a plateau.
- **P\*:** the reference optimum is computed separately and exactly with liblinear (`LinearSVC`, hinge, tol 1e-10). Every linear model's suboptimality `P(w) − P*` uses it.
- **Cost:** a9a 1,100 rounds in about 5 min; gisette 20 rounds in 11 s.

### 1.3 `gadget`: GADGET (Pegasos SGD and gossip averaging)

**Idea.** Each node runs Pegasos SGD on its own data. Nodes gossip their **whole weight vector `w`**, and the receiver **averages** it with its own. It's like friends repeatedly averaging their opinions until they agree.

**One cycle:**
1. 4096 Pegasos steps: `w ← (1 − 1/t)w`, and if `y_i wᵀx_i < 1`, then `w += (1/(λt)) y_i x_i`. Here `t` is the global step count, so the step size decays.
2. Push `w` to 3 neighbours.
3. Average whatever arrived.

- **Sends:** d × 8 bytes per push.
- **Stops:** GADGET's own rule. A node stops when its objective changes by less than `OBJ_THRESHOLD = 0.01` for `CONVERGENCE_COUNT = 10` evaluations in a row. There is no duality gap (primal only).
- **Cost:** cheap. a9a took 531 cycles in about 1 min.

### 1.4 `cola`: COLA (decentralized CoCoA, Jaggi's group)

**Idea.** Each node solves a small local **dual** problem around a shared vector `v`, then mixes `v` with its neighbours using `W`. It's like CoCoA without a server: neighbours repeatedly blend their estimates, and each node adds its own improvement.

**One round on node k:**
1. `v̄ = Σ_j W_kj v_j`, combining its own `v` and the neighbours' `v` from the last round.
2. Local solver (`LinearSVM`, a port of their `sparse_cd`): `COLA_LOCAL_ITERS = 1` pass of dual coordinate descent over the shard. It returns `Δx` and `Δv`.
3. `x += γΔx`, then `v = v̄ + γ K Δv`, with `γ = COLA_GAMMA = 1` and `σ = γK`.
4. Send `v` to **all** neighbours. The barrier delivers, then every node commits.

**Model and settings:**
- Model `w = C·v`, with `C = 1/(λ n)`.
- **Sends:** d × 8 bytes × degree per round.
- **Stops:** when all node gaps are below 1e-4, or on a plateau.
- **Budget:** `max(ROUNDS, COLA_ROUNDS = 20000)`, because COLA's gap shrinks roughly like 1/t.
- **Evaluation:** thinned to about 500 points.

**Cost: slow.**
- `sparse_cd` is a pure-Python loop: one Python iteration per local data point per round.
- a9a: about 32k Python steps per round, roughly 0.24 s per round. It needs about 10k rounds, so roughly **80 min per seed**.
- w8a and ijcnn1: about 50k steps per round, roughly 2 h per seed.
- Measured: gisette 10,441 rounds in 9 min (small shards).

### 1.5 `choco`: CHOCO-SGD with CHOCO-Gossip (Jaggi's group)

**Idea.** SGD plus gossip with **compressed** messages. Each node keeps a public copy `x̂` of itself and of each neighbour, and sends only a compressed **difference** `q = Q(x − x̂)`. It's like sending "here's what changed since last time" instead of the whole file, and the copies catch up over time.

**One round on node i:**
1. `x½ = x + lr·(−∇)` on `CHOCO_BATCH = 1` random sample. The gradient is the hinge subgradient; the official code uses logistic loss, which is the only change.
2. `x = x½ + γ_c Σ_j W_ij (x̂_j − x̂_i)`, with `γ_c = CHOCO_CONSENSUS_LR = 1.0`.
3. `q = Q(x − x̂_i)` with `Q = CHOCO_COMPRESSOR = full`. Send `q` to all neighbours, and everyone does `x̂_i += q`.

**Settings:**
- Learning rate `lr = CHOCO_INITIAL_LR / (λ (t + τ))`, with `CHOCO_INITIAL_LR = 0.1` and `τ = d`.
- **Sends:** the compressor's exact byte count × degree.
- **Stops:** never early (`fixed_budget`), because noisy SGD would trigger the plateau rule far too soon.
- **Budget:** `max(ROUNDS, 20 epochs × n_local, CHOCO_MIN_ROUNDS = 50000)`.
  - a9a: about 65k rounds.
  - w8a and ijcnn1: about 100k rounds.
  - gisette: 50k rounds.
- **Cost:** each round is tiny (one sample). gisette took 50k rounds in 55 s. The w8a and ijcnn1 runs are about 2× longer because of their larger budgets.

---

## 2. Nonlinear (kernel) models

### 2.1 `nonlinear`: our gossip kernel SDCA with DSRR

**Idea.** The same "sum of contributions" design as linear SDCA, but a kernel model has no weight vector. It is defined by its **support vectors** (points with α ≠ 0) and their α values. So nodes gossip **blocks of support vectors** instead of a vector `w`.

To make those blocks cheaper to send, DSRR first **projects** every point to `m = PROJECTION_DIM = 512` dimensions with a random matrix `A` that every node regenerates from the shared seed, so `A` itself is never sent.

**Setup per node:**
1. Project the shard: `x̂ = A x`.
   - Applies to gisette, where d = 5000 > 512.
   - a9a, w8a and ijcnn1 have d ≤ 512, so their raw features are used unchanged.
2. Precompute the local Gram matrix `K_local = K(x̂_i, x̂_j)`. This is **n_local × n_local** and dense.
3. Kernel: `KERNEL = rbf`, `exp(−γ‖a−b‖²)` with `γ = RBF_GAMMA = 0.005`. The other option is `poly`, `(g⟨a,b⟩ + 1)³`.

**One cycle on node k:**
1. Merge the incoming support-vector blocks, keeping the newest version per origin.
2. Recompute the "external score" of every local point:
   - `f_ext(x_i) = Σ_{other nodes' SVs} α_j K(x_j, x_i) / λn`
   - This is **(all other nodes' SVs) × (local points)** kernel evaluations.
3. 4096 kernel-SDCA steps:
   - `α_i ← clip(α_i + ((1−τ) − y_i f(x_i)) λn / K_ii)`
   - then `f_local += Δα_i K[:, i]/λn` (a vector update of length n_local).
4. Publish its own SV block and push unseen blocks to 3 neighbours.

**Evaluation (every 10 cycles, for every one of the 10 nodes):**
- Its model's score on **all n training points**, used for the gap: SVs × n kernel values.
- `‖f‖²`: SVs × SVs.
- Test accuracy: SVs × n_test.
- **Recovered accuracy** (`DSRR_RECOVERY = True`): the same α values scored with the kernel on the *raw* features. This is another SVs × n_test pass, on sparse data.

**Sends:** each support vector is `m × 8 + 16` bytes, sent once per version per neighbour.

**Stops:** the same latch, hysteresis and plateau rules as linear SDCA. The gap is τ-consistent: margin `1 − τ`, and `DSRR_TAU = 0`, so it is the plain hinge for now.

**Settings:**

| key | value | meaning |
|---|---|---|
| `KERNEL` | rbf | rbf or poly |
| `RBF_GAMMA` | 0.005 | kernel width; gisette with 0.1 overfits |
| `POLY_DEGREE` / `POLY_GAMMA` / `POLY_COEF0` | 3 / 1/d / 1.0 | polynomial kernel |
| `PROJECTION_DIM` | 512 | DSRR reduced dimension m |
| `PROJECTION_DIST` | sparse | sparse random signs (hashing) or gaussian |
| `DSRR_TAU` | 0.0 | dual-sparsity term; equivalent to hinge margin 1−τ |
| `DSRR_RECOVERY` | True | also report `accuracy_recovered` |

**Cost:** this is the expensive one. See section 4.

### 2.2 `ndsvm`: Forero MoM-NDSVM (JMLR 2010), standalone

**Idea.**
- A kernel model has no common weight vector to agree on. Forero picks **L = 100 shared "anchor" points** (Γ), drawn uniformly inside the data's bounding box from a shared seed.
- Nodes agree on the model's values **at those anchors**, using ADMM consensus with their direct neighbours.
- It's like neighbours who can't share their notes agreeing on answers to the same 100 reference questions.

**Setup:**
1. The paper's random geometric graph with J = 10 nodes (`FORERO_NETWORK = random`).
2. A class-balanced re-partition (`FORERO_PARTITION = balanced`).
3. A min/max flood to get the bounding box, then draw Γ.
4. Each node builds an **N_j × N_j** dense matrix `Q` from its local kernel.
5. Optional reference fits (`FORERO_REFERENCES = True`): a centralized sklearn `SVC` on all data, plus 10 local-only SVCs.

**One ADMM iteration, per node:**
1. Solve the local box-QP (35) **exactly**: Python coordinate descent to tolerance `NDSVM_QP_TOL = 1e-8`, with up to 10,000 sweeps.
2. Form the anchor values `w̃_j` (L numbers) and the bias `b_j`, and broadcast them to the neighbours.
3. Dual ascent on the disagreement: `α += η/2 Σ (w̃_j − w̃_i)`, with `η = NDSVM_ETA = 10`.

**Sends:** (L+1) × 8 bytes, as one broadcast per node per iteration (`FORERO_COMM_MODEL = broadcast`).

**Stops:** when every node's primal and dual residuals are ≤ `NDSVM_RESIDUAL_TOL = 1e-4`, or on a plateau of the mean objective.

**Cost:** the exact inner QP is a Python loop over N_j coordinates per sweep, each costing O(N_j). Cost grows like N_j². gisette (N_j ≈ 480) ran 211 iterations in about 2 min.

### 2.3 `fed_ksvm`: Fed-KSVM (Zhou & Wang, TNNLS 2024), standalone federated

**Idea.**
- Replace the RBF kernel with **D = 2000 random Fourier features** (`a(x) = √(2/D) cos(Wx + b)`, with W ~ N(0, 2γI) from a shared seed). The kernel SVM then becomes a *linear* SVM in 2000 dimensions.
- Devices run CoCoA-style local dual updates, and **a server** averages them.
- This is federated learning (one coordinator), not peer-to-peer.

**One round:**
1. The server broadcasts `w`.
2. Each device runs **block boosting**:
   - Split the 2000 features into `FEDKSVM_L = 10` blocks.
   - `Q = L` times: pick a block and run SDCA on it. That is `FEDKSVM_H = 1` pass over the local data per block, with only that block's 200 features touched (the paper's memory saving).
3. Each device uploads `Δ_m`. The server sets `w += β·(1/M) Σ Δ_m` (`β = FEDKSVM_BETA = 1`), and each device sets `u_m += (β/M) Δ_m`.
4. Optional pairwise-mask secure aggregation (`FEDKSVM_SECURE_AGG = False`).

**Sends:** 2 × D × 8 = 32 KB per device per round (down and up).

**Stops:** on a plateau of the objective, or after `ROUNDS`. Primal only; no gap is reported.

**Cost:**
- About 10 × n_local Python-level steps per device per round, so about 10 × n Python steps per round in total.
- That is roughly 1.5 s per round on a9a and about 2.5 s on w8a and ijcnn1.
- The feature matrices are large in memory. ijcnn1's test set alone is 91,701 × 2000 × 8 B ≈ 1.5 GB.

---

## 3. Ablations (linear only; `src/scripts/ablations.py`)

All five linear models run under each setting. Settings that don't apply to a model are skipped, and the reason is recorded in `comparison.json`.

| ablation | what changes |
|---|---|
| `stopping` | `GAP_THRESHOLD` ∈ {1e-2 … 1e-5}; `STOP_RULE` ∈ {node_global, node_local, mean} (SDCA only) |
| `stepsize` | λ ∈ {1e-3, 1e-4, 1e-5}; SDCA step scale {1, 0.5, 1/K}; COLA γ {1, 0.5}; CHOCO lr {0.01, 0.1, 1} |
| `solver` | SDCA vs GADGET and COLA vs CHOCO at low / mid / high local work per round |
| `topology` | ring, grid, connected cycle (c = 2, 3), complete, random k-out, all with K = 16 |
| `nodes` | K ∈ {10, 25, 50, 100} |
| `noniid` | iid, Dirichlet α = 1 and α = 0.1, label-sorted |
| `quantization` | full, lossless-sparse, QSGD 8-bit and 4-bit, top-10% |

---

## 4. Why some runs take so long

**The core reason: kernel models scale with (number of points)², linear models with (number of points).**

A linear model is one vector `w`, so scoring all the data costs about n × d. A kernel model is a list of support vectors, so scoring one point means comparing it with **every** support vector.

With λ = 1e-4 the problem is heavily regularized, so a large share of points become support vectors. The old gisette run had 2,737 of 4,800 points as support vectors (57%).

Going from gisette to a9a multiplies n by about 7, which multiplies kernel work by about **46×**. w8a and ijcnn1 multiply n by about 10, which multiplies kernel work by about **110×**.

**Where the time goes in `nonlinear`, assuming about half the points are support vectors:**

| step | how often | size on a9a | size on w8a / ijcnn1 |
|---|---|---|---|
| external-score rebuild at each merge | every cycle, every node | ~15k SVs × 3.3k pts ≈ 50M kernel values per node | ~22k × 5k ≈ 110M per node |
| evaluation: score on all training points | every 10 cycles, **× 10 nodes** | 16k × 32.5k ≈ 530M (**4 GB** array) | 25k × 50k ≈ 1.25B (**10 GB**) |
| evaluation: test and recovered test | same | 2 × 16k × 16k | ijcnn1: 2 × 25k × 92k ≈ 4.6B (**18 GB each**) |
| evaluation: ‖f‖² | same | 16k × 16k | 25k × 25k |

Three things make it worse:

1. **Evaluation repeats the same work 10 times.** Each of the 10 nodes rebuilds its score from scratch, even though they mostly hold the *same* support-vector blocks.
2. **Nothing is computed in chunks.** `kernel_scores` asks sklearn for the whole SVs × points matrix at once.
   - On w8a and ijcnn1 one matrix is larger than this machine's 15 GB RAM.
   - The run then swaps to disk, which looks like "forever", or gets killed.
3. **The runs don't converge; they plateau.** Because the absolute 1e-4 threshold is rarely met, every run goes at least 200 extra cycles of no progress before the plateau stop. The gisette run ended at gap 1.4e-4.

Rough cost: about 4 s per training cycle on a9a, plus about 2 min per evaluation, for about 1,000 cycles. That is **several hours per seed**, and ×5 seeds. On w8a and ijcnn1, memory runs out.

**The other slow models:**

| model | why it's slow | rough cost per seed |
|---|---|---|
| `ndsvm` | exact inner QP: a pure-Python coordinate loop with an O(N_j) update per step, to tolerance 1e-8; work grows like N_j² | gisette 2 min → a9a ~1.5 h, w8a/ijcnn1 ~3 h |
| `fed_ksvm` | pure-Python SDCA loop, ~10 × n steps per round; a slow 1/M averaging step, so the plateau comes late | a9a ~1.5 s/round, up to 5000 rounds (~2 h) |
| `cola` | pure-Python `sparse_cd`, ~n steps per round, ~10k rounds needed | a9a ~80 min, w8a/ijcnn1 ~2 h |
| `choco` | fixed 65k–100k rounds (each cheap) | minutes |
| `sdca`, `gadget`, `central_sdca` | fine | minutes |

**Fixes that give exactly the same numbers.** Items 1–4 are now implemented in `src/nonlinear_sdca/` (2026-11-04). Item 4 was done as "store `K_local` column-major" instead, because the RBF Gram is not bit-symmetric, so `K[i]` would change the last digit. Item 5 (numba) is not implemented:

1. **Nonlinear evaluation, cache per (origin, version):**
   - Compute each support-vector block's partial score on train and test once, and let every node that holds that version reuse it. That is about 10× less work.
   - Read `‖f‖²` off the training scores at the SV rows, which removes the SVs × SVs pass.
2. **Chunk every kernel product** in row blocks of about 2k, so memory stays bounded. This removes the swapping on w8a and ijcnn1.
3. **Incremental external scores:** recompute only the origins whose version changed, using only the α values that changed.
4. Use `K[i]` instead of `K[:, i]` in the SDCA step. The values are identical because K is symmetric, and row access is contiguous memory, so it's faster.
5. **numba** for the pure-Python inner loops (kernel SDCA, COLA `sparse_cd`, the Forero QP, Fed-KSVM block SDCA). Same arithmetic, roughly 20–100× faster. This needs `pip install numba`.

**Fixes that change what is measured (your call):**
- A larger `EVAL_EVERY` for the kernel models (fewer curve points).
- Compute recovered accuracy only at the end instead of at every evaluation.
- A relative gap threshold, such as gap/P < 1e-3, so runs converge instead of plateauing.
