# Unified linear-SVM training quality — authorized 2026-10-29

The user's latest request replaces G50 and each method's native stopping rule
with one training-quality requirement for all eight datasets and nine methods.
The final user deliverable is a ZIP containing exactly eight PNG files.

## Shared problem and quality certificate

Use unchanged dataset files, the same seeded training/test partitions, lambda
`1e-4`, and **no intercept** for every method:

`J(w) = mean_i max(0, 1-y_i*x_i^T*w) + lambda/2 * ||w||^2`.

For each dataset and seed, first obtain a high-accuracy centralized reference
and a feasible dual vector `a in [0,1]^N`. Its lower bound is

`D(a) = mean(a) - ||X^T(y*a)||^2/(2*lambda*N^2)`.

The reference stores both `J_ref` and `D_ref`; desired reference gap is at most
`1e-5`. A reference with gap above `1e-4` is rejected. This is not an assumed
exact optimum: weak duality makes `J(w)-D_ref` an upper bound on actual primal
suboptimality for **any** feasible reported model, including primal-only SGD.

At round 1 and every 10 rounds, evaluate every worker model on all training
shards. Stop only when **max_worker[J(w_worker)-D_ref] <= 1e-3**. This is a
full-training-set calculation, not a minibatch estimate. No G50 stationarity
requirement and no objective-change-only native stop remains active. The common
maximum is 50,000 gossip rounds. Reference generation and feasibility analysis
are offline benchmark preparation, excluded from every training timer.

## Algorithms and aligned settings

- Centralized: LinearSVC, hinge, no intercept, `C=1/(lambda*N)`, initial solver
  tolerance `1e-5`. Its returned model must also pass the common certificate;
  any finer-tolerance retry is included in its training time.
- SDCA, CoCoA, CoCoA+, CoLA: retain their update and communication schedules,
  4,096 local sample/coordinate draws per worker per gossip round.
- FedAvg: retain the cumulative-step Pegasos recurrence and 4,096 local draws.
- GADGET: pinned `Haimonti/GadgetSVM` source commit
  `ccc7e62a4e67b6c12dd45d13a366f91db5c98864`, 4,096 local steps. Its original
  vector transport and RNG-driven activation loop are retained in private
  modules under `src/baselines/gadget/`; only import paths change. **All peers
  remain active until the shared certificate passes or the shared cap is hit.**
  Previous per-peer native objective-change shutdown is disabled.
- CHOCO-SVM: same author compression/consensus and Bottou learning-rate formula,
  hinge adaptation, batch 128, top-20%, consensus step 0.1. Initial learning
  rates frozen from training-only internal validation: w8a 10, gisette .1,
  real-sim 10, rcv1 1, webspam 10; other datasets retain .1. W8a was revised
  after the original .1 setting capped: on a fixed stratified 80/20 split of
  training data (seed1042), after equal 5000-round budgets, validation mean
  objectives were .166506/.153909/.153245 for initial rates .1/1/10. Freeze
  10 and retain final iterates. Five full-training seeds42–46 then all passed
  the unchanged .001 certificate in 3100/3510/3320/3700/3400 rounds; independent
  model, dual, test-score and byte audits passed. Isolated timings still pending.
  The active original matrix uses a frozen old source and is not rewritten;
  archive its old w8a-CHOCO rows and replace only these five rows with explicit
  configuration/source provenance before final reporting. Gisette uses the author's
  linearly weighted output estimator (tau=1), selected by training-internal
  validation; other datasets retain final iterates. This changes the reported
  estimator without feeding an average into gradients or messages. A small
  independent test verified equal raw trajectories and byte counts across
  output-estimator modes. These are explicitly SVM adaptations, not an
  assertion that the author's logistic task was an SVM experiment.
- Linear BDSVM: the old pre-image budgets are **under revision at the user's
  request (2026-10-30)**. They came from the previous kernel-oriented tuning,
  not a paper requirement for this full-linear task. In particular, a fixed
  random P=1000 row space excludes models needed by real-sim and rcv1 at the
  current target; simply increasing the round limit cannot remove that error.
  Keep `C=1/(2*lambda*N)` and b=0 for the shared objective. Do not report those
  old-budget proof rows as the final revised BDSVM experiment.

### BDSVM budget revision under verification

The prototype `p2p/full_linear_irls.py` uses an implicit full linear basis
(P=d), so its model class includes every comparator's linear model. This
removes the representation barrier without storing d-by-d matrices. Distributed
preconditioned conjugate gradients solves the original IRWLS weighted least
squares equation using shard-local matrix-vector products. A test compares the
actual distributed update against a dense direct solution, and separate tests
check ring reductions, empty/unequal chunks, every byte, and analytic optimum.

The user requires native communication schedules. This prototype is diagnostic
only; promoting it into the final study requires an explicit user choice.
Numerical equivalence does not establish communication-protocol equivalence.

This is an explicit **matrix-free ring adaptation** of linear IRWLS, not the
paper's matrix-upload communication protocol or the previous asynchronous
statistics-flooding implementation. It is only a prototype until full-data
pilots pass. Do not merge it as a silent numerical refactor. All inner one-hop
ring messages and steps count toward communication and rounds, including RHS,
preconditioner, residual and CG reductions. For d-dimensional vectors and ten
workers, every physical ring step sends ten packets and `8*d+160` numeric
bytes globally, including two eight-byte numeric tags per packet. One complete
ring reduction takes 18 physical steps. All local CG work and observer work
are included in training time. Quality still uses the same independent gate;
inner CG or native iterate-change convergence cannot certify the final model.

The first training-only pilots use seeds42 on a9a, real-sim, rcv1 and gisette,
with rho=.5/.9, numerical residual floor1e-4, CG relative tolerance1e-5 and the
same 50000 physical-round cap. They do not load/use final test data. Freeze
parameters only after inspection; any departures from paper update/wire
rules must remain explicit. No finite cap or budget alone guarantees that all
algorithms will reach the required quality.

## Correctness repairs discovered by the certified reference

Zero feature rows have constant hinge loss 1 and dual optimum unsigned alpha=1.
Skipping their dual coordinates leaves a permanent certificate error although
primal weights do not change. Fix this in Python and compiled SDCA, CoCoA, and
CoLA. FedAvg must still execute regularization shrink when the sampled feature
row is zero; its hinge gradient is zero, but its full gradient is not. Both
backends now do so. These repairs preserve draw counts and communication rules.

Use one prediction tie convention across all reported models: **score > 0 is
+1; score <= 0 is -1**, matching LinearSVC. Previous >=0 code disagreed with
LinearSVC on zero rows. Never use test labels to choose this convention or to
stop/tune training. Test predictions are computed once, after the timer stops,
only for runs that pass the common quality gate.

## Hardware, reporting, and delivery

Datasets: a9a, w8a, ijcnn1, real-sim, gisette, covtype, rcv1, webspam.
Methods: centralized, SDCA, linear BDSVM, FedAvg, CoCoA, CoCoA+, GADGET, CoLA,
CHOCO-SVM. FedSSL is excluded. Seeds 42–46, means and sample SD (ddof=1).

Reference and learning trials run up to 24 in parallel on the private Windows
host. Isolated timing is separate, on the same verified three P cores
[10,11,12], with three threads, without overlapping learners. Overlapping
elapsed times are never reported as isolated speedups. GADGET retains native
Python loops while several other local solvers are compiled; figures must
state this implementation limitation.

Numeric directed payloads across all workers, MB=1e6 bytes, exclude headers and
offline-observer traffic; include compressed coordinate indices. Report
intrinsic local compute budgets rather than changing update schedules.

Six-panel reference layout: final test accuracy at target, communication,
training hinge curve, rounds, paired centralized/method time ratio, and
runtime/status table. Accuracy and time-to-target ratios are N/A if any of the
five seeds does not reach the common target; never average only successful
seeds. Capped-run costs remain explicitly labeled costs at the limit.
Mathematically unattainable budget costs are N/A, never misleading zero bars.

Freeze source/configuration and input hashes, independently recompute final
training objectives and test accuracies, verify all certified dual bounds,
matched isolated runs, thread/core metadata, and all eight rendered images.
The final ZIP must have exactly eight root-level `.png` entries, with no other
files or folders. Display **GADGET**, without the source-branch suffix. Retain
raw data and audit files locally for reproducibility, not in that user ZIP.

## Native communication-preserving storage path under verification (2026-10-30)

`p2p/native_sparse_bdsvm.py` is distinct from the matrix-free ring-reduction
diagnostic above. For full P=d, identity coordinates are an orthogonal change
from the full orthonormalized random linear basis. Retain the native versioned
C_k,d_k statistic gossip, activation and update schedule. Store the actual
local/received matrices losslessly in CSR and solve their summed equation by
local PCG. The solver uses only received sufficient statistics; it does not
access other workers' rows or introduce inner communication. Account each
matrix on the wire as its FULL dense float64 array, including every zero.
Thus a received entry costs `8*(d*d+d)+24` bytes, including origin/version/n.
Internal CSR storage must never be reported as a communication compression.

Windows tests compare 20 complete native gossip rounds with the original dense
full-basis implementation: physical models agree within 1e-8 and every round's
byte count agrees exactly. Full-data structural measurements bound one entire
generation of ten local CSR matrices at 1.96 GB (real-sim) and .966 GB (rcv1),
with global sums .726 GB and .413 GB, respectively. Versions and solver buffers
still require measured memory checks. Two training-only full-data pilots are
running with rho=.9, residual floor1e-4, PCG rtol1e-8 and diagnostic cap100 native
rounds, after an actual-driver one-round export smoke. This is NOT yet the
final frozen BDSVM experiment. Require the unchanged common quality certificate,
full native byte ledger, memory checks and remaining-dataset validation before
formal five-seed runs. The formal cap remains 50000.
