# Certified linear SVM experiments: ring and random graph

The completed studies compare centralized LinearSVC, gossip SDCA, linear BDSVM,
FedAvg, CoCoA, CoCoA+, GADGET, CoLA and CHOCO-SVM on eight datasets and seeds
42–46. The problem is mean training hinge loss plus `1e-4/2 * ||w||²`, without
an intercept. Every worker must attain objective minus a common feasible dual
lower bound at most 0.001. Evaluation is every ten rounds (also round one), with
a cap of 50,000 rounds. Test predictions are evaluated at the passing endpoint.

`run_unified_quality.py` provides the common evaluator and certificate creation;
`run_native_sparse_bdsvm.py` runs native full-span linear IRWLS. Root entrypoints
retain their ring configuration. The exact five engine versions used in the
random experiment are preserved under
`paper_reference/random_graph_20261101/stage/engines/`, with explicit SHA-256
manifest and topology-only diffs. Use the frozen engine matching the case:

| Engine | Cases |
| --- | --- |
| base | Non-BDSVM methods except w8a CHOCO |
| choco_w8a | w8a CHOCO, runtime initial LR override 10 |
| bd_v1 | BDSVM: a9a, w8a, ijcnn1, covtype, webspam |
| bd_gisette_v3 | BDSVM: gisette; dense Gram/Cholesky |
| bd_big_v4 | BDSVM: real-sim, rcv1; CSR/Numba CG |

The replay wrapper applies the w8a learning-rate override and records runtime
settings. Ring uses 10 nodes and k=2; random uses undirected random-k-out wiring
with k=3, fixed by seed. These differ in communication opportunities as well as
topology. Each seed uses the same graph across methods. Random graph audits
independently reproduce graph wiring and directed numeric payload accounting.

## Running individual cases

Supply the original processed dataset files; filenames and hashes are recorded
in the archived expected-data manifest. Data, certificates/models, logs and result
archives are intentionally not part of this code publication. From an engine
folder, first generate a topology-independent feasible dual reference:

```sh
python run_unified_quality.py a9a --method reference --seed 42 --out /absolute/ref \
  --reference /absolute/ref --data-dir /absolute/processed --threads 3
```

Then run a non-BDSVM case from the same engine:

```sh
python run_unified_quality.py a9a --method sdca --seed 42 --out /absolute/run \
  --reference /absolute/ref --data-dir /absolute/processed --threads 3 \
  --cycles 50000 --target .001
```

Use the frozen replay wrapper for the exact study overrides. Set
`SVM_COMPILED_LOCAL=1`, `NUMBA_NUM_THREADS=3`, and OMP/MKL/OpenBLAS/NumExpr thread
limits to 3 before imports. Native BDSVM uses full P=d, rho=.5, weight floor
1e-4 and solver tolerance 1e-8. GIS uses Cholesky/dense storage; the big engine
uses CG/no preconditioner/CSR/Numba. Other native cases retain v1 CG/Jacobi.
Reference weights are never learner warm starts.

Archived controllers contain workstation-specific orchestration paths and job
identifiers. They document the completed jobs; adapt data/output locations for
a new host. Generate references before starting their learning controller. The
bootstrap/resume scripts expect historical archives and are not fresh-run
entrypoints.

## Interpretation and validation

Parallel learning elapsed times are not speedups. Isolated replay uses the same
three verified Windows P cores [10,11,12], three threads, and an interference
monitor. Reported ratios describe one-host simulation, including observer
computation, not actual multi-machine network speedups. GADGET retains native
Python while several other local loops are compiled.

BDSVM charges its native dense float64 statistic payload, including zeros;
lossless CSR is a local storage optimization. Protocol bytes are not measured
NIC traffic. `p2p/full_linear_irls.py` was an abandoned extra-reduction prototype
and is not part of the formal study. Any occurrence inside a frozen engine is
historical provenance, not a formal entrypoint.

Independent audits recompute worker objectives, feasible dual bounds, test
predictions, exact graphs, byte schedules, runtime controls and paired endpoints.
All 360 learning cases and 360 isolated cases passed in each completed study.
The final figures show five-seed means and sample standard deviations, without
averaging only a successful subset. Refer to vendored author licenses and
provenance when reusing baseline implementations.
