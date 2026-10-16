# FedSSL-AMC and FDR-SVM PeerSim experiments

Run all eight full datasets and produce separate figures for the two methods:

```powershell
python run_additional_all.py --tag 2026-10-15 --jobs 8 --threads 3
```

The runner resumes completed method JSONs. It uses the same `src.config.CONFIG`
and `src.data_sharding.load_shards` as previous experiments: seed 42, ten nodes,
random-k-out topology with k=3, shared test set, evaluation every ten cycles,
and a 5000-cycle cap. No sample limit or alternative data split is used.
The engine is the repository's Python PeerSim cycle-driven port
(`src.peersim_python.cdsim.CDSimulator`), as in the earlier experiments.
The batch runner explicitly selects `--additional-protocol` for this FDR
variant. The existing `run_fdr_all.py` and default `run_compare.py --only fdr_svm`
continue using the FDR implementation already on the remote Lincoln branch.

Outputs:

- `results/additional_<dataset>_<tag>/<method>_metrics.json`: node histories,
  configuration, parameters, accuracy, simulated communication and termination.
- `results/additional_<dataset>_<tag>/plots/merged_with_fedssl_amc.png` and
  `merged_with_fdr_svm.png`: separate figures, each compared with the prior
  full-data SDCA and linear BDSVM results of 2026-10-05. Each metric also gets
  its own PNG, using the same existing plotting code and mean +/- std ribbons.
- `results/plots_additional_<tag>/overview_*_fedssl_amc.png` and
  `overview_*_fdr_svm.png`: separate eight-dataset overview pages.
- `results/additional_summary_<tag>.csv`: both methods, every dataset.

These are ports of the existing repository algorithms, not independently
verified reproductions of named papers. FedSSL uses local truncated SVD,
gossip/QR encoder averaging for ten cycles, and local LinearSVC (C=1,
64 components reduced when necessary). Frozen encoders differ between nodes;
classifiers are therefore lifted into original feature coordinates before
gossip, including their intercepts. This changes classifier payload size to
`4*d + 8` bytes. Fixed local sample-count weights avoid age overflow.

FDR keeps the existing quadratic robustness surrogate (lambda + eps),
eps=1/sqrt(local sample count), rho=1 and neighbourhood ADMM messages.
It does not implement or claim the exact eps*norm(w) Wasserstein objective.
The local solve uses the shared 4096-step work budget. A compiled loop matches
the original sparse Pegasos recursion and random row sequence; numerical
regression checks are in `test_additional.py`.
For small dense shards, the same update is represented through a local row-Gram
matrix instead of feature coordinates. It preserves row draws and the proximal
objective; equivalence was also checked on twelve 4096-step solves using the
actual gisette data. This avoids the cost of thousands of long feature dot
products without reducing solver work or the cycle budget.

Both methods stop only when all nodes' relative parameter changes are below
1e-3 for three consecutive evaluation windows. SSL cannot stop during encoder
training. This is an iterate-stability rule, not a duality-gap certificate;
the dual and gap remain undefined. Runs reaching 5000 without satisfying the
rule are explicitly labelled capped/not converged. Communication is modelled
payload volume, not measured network traffic.

Quick pipeline verification (kept separate from full experiment results):

```powershell
python -m unittest test_additional.py
python run_additional_all.py adult --cycles 31 --tag smoke_2026-10-15 --jobs 1
```

Validate every completed run, verify every PNG and dataset hash, and create
one archive per algorithm:

```powershell
python package_additional_results.py --tag 2026-10-15
```

For two image-only archives with exactly eight four-metric figures each:

```powershell
python package_eight_figures.py --tag 2026-10-15
```

Every figure includes the existing SDCA and linear BDSVM baselines. The bottom
line explicitly reports the named algorithm's total communication summed over
all ten nodes, in bytes and decimal MB. The communication panel retains the
existing mean-per-node convention.
