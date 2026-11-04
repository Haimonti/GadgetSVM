# svmrs — Consensus vs Federation of SDCA (linear + nonlinear)

## Philosophy

1. **One engine, many models.** `src/peersim_python/` is a learner-agnostic port of PeerSim's
   cycle-driven simulator. Our models and the P2P baselines are *tenant* packages on it.
2. **Fair comparison.** Every model sees the same dataset, test set, regularised problem
   (`LAMBDA`, with C = 1/(LAMBDA·n)) and metric definitions.
   - Our models + P2P baselines (GADGET, COLA, CHOCO) also share nodes, shards, seeded graph
     and budget — only the algorithm differs.
   - Forero and Fed-KSVM run standalone in their **own papers' setups**, fed our data.
3. **Faithful baselines.** Each baseline reproduces its paper/official code; deviations are
   listed in its `__init__.py` and checked by its `validate_*.py`.
4. **Models are model-only.** No file paths or plotting in model packages; wiring lives in `src/scripts/`.
5. **Train, then plot.** Training writes JSON; plotting reads JSON.
6. **Simulated decentralisation.** All nodes run in one process; decentralisation lives in the
   protocol (neighbour-only communication), not the runtime.

## Run

```bash
python src/scripts/extract_data.py                                   # once: datasets
python src/scripts/run_all.py                                        # EVERYTHING: linear + nonlinear + ablations
                                                                     #   on gisette, a9a, w8a, ijcnn1, 5 seeds
python src/scripts/main.py -m <models> -d <datasets> [--seeds 0 1 2 3 4] [--set KEY=VAL ...] [--no-plots]
python src/scripts/ablation_runner.py -a <ablation|all> -d <dataset> [--seeds ...] [--set ...]
python src/scripts/ablation_runner.py --replot <ablation_run_dir>   # rebuild summary + plots
```

- Models — linear: `sdca`, `central_sdca`, `gadget`, `cola`, `choco` · nonlinear: `nonlinear`, `ndsvm` (Forero), `fed_ksvm` · or `all`
- Datasets: `rcv`, `cov`, `gis`, `rsim`, `w8a`, `ijcnn`, `adult` (a9a), `webspam`, or `all`
- One model → individual plots. Two or more → comparative plots.

## Pipeline

```
main.py ─► model_picker.run_model(model, dataset, overrides)     per model × dataset × seed
             ├─ dataset_keywords → data_sharding.load_shards       (SHARDING: iid | dirichlet | label_sorted | kmeans)
             ├─ PeerSim tenant: <X>Simulation.run()   or   standalone: <X>Run.run()
             └─ results/json/<model>_<dataset>_seed<k>.json        (metrics + full config)
        ─► plot_analyzer.plot(models, datasets)
             ├─ 1 model   → results/individual_plots/<model>_run<N>_<dataset>_<date>/
             └─ 2+ models → results/comparative_plots/<date>/<dataset>/
                            comp_iter.png, comp_time.png (mean ± std over seeds; P(w) − P* panel),
                            comp_bars.png (std bars), summary.json — grouped P2P | federated | centralized

ablation_runner.py ─► same run_model per setting × model × seed
             └─ results/ablations/<ablation>_run<N>_<dataset>_<date>/
                  runs/<model>__<setting>__seed<k>.json, comparison.json, plots/, manifest.json
```

### One PeerSim cycle
```
gossip tenants (sdca, gadget, nonlinear):
  every node: merge inbox → local steps → push to GOSSIP_K random neighbours → evaluator/stop
synchronous tenants (cola, choco):
  every node computes from round-t values → SynchronousBarrier: deliver to ALL neighbours, commit → evaluator/stop
```
Stops on the convergence threshold, a plateau (`PLATEAU_PATIENCE`), or the round budget
(`ROUNDS`; COLA `COLA_ROUNDS`; CHOCO a fixed `CHOCO_MIN_ROUNDS`/`CHOCO_EPOCHS` budget).
Per-node SDCA latching: a converged node stops training but keeps relaying unseen versions,
and resumes if its gap rises again; the run ends when all nodes are below the threshold. Every JSON records `stop_reason` — `converged`, `plateau`,
or `budget` (= did not converge; `run_all.py` lists these at the end).

## Files

### `src/config.py`
The single `CONFIG` dict:
- **Setup:** paths, `DATASET`, `NUM_WORKERS`, `TOPOLOGY`/`GOSSIP_K`/`CYCLE_CONNECTIVITY`, `SEED`, `ROUNDS`, `EVAL_EVERY`
- **Problem/stopping:** `LAMBDA`, `GAP_THRESHOLD`, `STOP_RULE`, `STOP_ON_*`, `PLATEAU_*`
- **Experiments:** `SHARDING`/`DIRICHLET_ALPHA`, `COMPRESSOR`, `ABLATION_EPS`, `RUN_TAG`, `CENTRAL_GAP_TOL`
- **Per model:** `SDCA_*` · `KERNEL`/`RBF_GAMMA`/`POLY_*`, `PROJECTION_*`, `DSRR_TAU`, `DSRR_RECOVERY` · `COLA_*` · `CHOCO_*` · `NDSVM_*`/`FORERO_*` · `FEDKSVM_*`

### `src/scripts/` — orchestration + data
| File | Role |
|---|---|
| `run_all.py` | One command: linear comparison, nonlinear comparison, ablations (separate plots). |
| `main.py` | Entry point: models × datasets × seeds (+ `--set` overrides), then plots once. |
| `model_picker.py` | Model name → its PeerSim `Simulation` or standalone run; runs one leg; writes the JSON. |
| `plot_analyzer.py` | Individual / comparative plots, mean ± std across seeds, bar charts, suboptimality. |
| `ablations.py` | Ablation specs: settings, per-model overrides, skips (with reasons). |
| `ablation_runner.py` | Runs an ablation; writes runs/, comparison.json, plots/. |
| `dataset_keywords.py` | Keyword → dataset; builds it on first use. |
| `data_sharding.py` | Loads data, labels → ±1, shards (`SHARDING`) + shared test set. |
| `extract_data.py` | Downloads + preprocesses raw datasets into `data/processed/`. |
| `data_loader.py` | Legacy p2pfl loader (`own_network/` only). |

### `src/peersim_python/` — distribution engine (learner-agnostic)
| Part | Role |
|---|---|
| `core/`, `cdsim/` | PeerSim port: `Node`, `Network`, `CommonState`, `Control`, `CDProtocol`, `CDSimulator`. |
| `dynamics.py`, `graph.py` | Topologies `random_kout`, `ring`, `full`/`complete`, `star`, `mesh`, `grid`, `connected_cycle`; `make_topology(cfg)` used by every tenant. |
| `gossip_protocol.py`, `aggregator.py` | Gossip loop (inbox, push, merge) + merge strategies. |
| `sync_protocol.py` | Synchronous all-neighbour rounds (`SyncGossipProtocol`, `SynchronousBarrier`). |
| `mixing.py` | Metropolis mixing matrix `W` (+ spectral `beta`). |
| `compression.py` | Compressors: full, lossless sparse, top-k, random-k, QSGD — exact byte counts. |
| `idle_protocol.py`, `logger.py` | Neighbour list (`Linkable`); stdout logger. |

### Models
**Ours**

| Package | Model | Sends | Convergence |
|---|---|---|---|
| `src/linear_sdca/` | Gossip SDCA (linear) | Versioned per-origin dual contributions, accumulated | Duality gap |
| `src/nonlinear_sdca/` | Kernel SDCA (RBF / polynomial) over a DSRR projection | Projected support vectors + duals | Duality gap (τ-consistent) |

**Centralized baseline (no network)**

| Package | Model | Sends | Convergence |
|---|---|---|---|
| `src/linear_sdca/centralized.py` | Centralized SDCA: all data on one machine; also gives the reference optimum P* (liblinear) | — | Duality gap |

**Baselines on PeerSim (same setup as ours)**

| Package | Model | Sends | Convergence |
|---|---|---|---|
| `src/baselines/gadget/` | GADGET: Pegasos + weight averaging | Weight vector `w` | Objective |
| `src/baselines/cola/` | COLA (He et al. 2018): decentralised CoCoA, `W`-mixing | Estimate `v ≈ Ax`, to all neighbours | Duality gap |
| `src/baselines/choco/` | CHOCO-SGD + CHOCO-Gossip (Koloskova et al. 2019) | Compressed `Q(x − x̂)`, to all neighbours | Objective |

**Baselines standalone (own paper setup, our data)**

| Package | Model | Setup | Convergence |
|---|---|---|---|
| `src/baselines/forero/` | MoM-NDSVM (Forero 2010): ADMM consensus at anchors | Paper's random / ring / full network, broadcast | Objective |
| `src/baselines/fed_ksvm/` | Fed-KSVM (Zhou & Wang 2024): RFF + block boosting | 1 server + M devices | Objective |

**Inside the packages**
- `linear_sdca/`: `sdca_protocol.py`, `observers.py` (gap, `STOP_RULE`), `simulation.py`, `centralized.py`.
- `nonlinear_sdca/`: `kernel.py` (`make_kernel`: RBF + polynomial), `projection.py` (DSRR), `kernel_sdca_protocol.py`, `aggregator.py`, `observers.py` (gap + DSRR **recovery** → `accuracy_recovered`), `reference_kernel_sdca.py` (single-machine oracle). Efficiency, same numbers: `kernel_scores` builds the Gram in ≤256 MB row chunks (`MAX_KERNEL_ENTRIES`); each node keeps a private per-origin table of external score parts and re-scores only origins whose version changed; the evaluator scores each (origin, version) block once and reuses it for every node holding that version, and reads ‖f‖² off the training scores.
- `gadget/`: `pegasos_protocol.py`, `observers.py`, `simulation.py`.
- `cola/`, `choco/`: protocol, observers, simulation; `local_solver.py` (COLA's `svm.pyx` port); `validate_*.py` vs the official epfml code.
- `forero/`: `core.py` (math, errata E1–E3), `qp.py`, `anchors.py`, `network.py` (graph + partition), `algorithm.py` (the run), `validate_ndsvm.py`.
- `fed_ksvm/`: `rff.py`, `block_boosting.py`, `algorithm.py`, `validate_fedksvm.py`.
- `baselines/linear_evaluator.py`: shared evaluator for COLA / CHOCO.

### Ablations (`src/scripts/ablations.py`, linear models)
| Ablation | Settings |
|---|---|
| `stopping` | gap threshold 1e-2 … 1e-5; `STOP_RULE` node_global_gap / node_local_gap / mean_gap; native criteria |
| `stepsize` | λ 1e-3 / 1e-4 / 1e-5; SDCA step scale; COLA γ; CHOCO learning rate |
| `solver` | coordinate ascent vs SGD: (SDCA vs GADGET), (COLA vs CHOCO) at low / mid / high local work |
| `topology` | ring, grid, cycle c=2/3, complete, random k-out (K = 16) |
| `nodes` | K = 10, 25, 50, 100 |
| `noniid` | iid, Dirichlet α = 1 / 0.1, label-sorted |
| `quantization` | lossless (full, sparse) vs lossy (QSGD 8/4-bit, top-10%) |

### Other
- `src/evaluation/`: `visualizer.py` (plot helpers); `metrics.py` (legacy, `own_network/` only).
- Legacy, not on the PeerSim path: `src/model.py`, `src/own_network/`.
- `nonlinear_algo.tex`: write-up of the nonlinear method (kernel error, τ, recovery, RBF + polynomial).

## Metrics (every `EVAL_EVERY` cycles, per node)
| Key | Meaning |
|---|---|
| `round`, `wall_time` | Cycle; seconds since training started |
| `accuracy` | Node's own model on the shared test set |
| `hinge_loss` | Node's own model on all global training data (margin 1, comparable across models) |
| `comm_bytes` | Cumulative bytes sent by the node |
| `duality_gap` / `objective` | Convergence: dual methods / primal-only methods |
| `primal` | Primal objective → suboptimality P(w) − P* (P* solved exactly by liblinear in `central_sdca`) |
| `accuracy_recovered` | Nonlinear SDCA: DSRR-recovered (original-space) model |
| extras | e.g. `local_gap_est` (SDCA), `cv2` (COLA), `comm_bytes_p2p` / `test_risk` (Forero) |

## Layout
```
svmrs/
├── data/            raw/, processed/                                    (gitignored)
├── results/         json/, individual_plots/, comparative_plots/, ablations/   (gitignored)
├── src/
│   ├── config.py
│   ├── scripts/         orchestration, ablations, data
│   ├── peersim_python/  engine
│   ├── linear_sdca/     ours (linear) + centralized SDCA baseline
│   ├── nonlinear_sdca/  ours (kernel)
│   ├── baselines/       gadget/, cola/, choco/ (PeerSim) · forero/, fed_ksvm/ (standalone)
│   ├── evaluation/      plot helpers
│   ├── own_network/     legacy p2pfl runtime
│   └── model.py         legacy p2pfl model
├── nonlinear_algo.tex
└── requirements.txt
```

## Open issue: the stopping threshold must be relative, not absolute

**Current rule.** Dual models (SDCA, nonlinear SDCA, COLA) stop when the duality gap is below
`GAP_THRESHOLD = 1e-4`, an **absolute** number.

**Why that is wrong.** The gap only means something relative to the size of the objective
`P(w) = λ/2‖w‖² + mean hinge`, and that size depends heavily on the dataset:

| Dataset | Optimum P* | What gap 1e-4 means |
|---|---|---|
| a9a | 0.352 | 0.03% of the objective (strict, fine) |
| gisette | 0.00021 | ~50% of the objective (far too loose) |

gisette is almost linearly separable, so its optimum objective is tiny. Measured on the first
full run (5 seeds): our SDCA stopped at P(w) − P* ≈ 1.1e-4, i.e. ~50% above the optimum, while
centralized SDCA reaches a gap of 1e-8. A threshold that is strict on one dataset and meaningless
on another makes the "iterations / bytes to converge" comparison unfair across datasets.

**Fix (pending decision).** Use a *relative* gap, `gap / P(w) < ε` (e.g. ε = 1e-3, i.e. 0.1%),
for every dual model. It means the same thing on every dataset (a9a ≈ 3.5e-4, gisette ≈ 2e-7).
The plateau guard and round budgets stay as they are.

**Related finding to check first.** On gisette our SDCA's gap floors at ~1e-4 and stops
improving (plateau) instead of reaching the optimum. A likely cause is `SDCA_STEP_SCALE = 1.0`
(contributions added at full strength, without the 1/K damping CoCoA+ needs for safety), which
can make nodes overshoot near the optimum. To be tested (step scale 1 vs 0.5 vs 1/K on gisette)
before the relative threshold is switched on, since a stricter threshold would otherwise just
turn these runs into plateau / budget stops.
