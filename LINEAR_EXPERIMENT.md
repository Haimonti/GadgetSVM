# Linear BDSVM versus SDCA

`bdsvm_linear` is the existing gossip BDSVM and IRWLS solver with the kernel
changed from Gaussian RBF to the dot product. The original `bdsvm` method and
all existing results remain available. Both methods use the same `CONFIG`, and
`run_compare.py` loads and partitions each dataset once before passing the
identical shard objects to SDCA and linear BDSVM.

The linear BDSVM architecture uses seeded unit Gaussian pre-images. For
datasets with at most the original pre-image budget `P` features, `P` is reduced
to the feature count and these directions span the full linear feature space
with probability one. On higher-dimensional datasets the model is linear in a
fixed `P`-dimensional subspace, as required by BDSVM's budget. This restriction
matters when interpreting differences from full-feature SDCA. The remaining
BDSVM hyperparameters, including `C` and `ey_floor`, are inherited from the
RBF runs; they have not been re-tuned for the linear kernel.

Run all eight datasets with all available CPU threads allocated across dataset
processes:

```powershell
python run_linear_all.py --tag 2026-10-05
```

Use `--cycles 2 --tag smoke` for a short execution check. The runner writes
separate `results/linear_<dataset>_<tag>` directories and a summary CSV. It
keeps completed runs and refuses to overwrite a partially completed run. The
comparison figure in each directory is `plots/merged_linear_vs_sdca.png`.
Cross-dataset figures are in `results/plots_linear_<tag>`.

`linear_data_manifest.json` records SHA-256 values for the data files used.
The train/test split remains seed 42 with ten nodes: existing official train
and test files for rcv1, a9a, w8a, ijcnn1; an 80/20 seeded holdout for
covtype, gisette, real-sim, and webspam. The exact source filenames and
processed names are in `prepare_linear_data.py`.
