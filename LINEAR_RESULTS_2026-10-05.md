# Linear BDSVM versus SDCA: 2026-10-05

Eight paired runs completed. Each pair used the same loaded dataset and the
same ten node shards; the comparison changes only the optimization method.
The original RBF BDSVM code path and historical outputs were retained.

| Dataset | SDCA accuracy | Linear BDSVM accuracy | Difference (BDSVM − SDCA) | BDSVM stopping cycle |
| --- | ---: | ---: | ---: | ---: |
| a9a | 85.01% | 85.01% | +0.01 pp | 60 |
| covtype | 76.22% | 76.21% | −0.01 pp | 100 |
| gisette | 97.00% | 86.06% | −10.94 pp | 5000-cycle cap; not converged |
| ijcnn1 | 91.95% | 92.12% | +0.17 pp | 60 |
| rcv1 | 96.20% | 88.52% | −7.69 pp | 50 |
| real-sim | 96.80% | 89.12% | −7.68 pp | 50 |
| w8a | 90.41% | 98.67% | +8.26 pp | 70 |
| webspam | 91.49% | 93.20% | +1.71 pp | 60 |

The machine-readable source is `results/linear_summary_2026-10-05.csv`.
Each `results/linear_<dataset>_2026-10-05` directory contains method metrics
and paired plots. Cross-dataset plots are in
`results/plots_linear_2026-10-05`.

Interpretation: BDSVM remains a budgeted pre-image method. For datasets with
more features than its pre-image budget, its linear model is restricted to a
fixed subspace, while SDCA trains in the full feature space. BDSVM's other
hyperparameters are inherited from the RBF experiments and were not retuned.
The communication figures in the CSV are modelled byte counts, not measured
network traffic; the gisette BDSVM figure is especially large because it ran
to the cycle cap. This is a paired implementation comparison, not a claim
that the two optimizers use identical model capacity or tuning.

Data provenance and SHA-256 hashes are recorded in
`linear_data_manifest.json`; the current runs use the full 49,990-example
`ijcnn1` training set. Data archives are not committed. The source datasets
are described on the [official LIBSVM binary dataset page](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html).
