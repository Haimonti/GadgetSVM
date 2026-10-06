$env:PYTHONWARNINGS = 'ignore'
& python -W ignore run_fdr_all.py `
  --data-dir 'C:\Users\remote\GadgetSVM-Lincoln\data\processed' `
  --out results/fdr_exact_two --cycles 5000 --jobs 2 `
  --covtype-path 'D:\GadgetSVM-sreekar\data\processed\covtype.libsvm.binary.scale' `
  --gisette-path 'D:\LocalLLM\data\fdr_exact\gisette.binary.scale' `
  cov gis
exit $LASTEXITCODE
