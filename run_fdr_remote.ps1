param(
  [string]$DataDir = 'C:\Users\remote\GadgetSVM-Lincoln\data\processed',
  [string]$Out = 'results/fdr_comparison',
  [int]$Cycles = 5000,
  [int]$Jobs = 8,
  [string]$CovtypePath = '',
  [string]$GisettePath = '',
  [string[]]$Datasets = @()
)
$env:PYTHONWARNINGS = 'ignore'
$argsList = @('run_fdr_all.py', '--data-dir', $DataDir, '--out', $Out,
              '--cycles', $Cycles, '--jobs', $Jobs)
if ($CovtypePath) { $argsList += @('--covtype-path', $CovtypePath) }
if ($GisettePath) { $argsList += @('--gisette-path', $GisettePath) }
$argsList += $Datasets
& python -W ignore @argsList
exit $LASTEXITCODE
