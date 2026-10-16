"""Run both remaining PeerSim methods on all eight full datasets, resumably."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from run_compare import DATASET_KEYWORDS

ROOT = Path(__file__).resolve().parent
ORDER = ['cov', 'gis', 'rsim', 'rcv', 'ijcnn', 'adult', 'w8a', 'webspam']
METHODS = ['fedssl_amc', 'fdr_svm']


def run_one(kw, args):
    ds = DATASET_KEYWORDS[kw]
    out = ROOT/'results'/f'additional_{ds}_{args.tag}'
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
              'NUMEXPR_NUM_THREADS']:
        env[k] = str(args.threads)
    env['MPLBACKEND'] = 'Agg'
    for method in METHODS:
        target = out/f'{method}_metrics.json'
        if target.exists():
            saved = json.loads(target.read_text())
            if saved['cycles_requested'] != args.cycles:
                raise RuntimeError(f'{out}: cycle budget differs; choose another tag')
            print(f'{ds}/{method}: keeping completed result', flush=True)
            continue
        cmd = [sys.executable, 'run_compare.py', kw, '--only', method,
               '--cycles', str(args.cycles), '--out', str(out), '--additional-protocol']
        print(f'{ds}/{method}: started', flush=True)
        with (out/f'{method}.log').open('w', encoding='utf-8') as log:
            subprocess.run(cmd, cwd=ROOT, env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        print(f'{ds}/{method}: completed', flush=True)
    # Reuse the prior full-data linear comparisons without rerunning baselines.
    baseline = ROOT/'results'/f'linear_{ds}_2026-10-05'
    for method in ['sdca', 'bdsvm_linear']:
        source = baseline/f'{method}_metrics.json'
        if source.exists():
            shutil.copy2(source, out/source.name)
    with (out/'plots.log').open('w', encoding='utf-8') as log:
        subprocess.run([sys.executable, 'plot_merged.py', str(out)], cwd=ROOT,
                       env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    return ds, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('datasets', nargs='*')
    ap.add_argument('--tag', default='2026-10-15')
    ap.add_argument('--cycles', type=int, default=5000)
    ap.add_argument('--jobs', type=int, default=8)
    ap.add_argument('--threads', type=int, default=3)
    args = ap.parse_args()
    selected = args.datasets or ORDER
    if any(k not in DATASET_KEYWORDS for k in selected):
        ap.error('unknown dataset keyword')
    done, errors = [], []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(run_one, kw, args): kw for kw in selected}
        for f in as_completed(futures):
            try:
                done.append(f.result())
            except Exception as exc:
                errors.append(f'{futures[f]}: {exc}')
                print(errors[-1], flush=True)
    rows = []
    for ds, out in sorted(done):
        for m in METHODS:
            d = json.loads((out/f'{m}_metrics.json').read_text())
            rows.append(dict(dataset=ds, method=m, accuracy=d['average_accuracy'],
                             cycles=d['stopped_at'], converged=all(
                                 x is not None for x in d['per_node_stop_cycle']),
                             communication_mb=d['total_comm_bytes']/1e6,
                             elapsed_seconds=d['elapsed_seconds']))
    summary = ROOT/'results'/f'additional_summary_{args.tag}.csv'
    with summary.open('w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=['dataset', 'method', 'accuracy', 'cycles',
                           'converged', 'communication_mb', 'elapsed_seconds'])
        w.writeheader()
        w.writerows(rows)
    if done:
        for metric in ['accuracy', 'hinge_loss', 'comm_bytes']:
            subprocess.run([sys.executable, 'plot_overview.py',
                            *[str(o) for _, o in sorted(done)], '--metric', metric,
                            '--out', str(ROOT/'results'/f'plots_additional_{args.tag}')],
                           cwd=ROOT, check=True)
    if errors:
        raise SystemExit('\n'.join(errors))
    print(f'All {len(rows)} runs and figures complete: {summary}', flush=True)


if __name__ == '__main__':
    main()
