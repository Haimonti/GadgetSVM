"""Single-dataset comparison with train-only stopping and final test evaluation.

A sequential pass times methods in isolation. If parallel runs are requested,
separate isolated reruns measure runtime ratios without competing jobs.
"""
import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import warnings

ROOT = Path(__file__).resolve().parent
METHODS = ['centralized', 'sdca', 'bdsvm', 'fedavg', 'cocoa', 'cocoa_plus']


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def child(args):
    import numpy as np
    import scipy.sparse as sp
    from src.config import CONFIG
    from src.data_sharding import load_shards
    from run_compare import DATASET_KEYWORDS, DATASET_PATH_KEYS

    CONFIG.update(DATASET=DATASET_KEYWORDS[args.dataset],
                  CONVERGENCE_PATIENCE=args.patience,
                  CONVERGENCE_WINDOW=args.window, CONVERGENCE_RTOL=args.rtol,
                  WARM_START=False,CONVERGENCE_RULE='adjacent_windows',
                  CONVERGENCE_STABILITY_WINDOW=args.patience)
    paths = [CONFIG[key] for key in DATASET_PATH_KEYS[CONFIG['DATASET']]]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    worker_data = load_shards(CONFIG)
    X = sp.vstack([p['X_csr'] for p in worker_data]).tocsr()
    y = np.concatenate([p['y'] for p in worker_data])
    Xt, yt = worker_data[0]['X_test'], worker_data[0]['y_test']
    method = args.child
    if method == 'fedavg':
        CONFIG['CONVERGENCE_RULE'] = 'adjacent_windows'
        CONFIG['CONVERGENCE_STABILITY_WINDOW'] = args.patience
    started = time.perf_counter()
    if method == 'centralized':
        from sklearn.svm import LinearSVC
        from sklearn.exceptions import ConvergenceWarning
        clf = LinearSVC(C=1 / (len(y) * CONFIG['LAMBDA']), loss='hinge',
                         fit_intercept=False, dual=True, max_iter=100000,
                         tol=1e-5, random_state=CONFIG['SEED'])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', ConvergenceWarning)
            clf.fit(X, y)
        elapsed = time.perf_counter() - started
        converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
        # All test evaluation happens after the training timer stops.
        accuracy = float(np.mean(clf.predict(Xt) == yt)) if converged else None
        hinge = float(np.maximum(0., 1 - y * clf.decision_function(X)).mean())
        result = dict(method=method, converged=converged,
            stop_reason='solver_tolerance' if converged else 'max_iterations',
            rounds=None, solver_iterations=int(clf.n_iter_),
            average_accuracy=accuracy, per_node_accuracy=[accuracy],
            final_train_hinge=hinge, total_comm_bytes=0, history=[],
            training_seconds=elapsed, local_solver_converged=converged,
            per_node_metrics=[])
    else:
        from src.peersim_python.core import Network, GeneralNode, CommonState
        from src.peersim_python.idle_protocol import IdleProtocol
        from src.peersim_python.observers import DataInitializer
        from src.peersim_python.cdsim import CDSimulator
        from src.train_convergence import TrainConvergenceEvaluator
        from src.peersim_python.simulation import Simulation
        from src.bdsvm_simulation import BDSVMSimulation, params_for
        from src.fedavg_simulation import FedAvgSimulation
        from src.cocoa_simulation import CoCoASimulation
        if method == 'sdca':
            assembly = Simulation(CONFIG, worker_data)
        elif method == 'bdsvm':
            assembly = BDSVMSimulation(CONFIG, worker_data,
                                       params=params_for(CONFIG['DATASET']))
        elif method == 'fedavg':
            assembly = FedAvgSimulation(CONFIG, worker_data)
        elif method in ('cocoa', 'cocoa_plus'):
            assembly = CoCoASimulation(CONFIG, worker_data, variant=method)
        else:
            raise ValueError('Use run_ssl_tabular.py for the explicit neural tabular adaptation')
        if assembly.sim is None:
            assembly.build()
        observer = TrainConvergenceEvaluator(1, CONFIG, args.cycles, started)
        assembly.sim.controls = [observer]
        assembly.sim.cycles = args.cycles
        rounds = assembly.sim.nextExperiment()
        elapsed = time.perf_counter() - started
        protos = [Network.get(i).getProtocol(1) for i in range(Network.size())]
        # FedSSL encoding or incomplete classifier aggregation is not convergence.
        local_ok = all(getattr(p, 'local_solver_converged', True) for p in protos)
        converged = observer.converged and local_ok
        accuracy = [float(p.accuracy()) for p in protos] if converged else []
        result = dict(method=method, converged=converged,
            stop_reason=('train_objective_plateau' if converged else
                         'local_solver_limit' if not local_ok else 'max_rounds'),
            rounds=rounds, average_accuracy=float(np.mean(accuracy)) if accuracy else None,
            per_node_accuracy=accuracy,
            final_train_hinge=observer.history[-1]['train_hinge'],
            total_comm_bytes=sum(p.comm_bytes for p in protos),
            per_node_comm_bytes=[p.comm_bytes for p in protos],
            per_node_metrics=[p.metrics for p in protos],
            history=observer.history, training_seconds=elapsed,
            local_solver_converged=local_ok)
        from src.peersim_python.communication import POLICY
        result['communication_policy']=POLICY
        result['schedule_policy']='event-hashed activation and neighbor selection, independent of learning RNG'
        result['adjacency']=[sorted(Network.get(i).getProtocol(0).getNeighbor(j).getIndex() for j in range(Network.get(i).getProtocol(0).degree())) for i in range(Network.size())]
        result['messages_sent']=sum(p.messages_sent for p in protos)
        prefix=[[(c,list(d)) for c,d,size in p.communication_trace[:100]] for p in protos]
        result['neighbor_schedule_first_100']=prefix
        result['neighbor_schedule_sha256_first_100']=hashlib.sha256(json.dumps(prefix).encode()).hexdigest()
        if method == 'bdsvm':
            result['parameters'] = assembly.params
    result.update(dataset=CONFIG['DATASET'], train_samples=len(y), test_samples=len(yt),
                  features=X.shape[1], threads=args.threads,
                  stopping=dict(patience=args.patience, window=args.window, rtol=args.rtol,
                      rule=CONFIG.get('CONVERGENCE_RULE', 'objective_patience'),
                      stability_window=CONFIG.get('CONVERGENCE_STABILITY_WINDOW'),
                      consecutive_checks=2 if method != 'centralized' else None),
                  config={k: str(v) if isinstance(v, Path) else v for k, v in CONFIG.items()},
                  source_data=[dict(name=p.name, bytes=p.stat().st_size,
                      sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths])
    filename = f'{method}_timing.json' if args.timing else f'{method}_metrics.json'
    result['test_evaluations'] = len(result['per_node_accuracy']) if converged else 0
    write_json(args.out / filename, result)
    print(f'{method}: {result["stop_reason"]}, rounds={result["rounds"]}, '
          f'accuracy={result["average_accuracy"]}, seconds={elapsed:.3f}', flush=True)


def invoke(args, method, timing=False):
    env = os.environ.copy()
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        env[key] = str(args.threads)
    env['GADGETSVM_DATA_DIR'] = str(args.data_dir)
    env['PYTHONUTF8'] = '1'
    env['PYTHONIOENCODING'] = 'utf-8'
    command = [sys.executable, '-u', str(Path(__file__).resolve()), args.dataset,
        '--child', method, '--out', str(args.out), '--cycles', str(args.cycles),
        '--patience', str(args.patience), '--window', str(args.window),
        '--rtol', str(args.rtol), '--threads', str(args.threads)]
    if timing:
        command.append('--timing')
    suffix = 'timing' if timing else 'run'
    with (args.out / f'{method}_{suffix}.log').open('w', encoding='utf-8') as log:
        status = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                stderr=subprocess.STDOUT).returncode
    print(f'{method} {suffix}: exit={status}', flush=True)
    if status:
        raise RuntimeError(f'{method} failed; see {method}_{suffix}.log')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('dataset', choices=['adult', 'cov', 'gis', 'rsim', 'rcv', 'w8a', 'ijcnn', 'webspam'])
    ap.add_argument('--data-dir', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--cycles', type=int, default=5000)
    ap.add_argument('--patience', type=int, default=50)
    ap.add_argument('--window', type=int, default=10)
    ap.add_argument('--rtol', type=float, default=1e-3)
    ap.add_argument('--jobs', type=int, default=len(METHODS))
    ap.add_argument('--threads', type=int, default=None)
    ap.add_argument('--child', choices=METHODS)
    ap.add_argument('--timing', action='store_true')
    ap.add_argument('--only', nargs='+', choices=METHODS, default=METHODS)
    ap.add_argument('--skip-isolated-timing', action='store_true',
                    help='Execution checks only; no speedup is reported.')
    args = ap.parse_args()
    if min(args.cycles, args.patience, args.window, args.jobs) < 1 or args.rtol < 0:
        ap.error('Positive cycles, patience, window and jobs required; rtol must be nonnegative.')
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.child:
        return child(args)
    if args.data_dir is None:
        ap.error('--data-dir is required')
    if any(args.out.glob('*_metrics.json')):
        ap.error('Output already contains metrics; choose a fresh directory.')
    args.jobs = min(args.jobs, len(args.only), os.cpu_count() or 1)
    args.threads = args.threads or max(1, (os.cpu_count() or 1) // args.jobs)
    print(f'parallel jobs={args.jobs}, threads/job={args.threads}, '
          f'logical CPUs={os.cpu_count()}', flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(invoke, args, method) for method in args.only]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    # A single-worker pass already times each algorithm without overlap.
    # Repeating it would add no isolation and doubles the preview run time.
    sequential_timing = args.jobs == 1 and not args.skip_isolated_timing
    if not args.skip_isolated_timing and not sequential_timing:
        for method in args.only:
            invoke(args, method, timing=True)
    rows = []
    for method in args.only:
        path = args.out / f'{method}_metrics.json'
        result = json.loads(path.read_text(encoding='utf-8'))
        if sequential_timing:
            result['isolated_training_seconds'] = result['training_seconds']
            result['timing_protocol'] = 'sequential; equal thread budget; single-host simulation'
        timing_path = args.out / f'{method}_timing.json'
        if timing_path.exists():
            timing = json.loads(timing_path.read_text(encoding='utf-8'))
            if (timing['rounds'] != result['rounds'] or
                timing['total_comm_bytes'] != result['total_comm_bytes'] or
                (timing['average_accuracy'] is None) != (result['average_accuracy'] is None) or
                (timing['average_accuracy'] is not None and
                 abs(timing['average_accuracy'] - result['average_accuracy']) > 1e-6)):
                raise RuntimeError(f'{method}: isolated rerun disagrees with parallel result')
            result['isolated_training_seconds'] = timing['training_seconds']
            write_json(path, result)
        rows.append(result)
    baseline = next((r for r in rows if r['method'] == 'centralized'), None)
    for result in rows:
        result['runtime_ratio_at_stop'] = (
            baseline['isolated_training_seconds'] / result['isolated_training_seconds']
            if baseline and 'isolated_training_seconds' in baseline
            and 'isolated_training_seconds' in result else None)
        result['speedup_vs_centralized'] = (
            baseline['isolated_training_seconds'] / result['isolated_training_seconds']
            if baseline and baseline['converged'] and result['converged']
            and 'isolated_training_seconds' in baseline
            and 'isolated_training_seconds' in result else None)
        write_json(args.out / f'{result["method"]}_metrics.json', result)
    fields = ['method', 'converged', 'stop_reason', 'rounds', 'average_accuracy',
              'final_train_hinge', 'total_comm_bytes', 'training_seconds',
              'isolated_training_seconds', 'runtime_ratio_at_stop', 'speedup_vs_centralized']
    with (args.out / 'summary.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    from plot_convergence_compare import plot_results
    plot_results(args.out)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        # Keep traceback on stdout for the PowerShell compute wrapper.
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)
