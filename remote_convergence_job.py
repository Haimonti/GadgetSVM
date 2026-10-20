"""Windows compute entry point; route all diagnostics to stdout."""
import os
from pathlib import Path
import subprocess
import sys
import traceback
import zipfile


def main():
    # Set before numerical-library imports, including the test process.
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '3'
    os.environ['PYTHONUTF8'] = '1'
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    import psutil
    print('logical CPUs:', os.cpu_count(), 'available RAM GiB:',
          psutil.virtual_memory().available / 2**30, flush=True)
    mode = sys.argv[1]
    if mode == 'check':
        result = subprocess.run([sys.executable, '-m', 'unittest', 'discover',
            '-s', 'tests', '-v'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding='utf-8')
        print(result.stdout, flush=True)
        return result.returncode
    out = ('results/a9a_smoke_v2' if mode == 'smoke' else
           'results/a9a_2026-10-19_preview' if mode == 'preview' else
           'results/a9a_2026-10-16')
    command = [sys.executable, '-u', 'run_convergence_compare.py', 'adult',
        '--data-dir', 'C:/Users/remote/GadgetSVM-Lincoln/data/processed',
        '--out', out, '--jobs', '1' if mode == 'preview' else '8', '--threads', '3', '--cycles',
        '80' if mode == 'smoke' else '5000']
    if mode == 'smoke':
        command.append('--skip-isolated-timing')
    result = subprocess.run(command, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, encoding='utf-8')
    print(result.stdout, flush=True)
    if result.returncode:
        for path in Path(out).glob('*.log'):
            print(path.name, path.read_text(encoding='utf-8', errors='replace')[-4000:], flush=True)
    # POSIX member names avoid Windows backslashes becoming literal filenames
    # when the outer compute bundle is extracted on macOS.
    with zipfile.ZipFile('artifacts.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in Path(out).rglob('*'):
            if path.is_file():
                archive.write(path, path.as_posix())
    return result.returncode


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)
