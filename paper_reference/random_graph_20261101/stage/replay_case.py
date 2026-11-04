"""Replay a frozen engine; add only external timing/JIT evidence, never updates."""
import os,sys,json,runpy
from pathlib import Path
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='3'
os.environ['NUMBA_NUM_THREADS']='3'  # Limit both initialized pool capacity and active mask.
os.environ['SVM_COMPILED_LOCAL']='1'
engine=Path(sys.argv[1]).resolve();kind=sys.argv[2];arguments=sys.argv[3:]
os.chdir(engine);sys.path.insert(0,str(engine))
proof=None
if kind=='bd_big_v4':
    import psutil
    
    if '--timing' in arguments:psutil.Process().cpu_affinity([10,11,12])
    from numba import get_num_threads
    from p2p.native_csr_matvec import warmup,csr_matvec
    warmup(3)
    proof=dict(threads_before=get_num_threads(),signatures_before=[str(s) for s in csr_matvec.signatures])
    assert proof['threads_before']==3
if kind.startswith('bd_'):
    sys.argv=['run_native_sparse_bdsvm.py']+arguments
    runpy.run_path(str(engine/'run_native_sparse_bdsvm.py'),run_name='__main__')
else:
    import run_unified_quality as q
    if kind=='choco_w8a':q.CHOCO_LR['w8a']=10.
    sys.argv=['run_unified_quality.py']+arguments
    q.main()
if proof is not None:
    proof.update(threads_after=get_num_threads(),signatures_after=[str(s) for s in csr_matvec.signatures])
    proof['no_compilation_during_training']=proof['signatures_before']==proof['signatures_after']
    assert proof['threads_after']==3 and proof['no_compilation_during_training'],'Unexpected Numba specialization during timed run; discard timing and investigate warmup'
    output=Path(arguments[arguments.index('--out')+1]);assert output.is_absolute()
    (output/'bdsvm_timing_jit_proof.json').write_text(json.dumps(proof,indent=2))

output=Path(arguments[arguments.index('--out')+1]);assert output.is_absolute()
(output/(('bdsvm' if kind.startswith('bd_') else arguments[arguments.index('--method')+1])+'_runtime_environment.json')).write_text(json.dumps({k:os.environ.get(k) for k in ['SVM_COMPILED_LOCAL','NUMBA_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']},indent=2))
