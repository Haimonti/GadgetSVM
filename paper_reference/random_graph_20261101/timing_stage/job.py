"""Prepared sequential replay controller; refuses to start before all learning audits."""
import os,sys,json,time,hashlib,zipfile,subprocess,traceback
from pathlib import Path
sys.stderr=sys.stdout
os.environ.update(PYTHONUTF8='1',SVM_COMPILED_LOCAL='1',NUMBA_NUM_THREADS='3')
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='3'
ROOT=Path.cwd().resolve();OUT=ROOT/'output';OUT.mkdir(exist_ok=True)
DATA=Path('C:/Users/remote/GadgetSVM-Lincoln/data/processed')
METHODS=['centralized','sdca','bdsvm','fedavg','cocoa','cocoa_plus','gadget','cola','choco']
# Validate actual frozen method spelling from the original executable.
sys.path.insert(0,str(ROOT/'engines/base'))
from run_unified_quality import METHODS as FROZEN_METHODS
METHODS=list(FROZEN_METHODS)
from run_unified_matrix import hardware_topology,input_audit

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2));temporary.replace(path)

def check_quiet(ignore_pids=()):
    import psutil
    bad=[]
    for p in psutil.process_iter(['pid','cmdline']):
        try:
            if p.pid in ignore_pids:continue
            names=[Path(str(x)).name for x in (p.info['cmdline'] or [])]
            if any(x in names for x in ['run_unified_quality.py','run_native_sparse_bdsvm.py','run_convergence_compare.py','trial.py','replay_case.py']):bad.append(p.pid)
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    assert not bad,'Concurrent learner would invalidate isolation: '+str(bad)

def engine_for(ds,method):
    if method!='bdsvm':return 'choco_w8a' if (ds,method)==('w8a','choco') else 'base'
    if ds=='gisette':return 'bd_gisette_v3'
    if ds in ['real-sim','rcv1']:return 'bd_big_v4'
    return 'bd_v1'

def paired(ds,seed,method):
    folder=OUT/'study'/ds/f'seed_{seed}';p=folder/(method+'_timing.json')
    environment=json.loads((folder/(method+'_runtime_environment.json')).read_text());assert environment['SVM_COMPILED_LOCAL']=='1' and environment['NUMBA_NUM_THREADS']=='3'
    monitor=json.loads((folder/(method+'_isolation_monitor.json')).read_text());assert not monitor['interference']
    value=json.loads(p.read_text());learn=json.loads((ROOT/'learning/study'/ds/f'seed_{seed}'/(method+'_metrics.json')).read_text())
    assert p.with_suffix('.npz').exists()
    assert value['phase']=='isolated_timing' and value['execution_hardware']['cpu_affinity']==[10,11,12]
    assert value['execution_hardware']['threads']==3
    assert all(t['num_threads']<=3 for t in value['execution_hardware']['threadpools'])
    for k in ['rounds','total_comm_bytes','train_digest','stop_reason','target','round_cap','test_evaluations']:
        assert value[k]==learn[k],(ds,seed,method,k,value[k],learn[k])
    assert value['converged'] and value['max_quality_gap']<=.001
    assert value['adjacency']==learn['adjacency']
    assert value['config']['TOPOLOGY']=='random_kout' and value['config']['GOSSIP_K']==3
    from independent_wire import verify
    verify(value)
    assert abs(value['max_quality_gap']-learn['max_quality_gap'])<1e-7
    if engine_for(ds,method)=='bd_big_v4':
        proof=json.loads((folder/'bdsvm_timing_jit_proof.json').read_text())
        assert proof['no_compilation_during_training'] and proof['threads_before']==proof['threads_after']==3
    return value

def checkpoint():
    with zipfile.ZipFile(ROOT/'result.partial.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in OUT.rglob('*'):
            if p.is_file():z.write(p,p.relative_to(ROOT).as_posix())
        for p in (ROOT/'engines').rglob('*.py'):z.write(p,'source_snapshot/'+p.relative_to(ROOT).as_posix())
        for name in ['job.py','bootstrap.py','replay_case.py','engine_manifest.json','independent_wire.py']:
            z.write(ROOT/name,'source_snapshot/'+name)
    (ROOT/'result.partial.zip').replace(ROOT/'result.zip')

try:
    approval=json.loads((ROOT/'learning/FORMAL_360_AUDIT_APPROVED.json').read_text())
    assert approval['passed'] and approval['records']==360
    expected=json.loads((ROOT/'learning/expected_data.json').read_text())
    assert len(expected)==8 and len(METHODS)==9
    for ds in expected:
        for seed in range(42,47):
            for method in METHODS:
                p=ROOT/'learning/study'/ds/f'seed_{seed}'/(method+'_metrics.json')
                r=json.loads(p.read_text())
                assert r['converged'] and r['max_quality_gap']<=.001 and p.with_suffix('.npz').exists()
    manifest=json.loads((ROOT/'engine_manifest.json').read_text())
    for engine,meta in manifest.items():
        for path,sha in meta['files'].items():assert hashlib.sha256((ROOT/'engines'/engine/path).read_bytes()).hexdigest()==sha
    check_quiet();topology=hardware_topology()
    import psutil
    non_timing_cores=[r['logical'] for r in topology['cpu_sets'] if r['logical'] not in topology['Pcores']]
    assert non_timing_cores
    psutil.Process().cpu_affinity([non_timing_cores[0]])  # Monitor/checkpoint work stays off measured P cores.
    write(OUT/'hardware_topology.json',topology)
    write(OUT/'input_audit.json',input_audit(expected,DATA))
    write(OUT/'engine_manifest.json',manifest);write(OUT/'learning_audit_approval.json',approval)
    tasks=[(ds,seed,m) for ds in expected for seed in range(42,47) for m in METHODS]
    start=time.monotonic();rows=[]
    for ds,seed,method in tasks:
        folder=OUT/'study'/ds/f'seed_{seed}';folder.mkdir(parents=True,exist_ok=True)
        if (folder/(method+'_timing.json')).exists():
            paired(ds,seed,method);rows.append(dict(dataset=ds,seed=seed,method=method,status='cached'));continue
        if time.monotonic()-start>12*3600:break
        check_quiet();engine=engine_for(ds,method)
        args=[ds,'--seed',str(seed),'--out',str(folder),'--reference',str(ROOT/'learning/references'/ds/f'seed_{seed}'),
              '--data-dir',str(DATA),'--cycles','50000','--threads','3','--timing']
        if method=='bdsvm':
            args+=['--rho','.5','--cg-rtol','1e-8','--weight-floor','1e-4']
            if engine=='bd_gisette_v3':args+=['--linear-solver','cholesky','--preconditioner','none','--matrix-storage','dense']
            if engine=='bd_big_v4':args+=['--linear-solver','cg','--preconditioner','none','--matrix-storage','csr','--matvec-backend','numba']
        else:args+=['--method',method,'--target','.001']
        write(OUT/'progress.json',dict(phase='isolated_timing',complete=len(rows),total=360,active=[ds,seed,method],updated_utc=time.time()))
        began=time.time()
        interference=[];samples=0
        with (folder/(method+'_timing.log')).open('w',encoding='utf8') as log:
            child=subprocess.Popen([sys.executable,str(ROOT/'replay_case.py'),str(ROOT/'engines'/engine),engine]+args,stdout=log,stderr=subprocess.STDOUT)
            while child.poll() is None:
                try:check_quiet(ignore_pids=[child.pid])
                except AssertionError as error:interference.append(dict(at=time.time(),error=str(error)))
                samples+=1;time.sleep(2)
        write(folder/(method+'_isolation_monitor.json'),dict(child_pid=child.pid,samples=samples,interference=interference))
        assert child.returncode==0,(ds,seed,method,child.returncode)
        assert not interference,'Timing invalid due to concurrent learner; archive before any retry'
        value=paired(ds,seed,method)
        write(folder/(method+'_replay_provenance.json'),dict(engine=engine,started_utc=began,ended_utc=time.time(),source_manifest=manifest[engine]))
        rows.append(dict(dataset=ds,seed=seed,method=method,status='complete',rounds=value['rounds']))
        write(OUT/'progress.json',dict(phase='isolated_timing',complete=len(rows),total=360,active=None,results=rows));checkpoint()
    write(OUT/('COMPLETE.json' if len(rows)==360 else 'CHECKPOINT.json'),dict(records=len(rows),complete=len(rows)==360,results=rows))
except Exception:
    write(OUT/'error.json',dict(error=traceback.format_exc()));print(traceback.format_exc(),flush=True)
finally:
    checkpoint();print('Isolated replay checkpoint packaged',flush=True)
