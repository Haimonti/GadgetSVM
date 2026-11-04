"""Resumable 24-way learning and separate fixed-P-core timing matrix."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import ctypes
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import time
from run_unified_quality import METHODS,write


def hardware_topology():
    import psutil
    if os.name!='nt':raise RuntimeError('Benchmark must execute on the private Windows host')
    kernel=ctypes.WinDLL('kernel32',use_last_error=True);needed=ctypes.c_ulong()
    kernel.GetSystemCpuSetInformation(None,0,ctypes.byref(needed),None,0)
    buffer=ctypes.create_string_buffer(needed.value)
    if not kernel.GetSystemCpuSetInformation(buffer,needed,ctypes.byref(needed),None,0):
        raise ctypes.WinError(ctypes.get_last_error())
    raw=buffer.raw;offset=0;rows=[]
    while offset<needed.value:
        size,kind=struct.unpack_from('<II',raw,offset)
        if size<20:raise RuntimeError('Unexpected CPU-set record')
        if kind==0:
            id_,group,logical,core,cache,numa,efficiency,flags=struct.unpack_from('<IHBBBBBB',raw,offset+8)
            rows.append(dict(id=id_,group=group,logical=logical,core=core,efficiency_class=efficiency))
        offset+=size
    maximum=max(r['efficiency_class'] for r in rows)
    assert all(any(r['logical']==i and r['efficiency_class']==maximum for r in rows) for i in [10,11,12])
    return {'cpu_sets':rows,'Pcores':[r['logical'] for r in rows if r['efficiency_class']==maximum],
            'timing_affinity':[10,11,12],'logical_cpus':psutil.cpu_count(),
            'memory_bytes':psutil.virtual_memory().total}


def input_audit(expected,data_dir):
    seen={}
    for r in expected.values():
        for f in r['source_data']:
            if f['name'] in seen:continue
            p=data_dir/f['name'];h=hashlib.sha256()
            with p.open('rb') as stream:
                for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
            actual={'name':f['name'],'bytes':p.stat().st_size,'sha256':h.hexdigest()}
            assert actual==f,actual
            seen[f['name']]=actual
    return seen


def hashes():
    paths=[p for d in ['src','p2p','methods','cocoa'] for p in Path(d).rglob('*.py')]
    paths += [Path('run_unified_quality.py'),Path('run_unified_matrix.py')]
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def main(args):
    out=args.out;out.mkdir(parents=True,exist_ok=True)
    topology=hardware_topology();write(out/'hardware_topology.json',topology)
    expected=json.loads(Path('expected_data.json').read_text())
    write(out/'input_audit.json',input_audit(expected,args.data_dir))
    current_hashes=hashes();old=out/'source_hashes.json'
    if old.exists():assert json.loads(old.read_text())==current_hashes,'Mathematical source changed on resume'
    else:write(old,current_hashes)
    protocol={'version':'unified_quality_v1','datasets':list(expected),'seeds':list(range(42,47)),
        'methods':METHODS,'lambda':1e-4,'fit_intercept':False,'target':.001,
        'quality':'all worker primal minus same feasible reference dual <= target',
        'round_cap':50000,'evaluation_every':10,'topology':'ring','gossip_k':2,
        'local_steps':4096,'CHOCO_batch':128,'zero_score_label':-1,'learning_jobs':24,
        'timing_affinity':[10,11,12],'threads':3,'ssl_excluded':True}
    write(out/'protocol.json',protocol)
    env=os.environ.copy();env.update(PYTHONUTF8='1',SVM_COMPILED_LOCAL='1',GADGETSVM_DATA_DIR=str(args.data_dir))
    for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:env[k]='3'
    tasks=[(d,s,m) for d in expected for s in range(42,47) for m in METHODS]
    started=time.monotonic();completed=[];active={}
    def invoke(task):
        ds,seed,method=task;folder=out/'study'/ds/f'seed_{seed}';folder.mkdir(parents=True,exist_ok=True)
        name=method+('_timing' if args.phase=='timing' else '_metrics')
        path=folder/(name+'.json')
        if path.exists():
            r=json.loads(path.read_text());assert r['round_cap']==50000 and r['target']==.001
            if r['stop_reason']=='budget_quality_unattainable' or path.with_suffix('.npz').exists():return {'task':task,'status':'cached'}
        if time.monotonic()-started>args.launch_hours*3600:return {'task':task,'status':'deferred'}
        ref=out/'references'/ds/f'seed_{seed}'
        assert (ref/'reference.json').exists() and (ref/'budget_certificate.json').exists()
        cmd=[sys.executable,'-u','run_unified_quality.py',ds,'--method',method,
            '--seed',str(seed),'--out',str(folder),'--reference',str(ref),
            '--cycles','50000','--target','.001','--threads','3','--data-dir',str(args.data_dir)]
        if args.phase=='timing':cmd.append('--timing')
        active['/'.join(map(str,task))]=time.time()
        with (folder/(name+'.log')).open('w',encoding='utf8') as log:r=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
        active.pop('/'.join(map(str,task)),None)
        if r.returncode:return {'task':task,'status':'failed','returncode':r.returncode}
        value=json.loads(path.read_text())
        assert value['converged']==(value.get('max_quality_gap') is not None and value['max_quality_gap']<=.001)
        if args.phase=='timing' and value['phase']!='proof_preflight':
            hw=value['execution_hardware'];assert hw['cpu_affinity']==[10,11,12]
            assert all(p['num_threads']<=3 for p in hw['threadpools'])
        return {'task':task,'status':'complete','stop':value['stop_reason'],'rounds':value['rounds']}
    def progress():
        write(out/'progress.json',{'phase':args.phase,'complete':len(completed),'total':len(tasks),
            'active':dict(active),'results':completed,'jobs':24 if args.phase=='learning' else 1,
            'updated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
    progress()
    if args.phase=='learning':
        with ThreadPoolExecutor(max_workers=24) as ex:
            for future in as_completed([ex.submit(invoke,t) for t in tasks]):
                r=future.result();completed.append(r);progress();print(r,flush=True)
    else:
        # No other learner process may overlap the measured phase.
        import psutil
        for task in tasks:
            for p in psutil.process_iter(['pid','cmdline']):
                try:
                    if any(x in p.cmdline() for x in ['run_unified_quality.py','run_convergence_compare.py','trial.py']):
                        raise RuntimeError('Another training process would invalidate isolated timing: '+str(p.pid))
                except (psutil.NoSuchProcess,psutil.AccessDenied):pass
            r=invoke(task);completed.append(r);progress();print(r,flush=True)
    assert not any(r['status']=='failed' for r in completed),'A learner failed; inspect its log'
    done=all(r['status'] in ['complete','cached'] for r in completed)
    write(out/(args.phase+'_COMPLETE.json' if done else args.phase+'_CHECKPOINT.json'),
          {'phase':args.phase,'complete':done,'records':len(completed),'results':completed})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path('output'))
    p.add_argument('--phase',choices=['learning','timing'],required=True)
    p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--launch-hours',type=float,default=16.)
    main(p.parse_args())
