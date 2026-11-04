import sys,json,io,zipfile,hashlib,traceback
from pathlib import Path
sys.stderr=sys.stdout
from independent_wire import verify
out={}
try:
 import psutil
 assert not any(any(Path(str(a)).name=='replay_case.py' for a in p.info['cmdline'] or []) for p in psutil.process_iter(['cmdline']))
 def inner(id):
  p=Path('D:/LocalLLM/jobs/incoming')/(id+'.result.zip')
  raw=p.read_bytes();sha=hashlib.sha256(raw).hexdigest()
  with zipfile.ZipFile(io.BytesIO(raw)) as z:assert z.testzip() is None;data=z.read('result.zip')
  z=zipfile.ZipFile(io.BytesIO(data));assert z.testzip() is None;return z,sha
 old,oldsha=inner('08d2573d-fcb0-426a-8c35-c51f07771891');learn,learnsha=inner('c90c54dc-dc9c-40f0-aa2b-1e2f069aefd4')
 for n in ['output/error.json','output/bootstrap_error.json']:assert n not in old.namelist(),old.read(n).decode() if n in old.namelist() else ''
 checkpoint=json.loads(old.read('output/CHECKPOINT.json')) if 'output/CHECKPOINT.json' in old.namelist() else json.loads(old.read('output/COMPLETE.json'))
 manifest=json.loads(old.read('source_snapshot/engine_manifest.json'))
 for e,v in manifest.items():
  for f,sha in v['files'].items():assert hashlib.sha256(old.read('source_snapshot/engines/'+e+'/'+f)).hexdigest()==sha
 for r in json.loads(Path('formal_learning_360_manifest.json').read_text())['records']:
  p=f"output/study/{r['dataset']}/seed_{r['seed']}/{r['method']}_metrics"
  assert hashlib.sha256(learn.read(p+'.json')).hexdigest()==r['metric_sha256']
  assert hashlib.sha256(learn.read(p+'.npz')).hexdigest()==r['weights_sha256']
 cached=[]
 for n in old.namelist():
  if not n.startswith('output/study/') or not n.endswith('_timing.json'):continue
  r=json.loads(old.read(n));m=r['method'];folder=n.rsplit('/',1)[0]+'/'
  l=json.loads(learn.read(folder+m+'_metrics.json'))
  for k in ['rounds','total_comm_bytes','train_digest','stop_reason','target','round_cap','test_evaluations','adjacency']:assert r[k]==l[k],(n,k)
  assert r['converged'] and abs(r['max_quality_gap']-l['max_quality_gap'])<1e-7
  verify(r);assert r['phase']=='isolated_timing' and r['execution_hardware']['cpu_affinity']==[10,11,12]
  assert r['execution_hardware']['threads']==3 and all(p['num_threads']<=3 for p in r['execution_hardware']['threadpools'])
  assert not json.loads(old.read(folder+m+'_isolation_monitor.json'))['interference']
  env=json.loads(old.read(folder+m+'_runtime_environment.json'));assert env['SVM_COMPILED_LOCAL']=='1' and env['NUMBA_NUM_THREADS']=='3'
  proof=folder+'bdsvm_timing_jit_proof.json'
  if m=='bdsvm' and r['parameters'].get('matvec_backend')=='numba':
   j=json.loads(old.read(proof));assert j['no_compilation_during_training'] and j['signatures_before']==j['signatures_after'] and j['threads_before']==j['threads_after']==3
  cached.append(dict(path=n,metric_sha256=hashlib.sha256(old.read(n)).hexdigest(),weights_sha256=hashlib.sha256(old.read(n[:-5]+'.npz')).hexdigest()))
 assert len(cached)==checkpoint['records']
 out=dict(passed=True,records=len(cached),complete=checkpoint['complete'],timing_outer_sha256=oldsha,learning_outer_sha256=learnsha,cached=cached,source_sha={n:hashlib.sha256(old.read('source_snapshot/'+n)).hexdigest() for n in ['job.py','replay_case.py','engine_manifest.json','independent_wire.py']})
 print(json.dumps({k:v for k,v in out.items() if k not in ['cached','source_sha']}),flush=True)
except Exception:out={'passed':False,'error':traceback.format_exc()};print(traceback.format_exc(),flush=True)
Path('checkpoint_audit.json').write_text(json.dumps(out,indent=2))
with zipfile.ZipFile('result.zip','w',zipfile.ZIP_DEFLATED) as z:z.write('checkpoint_audit.json')
