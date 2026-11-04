import sys,os,json,io,zipfile,hashlib,shutil,traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
sys.stderr=sys.stdout
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ[k]='3'
OUT=Path('audit_output');OUT.mkdir(exist_ok=True)
try:
 import psutil
 for p in psutil.process_iter(['pid','cmdline']):
  if p.pid==os.getpid():continue
  assert not any(Path(str(a)).name in ['replay_case.py','run_unified_quality.py','run_native_sparse_bdsvm.py'] for a in p.info['cmdline'] or []),p.pid
 p=Path('D:/LocalLLM/jobs/incoming/c90c54dc-dc9c-40f0-aa2b-1e2f069aefd4.result.zip')
 with zipfile.ZipFile(p) as z:
  assert z.testzip() is None;raw=z.read('result.zip')
 with zipfile.ZipFile(io.BytesIO(raw)) as z:
  assert z.testzip() is None;z.extractall('learning')
 base=Path('learning/engines/base').resolve();sys.path.insert(0,str(base))
 import audit_random_quality as a
 from run_unified_matrix import input_audit
 root=Path('learning/output');expected=json.loads(Path('learning/source_snapshot/expected_data.json').read_text())
 shutil.copytree('learning/references',root/'references',dirs_exist_ok=True)
 manifest=json.loads(Path('learning/source_snapshot/engine_manifest.json').read_text())
 for e,v in manifest.items():
  for f,sha in v['files'].items():assert hashlib.sha256((Path('learning/engines')/e/f).read_bytes()).hexdigest()==sha
 records=[]
 for p in (root/'study').rglob('*_metrics.json'):
  r=json.loads(p.read_text());m=r['method'];folder=p.parent
  assert r['config']['TOPOLOGY']=='random_kout' and r['config']['GOSSIP_K']==3
  assert r['execution_hardware']['threads']==3 and all(t['num_threads']<=3 for t in r['execution_hardware']['threadpools'])
  env=json.loads((folder/(m+'_runtime_environment.json')).read_text());assert env['SVM_COMPILED_LOCAL']=='1' and all(env[k]=='3' for k in ['NUMBA_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'])
  prov=json.loads((folder/(m+'_learning_provenance.json')).read_text());assert prov['source_manifest']==manifest[prov['engine']]
  jit=folder/'bdsvm_timing_jit_proof.json'
  if m=='bdsvm' and r['parameters'].get('matvec_backend')=='numba':
   j=json.loads(jit.read_text());assert j['no_compilation_during_training'] and j['threads_before']==j['threads_after']==3 and j['signatures_before']==j['signatures_after']
  records.append(dict(dataset=r['dataset'],seed=r['seed'],method=m,metric_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),weights_sha256=hashlib.sha256(p.with_suffix('.npz').read_bytes()).hexdigest()))
 assert len(records)==360
 (OUT/'formal_learning_360_manifest.json').write_text(json.dumps({'records':records},indent=2))
 data=Path('C:/Users/remote/GadgetSVM-Lincoln/data/processed')
 (OUT/'input_audit.json').write_text(json.dumps(input_audit(expected,data),indent=2))
 # Recover complete valid timing snapshot only after old timing process ended.
 outer=Path('D:/LocalLLM/jobs/incoming/a7c62726-8f13-4316-a88a-cc165b01ed48.result.zip')
 with zipfile.ZipFile(outer) as z:
  assert z.testzip() is None;data2=z.read('result.zip')
 with zipfile.ZipFile(io.BytesIO(data2)) as z:
  assert z.testzip() is None
  complete=json.loads(z.read('output/COMPLETE.json'));assert complete['complete'] and complete['records']==360
  for n in z.namelist():
   if n.startswith('output/study/') and not n.endswith('/'):
    assert '_metrics.' not in n
    p=root/Path(n).relative_to('output');p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(z.read(n))
  for e,v in manifest.items():
   for f,sha in v['files'].items():assert hashlib.sha256(z.read('source_snapshot/engines/'+e+'/'+f)).hexdigest()==sha
 assert len(list((root/'study').rglob('*_timing.json')))==360
 for p in (root/'study').rglob('*_timing.json'):
  r=json.loads(p.read_text());m=r['method'];folder=p.parent
  env=json.loads((folder/(m+'_runtime_environment.json')).read_text());assert env['SVM_COMPILED_LOCAL']=='1' and all(env[k]=='3' for k in ['NUMBA_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'])
  assert not json.loads((folder/(m+'_isolation_monitor.json')).read_text())['interference']
  prov=json.loads((folder/(m+'_replay_provenance.json')).read_text());assert prov['source_manifest']==manifest[prov['engine']]
  learning=json.loads((folder/(m+'_metrics.json')).read_text());assert r['adjacency']==learning['adjacency']
  if m=='bdsvm' and r['parameters'].get('matvec_backend')=='numba':
   j=json.loads((folder/'bdsvm_timing_jit_proof.json').read_text());assert j['threads_before']==j['threads_after']==3 and j['no_compilation_during_training'] and j['signatures_before']==j['signatures_after']
 (root/'expected_data.json').write_text(json.dumps(expected))
 report=a.audit(root,data)
 assert report['actual_isolated_runs']==report['target_passed_runs']==360 and report['capped_runs']==report['certified_budget_infeasible_runs']==0
 (OUT/'FINAL_360_AUDIT_APPROVED.json').write_text(json.dumps(report,indent=2))
 import plot_unified_quality as plot
 for ds in expected:plot.render(root,ds);print('RENDERED',ds,flush=True)
 for p in (root/'figures').glob('*.png'):shutil.copy2(p,OUT/p.name)
 print('FINAL RANDOM ALL360 AUDIT AND EIGHT FIGURES PASS',flush=True)
except Exception:
 (OUT/'error.json').write_text(json.dumps({'error':traceback.format_exc()},indent=2));print(traceback.format_exc(),flush=True)
finally:
 with zipfile.ZipFile('result.zip','w',zipfile.ZIP_DEFLATED) as z:
  for p in OUT.iterdir():z.write(p,p.name)
