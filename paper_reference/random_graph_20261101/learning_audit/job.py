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
 tasks=[(ds,s) for ds in expected for s in range(42,47)]
 with ThreadPoolExecutor(max_workers=4) as ex:reports=list(ex.map(lambda t:a.audit_case(root,t[0],t[1],data,learning_only=True),tasks))
 assert sum(r['passed'] for r in reports)==360 and sum(r['capped']+r['budget_infeasible'] for r in reports)==0
 report=dict(passed=True,records=360,phase='random_graph_learning',checks=['independent train/ref/dual/model/test arithmetic','independent graph and selected3/versioned/full-dense wire arithmetic','frozen engine SHA','compiled1 and all reported threadpools3','Numba mask3 and no new JIT signatures','all360 full five-seed matrix, no capped/infeasible subset'],cases=reports)
 (OUT/'FORMAL_360_AUDIT_APPROVED.json').write_text(json.dumps(report,indent=2));print('RANDOM ALL360 LEARNING INDEPENDENT AUDIT PASS',flush=True)
except Exception:
 (OUT/'error.json').write_text(json.dumps({'error':traceback.format_exc()},indent=2));print(traceback.format_exc(),flush=True)
finally:
 with zipfile.ZipFile('result.zip','w',zipfile.ZIP_DEFLATED) as z:
  for p in OUT.iterdir():z.write(p,p.name)
