"""Fresh random-graph learning; frozen ring mathematics with explicit topology-only changes."""
import os,sys,time,json,hashlib,random,zipfile,subprocess,traceback,threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
sys.stderr=sys.stdout
os.environ.update(PYTHONUTF8='1',SVM_COMPILED_LOCAL='1',NUMBA_NUM_THREADS='3')
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='3'
ROOT=Path.cwd().resolve();OUT=ROOT/'output';OUT.mkdir(exist_ok=True)
DATA=Path('C:/Users/remote/GadgetSVM-Lincoln/data/processed')
sys.path.insert(0,str(ROOT/'engines/base'))
from run_unified_quality import METHODS
from run_unified_matrix import hardware_topology,input_audit
import psutil

def write(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2));tmp.replace(p)
def graph(seed):
 # Independent port of the declared partial Fisher-Yates graph construction.
 rng=random.Random(seed);nodes=list(range(10));adj=[set() for _ in nodes]
 for i in range(10):
  j=0
  while j<3:
   q=j+rng.randint(0,9-j);nodes[j],nodes[q]=nodes[q],nodes[j]
   if nodes[j]!=i:adj[i].add(nodes[j]);adj[nodes[j]].add(i);j+=1
 seen={0};todo=[0]
 while todo:
  for j in adj[todo.pop()]-seen:seen.add(j);todo.append(j)
 assert len(seen)==10 and all(len(a)>=3 for a in adj)
 return [sorted(a) for a in adj]
GRAPHS={str(s):graph(s) for s in range(42,47)}
def engine_for(ds,m):
 if m!='bdsvm':return 'choco_w8a' if (ds,m)==('w8a','choco') else 'base'
 return 'bd_big_v4' if ds in ['real-sim','rcv1'] else 'bd_gisette_v3' if ds=='gisette' else 'bd_v1'
def args_for(ds,seed,m,folder,cycles=50000,engine=None):
 e=engine or engine_for(ds,m)
 a=[ds,'--seed',str(seed),'--out',str(folder),'--reference',str(ROOT/'references'/ds/f'seed_{seed}'),'--data-dir',str(DATA),'--cycles',str(cycles),'--threads','3']
 if m=='bdsvm':
  a+=['--rho','.5','--cg-rtol','1e-8','--weight-floor','1e-4']
  if e=='bd_gisette_v3':a+=['--linear-solver','cholesky','--preconditioner','none','--matrix-storage','dense']
  if e=='bd_big_v4':a+=['--linear-solver','cg','--preconditioner','none','--matrix-storage','csr','--matvec-backend','numba']
 else:a+=['--method',m,'--target','.001']
 return [sys.executable,'-u',str(ROOT/'replay_case.py'),str(ROOT/'engines'/e),e]+a

def check_result(p,formal=True):
 r=json.loads(p.read_text());assert r['config']['TOPOLOGY']=='random_kout' and r['config']['GOSSIP_K']==3
 assert r['lambda']==1e-4 and r['target']==.001 and not r['fit_intercept']
 assert p.with_suffix('.npz').exists() and r['phase']=='parallel_learning'
 assert r['execution_hardware']['threads']==3 and all(t['num_threads']<=3 for t in r['execution_hardware']['threadpools'])
 if r['method']!='centralized':assert [sorted(a) for a in r['adjacency']]==GRAPHS[str(r['seed'])],(p,r['adjacency'])
 env=json.loads((p.parent/(r['method']+'_runtime_environment.json')).read_text());assert env['SVM_COMPILED_LOCAL']=='1' and env['NUMBA_NUM_THREADS']=='3'
 proof=p.parent/'bdsvm_timing_jit_proof.json'
 if r['method']=='bdsvm' and r['parameters'].get('matvec_backend')=='numba':
  q=json.loads(proof.read_text());assert q['no_compilation_during_training'] and q['threads_before']==q['threads_after']==3
 if formal:assert r['round_cap']==50000 and r['converged'] and r['stop_reason']=='quality_target' and r['max_quality_gap']<=.001
 return r

def checkpoint():
 with zipfile.ZipFile(ROOT/'result.partial.zip','w',zipfile.ZIP_DEFLATED) as z:
  for d in ['output','references','engines']:
   for p in (ROOT/d).rglob('*'):
    if p.is_file() and '__pycache__' not in p.parts:z.write(p,p.relative_to(ROOT).as_posix())
  for n in ['job.py','replay_case.py','engine_manifest.json','topology_only_diff.json','expected_data.json']:z.write(ROOT/n,'source_snapshot/'+n)
 (ROOT/'result.partial.zip').replace(ROOT/'result.zip')

lock=threading.RLock();cv=threading.Condition(lock);active={};rows=[];failed=threading.Event();reserved=0;began=time.monotonic()

def progress():write(OUT/'progress.json',dict(phase='random_graph_parallel_learning',complete=len([r for r in rows if r['status'] in ['complete','cached']]),total=360,active=active,results=rows,updated_utc=time.time()))
def invoke(t):
 global reserved
 ds,seed,m=t;key=f'{ds}/{seed}/{m}';folder=OUT/'study'/ds/f'seed_{seed}';folder.mkdir(parents=True,exist_ok=True);p=folder/(m+'_metrics.json')
 if p.exists():check_result(p);return dict(task=t,status='cached')
 weight=10 if m=='bdsvm' and ds in ['real-sim','rcv1'] else 4 if ds in ['gisette','webspam','covtype'] else 2
 with cv:
  while reserved+weight>memory_budget or psutil.virtual_memory().available<(16+weight)*1024**3:
   if failed.is_set() or time.monotonic()-began>12*3600:return dict(task=t,status='deferred')
   cv.wait(10)
  if failed.is_set() or time.monotonic()-began>12*3600:return dict(task=t,status='deferred')
  reserved+=weight;active[key]={'started_utc':time.time(),'reserved_GiB':weight};progress()
 try:
  cmd=args_for(ds,seed,m,folder)
  write(folder/(m+'_learning_provenance.json'),dict(engine=engine_for(ds,m),argv=cmd,seed_graph=GRAPHS[str(seed)],source_manifest=manifest[engine_for(ds,m)]))
  with (folder/(m+'_metrics.log')).open('w',encoding='utf8') as log:code=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT).returncode
  assert code==0,(key,code)
  v=check_result(p);return dict(task=t,status='complete',rounds=v['rounds'],gap=v['max_quality_gap'])
 except Exception:
  failed.set();return dict(task=t,status='failed',error=traceback.format_exc())
 finally:
  with cv:reserved-=weight;active.pop(key,None);cv.notify_all()
try:
 # No old experiment may overlap the newly authorized learning batch.
 for proc in psutil.process_iter(['pid','cmdline']):
  if proc.pid==os.getpid():continue
  names=[Path(str(x)).name for x in proc.info['cmdline'] or []]
  assert not any(n in names for n in ['run_unified_quality.py','run_native_sparse_bdsvm.py','replay_case.py']),proc.pid
 expected=json.loads((ROOT/'expected_data.json').read_text());manifest=json.loads((ROOT/'engine_manifest.json').read_text())
 for e,v in manifest.items():
  for n,sha in v['files'].items():assert hashlib.sha256((ROOT/'engines'/e/n).read_bytes()).hexdigest()==sha
 write(OUT/'hardware_topology.json',hardware_topology());write(OUT/'input_audit.json',input_audit(expected,DATA))
 memory_budget=max(10,int(psutil.virtual_memory().available/1024**3)-16)
 write(OUT/'protocol.json',dict(topology='random_kout',gossip_k=3,undirected=True,workers=10,graphs=GRAPHS,seeds=list(range(42,47)),datasets=list(expected),methods=list(METHODS),target=.001,lambda_reg=1e-4,bias=0,cap=50000,evaluation_every=10,learning_parallelism=24,reserved_memory_budget_GiB=memory_budget,free_memory_floor_GiB=16,ring_results_reused=False,reference_reuse='topology-independent feasible dual certificates; no model warm start',isolated_timing='Only after all360 learning and independent audit; same Pcores[10,11,12]/3threads'))
 # Tiny new-graph verification before formal learning; never used as formal results.
 smoke=[('base',m,42) for m in METHODS if m not in ['centralized','bdsvm']]+[(e,'bdsvm',42) for e in ['bd_v1','bd_gisette_v3','bd_big_v4']]+[('base','sdca',s) for s in range(43,47)]
 for e,m,seed in smoke:
  f=OUT/'smoke'/e/f'seed_{seed}'/m;f.mkdir(parents=True,exist_ok=True)
  with (f/'run.log').open('w',encoding='utf8') as log:code=subprocess.run(args_for('a9a',seed,m,f,cycles=2,engine=e),stdout=log,stderr=subprocess.STDOUT).returncode
  assert code==0,(e,m,seed)
  check_result(f/(m+'_metrics.json'),formal=False)
 write(OUT/'PREFLIGHT_PASS.json',dict(cases=len(smoke),graphs=GRAPHS,all_engine_sha_verified=True,all_source_data_hash_verified=True,actual_graphs_equal_independent_graph=True))
 checkpoint();print('RANDOM_GRAPH_PREFLIGHT_PASS; starting fresh360 learning',flush=True)
 tasks=[(ds,s,m) for ds in expected for s in range(42,47) for m in METHODS]
 tasks.sort(key=lambda t:0 if t[0] in ['real-sim','rcv1'] and t[2]=='bdsvm' else 1)
 with ThreadPoolExecutor(max_workers=24) as ex:
  for f in as_completed([ex.submit(invoke,t) for t in tasks]):
   row=f.result()
   with lock:rows.append(row);progress();checkpoint()
   print(row,flush=True)
 complete=len(rows)==360 and all(r['status'] in ['complete','cached'] for r in rows)
 write(OUT/('LEARNING_COMPLETE.json' if complete else 'LEARNING_CHECKPOINT.json'),dict(complete=complete,records=len(rows),results=rows))
 if failed.is_set():raise RuntimeError('Learner failed; frozen logs retained, new launches stopped; inspect before recovery')
except Exception:write(OUT/'error.json',dict(error=traceback.format_exc()));print(traceback.format_exc(),flush=True)
finally:checkpoint();print('Random learning output checkpoint packaged',flush=True)
