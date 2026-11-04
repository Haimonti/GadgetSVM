import sys,json,time,zipfile
from pathlib import Path
sys.stderr=sys.stdout
w=Path('D:/LocalLLM/jobs/runs/a7c62726-8f13-4316-a88a-cc165b01ed48/work')
r={'captured_utc':time.time(),'remote_work_exists':w.exists(),'job_id':'a7c62726-8f13-4316-a88a-cc165b01ed48'}
for name in ['progress.json','error.json','LEARNING_COMPLETE.json','LEARNING_CHECKPOINT.json','PREFLIGHT_PASS.json','COMPLETE.json','CHECKPOINT.json','bootstrap_error.json','resume_error.json']:
 p=w/'output'/name
 if p.exists():r[name]={'mtime':p.stat().st_mtime,'value':json.loads(p.read_text())}
import psutil
r['python_processes']=[]
for p in psutil.process_iter(['pid','cmdline','cpu_times','memory_info']):
 try:
  args=p.info['cmdline'] or []
  if any(str(a).endswith('replay_case.py') for a in args):r['python_processes'].append({'pid':p.pid,'args':args,'cpu_seconds':sum(p.cpu_times()[:2]),'rss':p.memory_info().rss,'affinity':p.cpu_affinity()})
 except (psutil.NoSuchProcess,psutil.AccessDenied):pass
r['available_memory']=psutil.virtual_memory().available
from independent_wire import verify
if (w/'output/smoke').exists():
 proofs=[]
 for p in (w/'output/smoke').rglob('*_metrics.json'):
  v=json.loads(p.read_text());verify(v);proofs.append(str(p.relative_to(w)))
 r['independent_smoke_wire']={'passed':True,'records':len(proofs),'cases':proofs}
Path('status.json').write_text(json.dumps(r,indent=2))
with zipfile.ZipFile('result.zip','w',zipfile.ZIP_DEFLATED) as z:z.write('status.json')
print(json.dumps({'captured_utc':r['captured_utc'],'work_exists':r['remote_work_exists'],'active_children':len(r['python_processes']),'completed':r.get('progress.json',{}).get('value',{}).get('complete')}),flush=True)
