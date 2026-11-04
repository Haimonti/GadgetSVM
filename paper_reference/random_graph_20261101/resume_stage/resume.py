"""Restore exact verified300 timing cache + exact360learning; original controller runs remaining60 only."""
import sys,os,json,hashlib,zipfile,io,runpy,traceback,shutil
from pathlib import Path
sys.stderr=sys.stdout
ROOT=Path.cwd().resolve();OUT=ROOT/'output'
try:
 import psutil
 assert not any(any(Path(str(a)).name=='replay_case.py' for a in p.info['cmdline'] or []) for p in psutil.process_iter(['cmdline']))
 proof=json.loads((ROOT/'checkpoint_audit.json').read_text());assert proof['passed'] and proof['records']==300
 def read_inner(id,sha):
  raw=(Path('D:/LocalLLM/jobs/incoming')/(id+'.result.zip')).read_bytes();assert hashlib.sha256(raw).hexdigest()==sha
  with zipfile.ZipFile(io.BytesIO(raw)) as z:assert z.testzip() is None;data=z.read('result.zip')
  z=zipfile.ZipFile(io.BytesIO(data));assert z.testzip() is None;return z
 old=read_inner('08d2573d-fcb0-426a-8c35-c51f07771891',proof['timing_outer_sha256'])
 for n in old.namelist():
  if n.endswith('/'):continue
  if n.startswith('output/'):target=ROOT/n
  elif n.startswith('source_snapshot/'):target=ROOT/Path(n).relative_to('source_snapshot')
  else:continue
  target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(old.read(n))
 for n,sha in proof['source_sha'].items():assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==sha
 assert len(list((OUT/'study').rglob('*_timing.json')))==300
 for r in proof['cached']:
  p=ROOT/r['path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==r['metric_sha256']
  assert hashlib.sha256(p.with_suffix('.npz').read_bytes()).hexdigest()==r['weights_sha256']
 learn=read_inner('c90c54dc-dc9c-40f0-aa2b-1e2f069aefd4',proof['learning_outer_sha256'])
 for n in learn.namelist():
  if n.endswith('/'):continue
  if n.startswith('references/'):target=ROOT/'learning'/n
  elif n.startswith('output/study/'):target=ROOT/'learning'/Path(n).relative_to('output')
  elif n=='source_snapshot/expected_data.json':target=ROOT/'learning/expected_data.json'
  elif n=='output/protocol.json':target=ROOT/'learning/protocol.json'
  else:continue
  target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(learn.read(n))
 (ROOT/'learning/FORMAL_360_AUDIT_APPROVED.json').write_bytes((ROOT/'FORMAL_360_AUDIT_APPROVED.json').read_bytes())
 journal=OUT/'continuation_resume';journal.mkdir(exist_ok=True)
 for n in ['resume.py','checkpoint_audit.json','FORMAL_360_AUDIT_APPROVED.json']:shutil.copy2(ROOT/n,journal/n)
 (journal/'restored.json').write_text(json.dumps(dict(records=300,learning_records=360,controller_and_engines_unchanged=True,only_remaining_tasks=60),indent=2))
 print('EXACT300CACHES RESTORED; original frozen controller resumesremaining60',flush=True)
 runpy.run_path(str(ROOT/'job.py'),run_name='__main__')
except Exception:
 OUT.mkdir(exist_ok=True);(OUT/'resume_error.json').write_text(json.dumps({'error':traceback.format_exc()},indent=2));print(traceback.format_exc(),flush=True)
 with zipfile.ZipFile(ROOT/'result.zip','w',zipfile.ZIP_DEFLATED) as z:
  for p in OUT.rglob('*'):
   if p.is_file():z.write(p,p.relative_to(ROOT).as_posix())
