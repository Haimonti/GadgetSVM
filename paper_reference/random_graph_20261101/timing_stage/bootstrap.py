import sys,os,zipfile,io,json,runpy,traceback
from pathlib import Path
sys.stderr=sys.stdout
ROOT=Path.cwd().resolve()
try:
 approval=json.loads((ROOT/'FORMAL_360_AUDIT_APPROVED.json').read_text());assert approval['passed'] and approval['records']==360
 outer=Path('D:/LocalLLM/jobs/incoming/c90c54dc-dc9c-40f0-aa2b-1e2f069aefd4.result.zip')
 with zipfile.ZipFile(outer) as z:
  assert z.testzip() is None;data=z.read('result.zip')
 with zipfile.ZipFile(io.BytesIO(data)) as z:
  assert z.testzip() is None
  for n in z.namelist():
   if n.endswith('/'):continue
   if n.startswith('engines/'):target=ROOT/n
   elif n.startswith('references/'):target=ROOT/'learning'/n
   elif n.startswith('output/study/'):target=ROOT/'learning'/Path(n).relative_to('output')
   elif n=='source_snapshot/expected_data.json':target=ROOT/'learning/expected_data.json'
   elif n=='output/protocol.json':target=ROOT/'learning/protocol.json'
   else:continue
   target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(n))
 (ROOT/'learning/FORMAL_360_AUDIT_APPROVED.json').write_text(json.dumps(approval,indent=2))
 runpy.run_path(str(ROOT/'job.py'),run_name='__main__')
except Exception:
 out=ROOT/'output';out.mkdir(exist_ok=True);(out/'bootstrap_error.json').write_text(json.dumps({'error':traceback.format_exc()},indent=2));print(traceback.format_exc(),flush=True)
 with zipfile.ZipFile(ROOT/'result.zip','w',zipfile.ZIP_DEFLATED) as z:
  for p in out.rglob('*'):
   if p.is_file():z.write(p,p.relative_to(ROOT).as_posix())
