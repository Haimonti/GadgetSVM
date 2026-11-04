import sys,json,zipfile,io,statistics,psutil
from pathlib import Path
sys.stderr=sys.stdout
p=Path('D:/LocalLLM/jobs/incoming/c90c54dc-dc9c-40f0-aa2b-1e2f069aefd4.result.zip')
with zipfile.ZipFile(p) as z:
 assert z.testzip() is None
 raw=z.read('result.zip')
with zipfile.ZipFile(io.BytesIO(raw)) as z:
 assert z.testzip() is None
 rows=[];memory=[]
 for n in z.namelist():
  if not n.startswith('output/study/') or not n.endswith('_metrics.json'):continue
  r=json.loads(z.read(n));rows.append({k:r.get(k) for k in ['dataset','seed','method','converged','stop_reason','rounds','total_comm_bytes','average_accuracy','max_quality_gap','process_peak_RSS_bytes']})
  for k,v in r.items():
   if 'memory' in k.lower() or 'rss' in k.lower():memory.append({'dataset':r['dataset'],'seed':r['seed'],'method':r['method'],'key':k,'value':v})
 complete=json.loads(z.read('output/LEARNING_COMPLETE.json')) if 'output/LEARNING_COMPLETE.json' in z.namelist() else None
 protocol=json.loads(z.read('output/protocol.json'))
 report={'complete':complete,'records':rows,'groups':[],'memory_fields':memory,'protocol':protocol,'current_memory':dict(psutil.virtual_memory()._asdict())}
 for ds in protocol['datasets']:
  for m in protocol['methods']:
   rs=sorted([r for r in rows if r['dataset']==ds and r['method']==m],key=lambda r:r['seed'])
   assert len(rs)==5 and [r['seed'] for r in rs]==list(range(42,47))
   g={'dataset':ds,'method':m,'n':5,'passed':sum(r['converged'] for r in rs)}
   for k in ['rounds','total_comm_bytes','average_accuracy','max_quality_gap']:
    vs=[r[k] for r in rs]
    if all(v is not None for v in vs):g[k+'_mean']=statistics.mean(vs);g[k+'_sd']=statistics.stdev(vs)
   report['groups'].append(g)
Path('preliminary.json').write_text(json.dumps(report,indent=2))
with zipfile.ZipFile('result.zip','w',zipfile.ZIP_DEFLATED) as z:z.write('preliminary.json')
print(json.dumps({'records':len(rows),'passed':sum(r['converged'] for r in rows),'memory_fields':len(memory)}))
