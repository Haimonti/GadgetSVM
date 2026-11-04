"""Independent topology/schedule arithmetic, imports no training implementation."""
import random,hashlib

def graph(seed):
 rng=random.Random(seed);pool=list(range(10));a=[set() for _ in pool]
 for i in range(10):
  j=0
  while j<3:
   q=j+rng.randint(0,9-j);pool[j],pool[q]=pool[q],pool[j]
   if pool[j]!=i:a[i].add(pool[j]);a[pool[j]].add(i);j+=1
 return [sorted(v) for v in a]
def rng(seed,cycle,node,stream):
 return random.Random(int.from_bytes(hashlib.sha256(f'{seed}:{cycle}:{node}:{stream}'.encode()).digest()[:16],'big'))
def entries(rounds,seed,a):
 known=[{i} for i in range(10)];inbox=[[] for _ in range(10)];total=0
 for cycle in range(min(rounds,100)):
  order=list(range(10));rng(seed,cycle,-1,'activation').shuffle(order)
  for i in order:
   for msg in inbox[i]:known[i]|=msg
   inbox[i]=[];dest=rng(seed,cycle,i,'neighbors').sample(a[i],min(3,len(a[i])))
   total+=len(dest)*len(known[i])
   for j in dest:inbox[j].append(set(known[i]))
  if all(len(s)==10 for s in known):return total+(rounds-cycle-1)*sum(min(3,len(v)) for v in a)*10
 assert rounds<=100,'Origin propagation did not saturate as expected'
 return total

def verify(r):
 m=r['method'];seed=r['seed'];d=r['features'];cycles=r['rounds']
 assert r['config']['TOPOLOGY']=='random_kout' and r['config']['GOSSIP_K']==3
 if m=='centralized':assert r['total_comm_bytes']==0;return
 a=graph(seed);assert [sorted(v) for v in r['adjacency']]==a
 selected=sum(min(3,len(v)) for v in a);all_edges=sum(map(len,a))
 if m in ['sdca','cocoa','cocoa_plus','bdsvm']:
  size=8*(d*d+d)+24 if m=='bdsvm' else 8*d+24
  expected=entries(cycles,seed,a)*size
 elif m=='fedavg':expected=selected*cycles*(8*d+8)
 elif m=='gadget':expected=selected*cycles*8*d
 elif m=='cola':expected=all_edges*cycles*8*d
 elif m=='choco':expected=all_edges*cycles*((d+4)//5)*12
 else:raise ValueError(m)
 assert expected==r['total_comm_bytes'],(m,seed,expected,r['total_comm_bytes'])
 return expected
