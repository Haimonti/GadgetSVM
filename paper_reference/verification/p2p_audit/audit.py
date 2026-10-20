import sys,json,warnings
sys.stderr=sys.stdout
import numpy as np
import scipy.sparse as sp
from p2p.cocoa_gossip import CoCoAGossipProtocol,CoCoAPlusGossipProtocol
from src.peersim_python.core import CommonState
from cocoa.cocoa import _local_sdca
from p2p.bdsvm_gossip import BDSVMGossipProtocol
from p2p.fedavg_gossip import SizeWeightedAverageAggregator
import torch
from paper_reference.ssl import SSLConfig,encoder
warnings.filterwarnings('ignore',message='.*weight_norm.*deprecated.*')
r={};rng=np.random.default_rng(8)
X=rng.normal(size=(8,3));y=np.array([1.,-1.]*4);a=rng.uniform(0,.2,8);w=rng.normal(size=3)*.01
for cls in [CoCoAGossipProtocol,CoCoAPlusGossipProtocol]:
 p=cls();p.set_data(sp.csr_matrix(X),y,sp.csr_matrix(X),y,.3);p.configure_network(0,16,2,local_steps=20)
 p.alpha=a*y;p.w=w.copy()
 indices=np.random.RandomState(19).randint(8,size=20).tolist()
 class Sequence:
  def __init__(self):self.i=0
  def randrange(self,n):v=indices[self.i];self.i+=1;return v
 old=CommonState.r;CommonState.r=Sequence()
 try:actual=p._local_sdca()
 finally:CommonState.r=old
 expected,_=_local_sdca(X,y,w,a.copy(),20,.3,16,19,p.PLUS,p.sigma)
 err=float(np.max(np.abs(actual-y*expected)));assert err<1e-12
 r[cls.__name__]={'local_dual_max_error':err,'commit_scale':p.scaling,'sigma':p.sigma}
# Fixed geometry across clients is a necessary invariant for summed kernel statistics.
models=[]
for i,scale in enumerate([.1,10.]):
 p=BDSVMGossipProtocol(P=4,C=1.,arch_seed=0)
 xx=sp.csr_matrix(scale*np.array([[1.,2.],[2.,1.],[1.,1.]]));yy=np.array([1.,-1.,1.])
 p.set_data(xx,yy,xx,yy,.01);p.configure_network(i,6,2);models.append(p)
r['bdsvm']={'default_local_gammas':[float(p.gamma) for p in models],
 'kernel_matrix_max_difference':float(np.max(np.abs(models[0].Kpp-models[1].Kpp)))}
assert not np.isclose(models[0].gamma,models[1].gamma)
# Synchronous star neighbor mixing: row-stochastic but not average preserving.
agg=SizeWeightedAverageAggregator();states=[{'w':np.array([v]),'n':1} for v in [0.,1.,0.]]
graph=[[1,2],[0],[0]]
merged=[agg.aggregate(states[i],[states[j] for j in graph[i]])['w'][0] for i in range(3)]
r['fedavg']={'equal_shard_star_initial_mean':1/3,'after_neighbor_mixing_mean':float(np.mean(merged)),
 'interpretation':'generic neighbor averaging need not preserve global data-weighted mean; not a proof of divergence'}
m=encoder(SSLConfig())
r['ssl_architecture']={'upstream_depth':10,'conv1d_count':sum(isinstance(x,torch.nn.Conv1d) for x in m.modules()),
 'main_kernel3_conv_count':sum(isinstance(x,torch.nn.Conv1d) and x.kernel_size==(3,) for x in m.modules()),
 'parameters':sum(p.numel() for p in m.parameters())}
open('audit.json','w').write(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
