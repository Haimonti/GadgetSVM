"""Run the pinned author CoLA loop using portable coordinate/transport adapters.

The original Cython extension cannot be built without MSVC on the Windows
host. Its coordinate arithmetic is checked against independent box QP tests;
this check separately verifies the author's full loop, subproblem conversion,
sampling and K-scaling against the PeerSim port. Not a native MPI benchmark.
"""
import importlib.util
import sys
import threading
import types
from pathlib import Path
import numpy as np
import scipy.sparse as sp


def verify():
    root=Path(__file__).parent/'author_cola'
    sys.path.insert(0,str(root))
    # Portable dense implementation of the author's signed box-QP backend.
    def dense_cd(alpha,y,b,A,max_iter,C,tol,rng):
        a=alpha.copy();v=A@a;norm=(A*A).sum(axis=0)
        state=int(rng.randint(0,0x7fffffff))
        for _ in range(int(max_iter*len(a))):
            state^=(state<<13)&0xffffffff;state^=state>>17
            state^=(state<<5)&0xffffffff;state&=0xffffffff
            i=(state%0x80000000)%len(a)
            if norm[i]==0:continue
            proposal=a[i]+(y[i]+b[i]-C*(A[:,i]@v))/(C*norm[i])
            new=y[i]*min(1.,max(0.,y[i]*proposal))
            v+=(new-a[i])*A[:,i];a[i]=new
        return a,0.,tol,int(max_iter*len(a))
    svm=types.ModuleType('fast_cd.svm');svm.dense_cd=dense_cd
    sys.modules['fast_cd.svm']=svm
    sys.modules['fast_cd.elasticnet']=types.ModuleType('fast_cd.elasticnet')
    K=3;rounds=4;H=60;thread_state=threading.local()
    barrier=threading.Barrier(K,timeout=30);payloads={};lock=threading.Lock()
    comm=types.ModuleType('cola.communication')
    comm.get_rank=lambda:thread_state.rank
    comm.local_average=lambda n,lookup,vs:sum(w*vs[i] for i,w in lookup.items())
    def exchange(rank,lookup,vs):
        with lock:payloads[rank]=vs[rank].copy()
        barrier.wait()
        for j in lookup:vs[j]=payloads[j].copy()
        barrier.wait()
    comm.p2p_communicate_neighborhood_tensors=exchange
    sys.modules['cola.communication']=comm
    from cola.cocoasolvers import LinearSVM
    from cola.algo import cola
    from src.linear_baseline_simulation import LinearBaselineSimulation
    from src.peersim_python.core import Network
    rng=np.random.default_rng(11);X=rng.normal(size=(24,4));y=np.tile([-1.,1.],12)
    indices=np.array_split(np.arange(24),K);reference={};errors=[]
    class Graph:
        def get_neighborhood(self,rank):return {i:1./K for i in range(K)}
    class Monitor:
        ckpt_freq=1000
        def log(self,*args):return False
    def run(i):
        try:
            thread_state.rank=i;ix=indices[i]
            solver=LinearSVM(C=1./(.2*24),random_state=42+1009*i)
            reference[i]=cola(X[ix].T,y[ix],solver,1.,0.,rounds,H/len(ix),K,Graph(),Monitor())
        except Exception as exc:errors.append(repr(exc));barrier.abort()
    threads=[threading.Thread(target=run,args=(i,)) for i in range(K)]
    for t in threads:t.start()
    for t in threads:t.join(40)
    if errors or any(t.is_alive() for t in threads):raise AssertionError(errors or 'reference stalled')
    data=[dict(X_csr=sp.csr_matrix(X[ix]),y=y[ix],X_test=object(),y_test=object(),n_local=len(ix)) for ix in indices]
    cfg=dict(NUM_WORKERS=K,SEED=42,LAMBDA=.2,GOSSIP_K=2,TOPOLOGY='full',SDCA_LOCAL_STEPS=H)
    sim=LinearBaselineSimulation(cfg,data,'cola').build().sim
    sim.cycles=rounds;sim.nextExperiment()
    for i in range(K):
        p=Network.get(i).getProtocol(1)
        np.testing.assert_allclose(p.alpha,reference[i][1],atol=1e-11,rtol=1e-11)
        np.testing.assert_allclose(p.local_product,reference[i][0],atol=1e-11,rtol=1e-11)
    return {'author_commit':'c87ba35b3fa54166744585549cd0c85ce279c2ed',
            'reference':'unmodified author CoLA loop and LinearSVM class; portable coordinate and barrier adapters',
            'nodes':K,'rounds':rounds,'tolerance':1e-11,'passed':True,
            'native_cython_compiled':False,'native_failure':'Missing MSVC C++ build tools on Windows'}


if __name__=='__main__':
    import json
    print(json.dumps(verify(),indent=2))
