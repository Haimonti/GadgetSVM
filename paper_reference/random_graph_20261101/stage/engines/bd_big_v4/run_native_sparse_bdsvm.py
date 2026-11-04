"""Native full-linear IRWLS: lossless sparse storage, unchanged dense-statistic gossip."""
import argparse,os,json,time,hashlib
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument('dataset');p.add_argument('--seed',type=int,default=42)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--cycles',type=int,default=50000)
    p.add_argument('--threads',type=int,default=3);p.add_argument('--timing',action='store_true')
    p.add_argument('--training-only',action='store_true');p.add_argument('--rho',type=float,default=.5)
    p.add_argument('--cg-rtol',type=float,default=1e-8);p.add_argument('--weight-floor',type=float,default=1e-4)
    p.add_argument('--linear-solver',choices=['cg','cholesky'],default='cg')
    p.add_argument('--preconditioner',choices=['jacobi','none'],default='jacobi')
    p.add_argument('--matrix-storage',choices=['csr','dense'],default='csr')
    p.add_argument('--matvec-backend',choices=['scipy','numba'],default='scipy')
    a=p.parse_args()
    for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:
        os.environ[k]=str(a.threads)
    import psutil,numpy as np,scipy.sparse as sp
    if a.timing:psutil.Process().cpu_affinity([10,11,12])
    from src.config import CONFIG
    from run_unified_quality import load_training,test_data,write
    from src.peersim_python.core import Network
    from p2p.native_sparse_bdsvm import NativeSparseLinearBDSVM
    from src.bdsvm_simulation import BDSVMSimulation
    from src.unified_quality import CommonQualityEvaluator
    c=dict(CONFIG);c.update(DATASET=a.dataset,SEED=a.seed,LAMBDA=1e-4,NUM_WORKERS=10,
        TOPOLOGY='random_kout',GOSSIP_K=3,ROUNDS=a.cycles,WARM_START=False,
        STOP_ON_PLATEAU=False,STOP_ON_THRESHOLD=False)
    for k in c:
        if k.endswith('_PATH'):c[k]=a.data_dir/Path(str(c[k]).replace('\\','/')).name
    X,y,heldout,digest=load_training(c);ref=json.loads((a.reference/'reference.json').read_text())
    assert ref['train_digest']==digest and ref['lambda']==c['LAMBDA'] and ref['target']==.001
    assert ref['dual_feasible'] and ref['reference_gap']<=1e-4
    empty=sp.csr_matrix((0,X.shape[1]));shards=[dict(X_csr=X[ix].tocsr(),y=y[ix],X_test=empty,
        y_test=np.empty(0),n_local=len(ix)) for ix in np.array_split(np.arange(len(y)),10)]
    if a.matvec_backend=='numba':
        from p2p.native_csr_matvec import warmup
        warmup(a.threads)  # Same JIT-exclusion rule as the other compiled local solvers.
    a.out.mkdir(parents=True,exist_ok=True)
    class ReportingQuality(CommonQualityEvaluator):
        def execute(self):
            previous=len(self.history)
            reached=super().execute()
            if not a.timing and len(self.history)>previous:
                write(a.out/'training_progress.json',self.history[-1])
            return reached
    start=time.perf_counter()
    params=dict(P=X.shape[1],C=1/(2e-4*len(y)),rho=a.rho,kernel='linear',preimage='unit',arch_seed=0,ey_floor=a.weight_floor,cg_rtol=a.cg_rtol,
        linear_solver=a.linear_solver,preconditioner=a.preconditioner,matrix_storage=a.matrix_storage,
        matvec_backend=a.matvec_backend)
    assembly=BDSVMSimulation(c,shards,params=params,protocol_cls=NativeSparseLinearBDSVM).build()
    gate=ReportingQuality(X,y,c['LAMBDA'],ref['dual_lower'],.001,a.cycles,start)
    assembly.sim.controls=[gate];assembly.sim.cycles=a.cycles;rounds=assembly.sim.nextExperiment()
    seconds=time.perf_counter()-start;ps=[Network.get(i).getProtocol(1) for i in range(10)];W=np.stack([gate.weights(q) for q in ps])
    accuracy=[];ntest=None
    if gate.reached and not a.training_only:
        Xt,yt=test_data(c,X.shape[1],heldout);ntest=len(yt)
        accuracy=np.mean(np.where(Xt@W.T>0,1.,-1.)==yt[:,None],axis=0).tolist()
    d=X.shape[1];comm=sum(q.comm_bytes for q in ps);messages=sum(q.messages_sent for q in ps)
    assert messages==sum(min(3,Network.get(i).getProtocol(0).degree()) for i in range(10))*rounds
    entry_size=8*(d*d+d)+24
    ledger=[[(int(cycle),len(dest),int(size//entry_size)) for cycle,dest,size in p.communication_trace] for p in ps]
    assert all(size%entry_size==0 for p in ps for _,_,size in p.communication_trace)
    assert comm==sum(k*origins*entry_size for rows in ledger for _,k,origins in rows)
    from threadpoolctl import threadpool_info
    result=dict(dataset=a.dataset,seed=a.seed,method='bdsvm',converged=gate.reached,
        stop_reason='quality_target' if gate.reached else 'round_limit',rounds=rounds,round_cap=a.cycles,
        target=.001,objective_definition='mean_hinge + lambda/2 * squared_weight_norm; bias=0',
        reference_dual_lower=ref['dual_lower'],reference_primal_upper=ref['primal_upper'],reference_gap=ref['reference_gap'],
        final_train_objectives=gate.final_objectives.tolist(),final_train_hinge=float(gate.final_hinges.mean()),
        max_quality_gap=float(max(gate.final_objectives-ref['dual_lower'])),train_samples=len(y),features=d,
        train_digest=digest,fit_intercept=False,zero_score_label=-1,test_samples=ntest,
        average_accuracy=float(np.mean(accuracy)) if accuracy else None,per_node_accuracy=accuracy,
        test_evaluations=len(accuracy),total_comm_bytes=comm,messages_sent=messages,training_seconds=seconds,
        phase='training_only_pilot' if a.training_only else 'isolated_timing' if a.timing else 'parallel_learning',
        config={k:str(v) if isinstance(v,Path) else v for k,v in c.items()},
        execution_hardware=dict(cpu_affinity=psutil.Process().cpu_affinity(),threads=a.threads,threadpools=threadpool_info()),
        parameters=dict(P=d,coordinate_basis='identity; equivalent full orthonormal linear space',
            linear_solver=a.linear_solver,preconditioner=a.preconditioner,cg_maxiter=ps[0].cg_maxiter,
            matrix_storage=a.matrix_storage,
            matvec_backend=a.matvec_backend,
            full_linear_span=True,rho=a.rho,C=params['C'],weight_floor=a.weight_floor,cg_rtol=a.cg_rtol,
            wire_implementation='unchanged native versioned dense C_k,d_k gossip; zeros charged',
            storage_implementation='lossless '+a.matrix_storage+' for local and received matrices',
            cg_iterations_per_worker=[p.cg_iterations_total for p in ps],
            linear_solves_per_worker=[p.linear_solves for p in ps],
            statistic_formations_per_worker=[p.statistic_formations for p in ps],
            final_linear_residual_per_worker=[p.last_linear_residual for p in ps],
            max_local_statistic_storage_per_worker=[p.max_local_statistic_storage for p in ps]),history=gate.history,
        wire_ledger=ledger,process_peak_RSS_bytes=getattr(psutil.Process().memory_info(),'peak_wset',psutil.Process().memory_info().rss),
        adjacency=[[Network.get(i).getProtocol(0).getNeighbor(j).getIndex()
            for j in range(Network.get(i).getProtocol(0).degree())] for i in range(10)],
        local_compute_budget=dict(full_shard_IRWLS=True,basis_budget=d,implicit_full_basis=True,
            inner_CG_iterations_per_worker=[p.cg_iterations_total for p in ps],CG_uses_only_received_statistics=True,no_extra_inner_communication=True))
    result['lambda']=c['LAMBDA'];a.out.mkdir(parents=True,exist_ok=True)
    file=a.out/('bdsvm_timing.json' if a.timing else 'bdsvm_metrics.json');write(file,result)
    np.savez_compressed(file.with_suffix('.npz'),weights=W)
    print('FULL LINEAR BDSVM',a.dataset,result['stop_reason'],'rounds',rounds,
        'CG',sum(p.cg_iterations_total for p in ps),'gap',result['max_quality_gap'],
        'bytes',comm,'seconds',seconds,flush=True)
if __name__=='__main__':main()
