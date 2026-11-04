"""Shared linear-SVM training quality benchmark (all workers, no test stopping)."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import warnings

METHODS = ['centralized','sdca','bdsvm','fedavg','cocoa','cocoa_plus','gadget','cola','choco']
CHOCO_LR = {'w8a':10.,'gisette':.1,'real-sim':10.,'rcv1':1.,'webspam':10.}


def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf8')
    temp.replace(path)


def load_training(config):
    """Same shuffle, labels, holdout and shard order as src.data_sharding.

    Two-file test data is loaded only after training has stopped successfully.
    """
    import numpy as np
    from sklearn.datasets import load_svmlight_file
    from src.data_sharding import _to_pm1
    from run_compare import DATASET_PATH_KEYS
    ds=config['DATASET'];keys=DATASET_PATH_KEYS[ds]
    X,y=load_svmlight_file(str(config[keys[0]]));y=_to_pm1(y).astype(float)
    perm=np.random.RandomState(config['SEED']).permutation(len(y))
    X=X[perm].tocsr();y=y[perm];heldout=None
    if len(keys)==1:
        n_test=int(config['TEST_FRACTION']*len(y))
        heldout=(X[:n_test].tocsr(),y[:n_test])
        X,y=X[n_test:].tocsr(),y[n_test:]
    h=hashlib.sha256()
    for a in [X.indptr,X.indices,X.data,y]:h.update(memoryview(a))
    return X,y,heldout,h.hexdigest()


def test_data(config, features, heldout):
    if heldout is not None:return heldout
    from sklearn.datasets import load_svmlight_file
    from src.data_sharding import _to_pm1
    from run_compare import DATASET_PATH_KEYS
    Xt,yt=load_svmlight_file(str(config[DATASET_PATH_KEYS[config['DATASET']][1]]),n_features=features)
    return Xt.tocsr(),_to_pm1(yt).astype(float)


def make_reference(X,y,config,out,target,digest):
    import numpy as np
    from sklearn.svm import LinearSVC
    from sklearn.exceptions import ConvergenceWarning
    from p2p._linear_fast import dual_steps
    from src.unified_quality import objectives
    lam=config['LAMBDA'];n,d=X.shape
    cached=out/'reference.json'
    if cached.exists():
        old=json.loads(cached.read_text())
        if old['train_digest']==digest and old['target']==target and old['reference_gap']<=target/10:
            arrays=np.load(out/'reference.npz');a=arrays['signed_alpha'];w=arrays['weights']
            assert np.all((a*y>=0)&(a*y<=1))
            mapped=np.asarray(X.T@a).ravel()/(lam*n)
            lower=float((a*y).mean()-.5*lam*(mapped@mapped))
            assert abs(lower-old['dual_lower'])<1e-10
            assert abs(float(objectives(X,y,w,lam)[0][0])-old['primal_upper'])<1e-10
            print('Reused certified reference',config['DATASET'],config['SEED'],flush=True)
            return
    clf=LinearSVC(C=1/(lam*n),loss='hinge',fit_intercept=False,dual=True,
        max_iter=1000000,tol=1e-8,random_state=config['SEED'])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always',ConvergenceWarning);clf.fit(X,y)
    refw=clf.coef_[0].copy();upper=float(objectives(X,y,refw,lam)[0][0])
    alpha=np.zeros(n);w=np.zeros(d);rng=np.random.default_rng(81000+config['SEED'])
    batch=max(4096,min(n,65536));draws_done=0;lower=0.;trace=[]
    for block in range(10000):
        draws=rng.integers(n,size=batch,dtype=np.int64)
        alpha,w=dual_steps(X.indptr,X.indices,X.data,y,alpha,w,draws,lam*n,1.,1.)
        draws_done+=batch
        if block%5 and block!=9999:continue
        # Rebuild exactly before evaluating a feasible dual lower bound.
        unsigned=np.clip(alpha*y,0.,1.);alpha=y*unsigned
        w=np.asarray(X.T@alpha).ravel()/(lam*n)
        lower=float(unsigned.mean()-.5*lam*(w@w))
        j=float(objectives(X,y,w,lam)[0][0])
        if j<upper:upper=j;refw=w.copy()
        if lower>upper+1e-8:raise ValueError('Invalid primal-dual reference bounds')
        trace.append({'draws':draws_done,'upper':upper,'lower':lower,'gap':upper-lower})
        if block%100==0:print('reference',config['DATASET'],config['SEED'],trace[-1],flush=True)
        if upper-lower<=target/100:break
    if upper-lower>target/10:raise RuntimeError('Reference is too loose for certified common target')
    out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'reference.npz',weights=refw,signed_alpha=alpha)
    r={'dataset':config['DATASET'],'seed':config['SEED'],'lambda':lam,'fit_intercept':False,
       'target':target,'train_samples':n,'features':d,'train_digest':digest,
       'primal_upper':upper,'dual_lower':lower,'reference_gap':upper-lower,
       'dual_feasible':bool(np.all((alpha*y>=0)&(alpha*y<=1))),
       'reference_draws':draws_done,'reference_solver_warning':bool(caught),'trace':trace}
    write(out/'reference.json',r)


def certify_budget_space(X,config,out,target):
    """Training-only strong-convexity bound for a fixed budget subspace.

    A reference primal-dual gap delta bounds ||w_ref-w_star|| by
    sqrt(2*delta/lambda). Distance from w_ref to the budget subspace therefore
    gives a lower bound on the objective error of *every* budget model.
    This is an offline benchmark certificate, not a new learner update.
    """
    import numpy as np
    from methods.bdsvm import _make_preimages
    from src.bdsvm_simulation import params_for
    from src.unified_quality import subspace_excess_bound
    ref=json.loads((out/'reference.json').read_text());d=X.shape[1]
    budget=min(params_for(config['DATASET'])['P'],d)
    r={'dataset':config['DATASET'],'seed':config['SEED'],'basis_budget':budget,
       'features':d,'full_linear_span':budget==d,'target':target,
       'reference_gap':ref['reference_gap'],'quality_unattainable':False}
    if budget<d:
        w=np.load(out/'reference.npz')['weights']
        basis=_make_preimages(budget,d,0,kind='unit')
        distance,radius,error_bound=subspace_excess_bound(
            w,basis,config['LAMBDA'],ref['reference_gap'])
        r.update(reference_distance_to_subspace=distance,
            reference_error_radius=radius,minimum_objective_excess_bound=error_bound,
            quality_unattainable=error_bound>target,
            basis_sha256=hashlib.sha256(memoryview(basis)).hexdigest())
    write(out/'budget_certificate.json',r)
    return r


def run(args):
    import psutil
    process=psutil.Process()
    if args.timing:process.cpu_affinity([10,11,12])
    import numpy as np
    import scipy.sparse as sp
    from src.config import CONFIG
    from src.unified_quality import objectives,CommonQualityEvaluator,LinearNoBiasBDSVM
    from p2p._linear_fast import warmup
    from run_compare import DATASET_PATH_KEYS
    config=dict(CONFIG);config.update(DATASET=args.dataset,SEED=args.seed,LAMBDA=1e-4,
        NUM_WORKERS=10,TOPOLOGY='ring',GOSSIP_K=2,WARM_START=False,ROUNDS=args.cycles,
        SDCA_LOCAL_STEPS=4096,SDCA_STEP_SCALE=1.,STOP_ON_THRESHOLD=False,
        STOP_ON_PLATEAU=False,OBJ_THRESHOLD=.01,EVAL_EVERY=10,
        CHOCO_INITIAL_LR=CHOCO_LR.get(args.dataset,.1),
        CHOCO_ESTIMATE='linear' if args.dataset=='gisette' else 'final')
    for k in config:
        if k.endswith('_PATH'):config[k]=args.data_dir/Path(str(config[k]).replace('\\','/')).name
    X,y,heldout,digest=load_training(config)
    warmup()
    if args.method=='reference':
        make_reference(X,y,config,args.reference,args.target,digest);return
    if args.method=='feasibility':
        certify_budget_space(X,config,args.reference,args.target);return
    ref=json.loads((args.reference/'reference.json').read_text())
    assert ref['train_digest']==digest and ref['lambda']==config['LAMBDA']
    assert ref['target']==args.target and ref['dual_feasible']
    assert ref['reference_gap']<=args.target/10
    lower=ref['dual_lower'];lam=config['LAMBDA']
    if args.method=='bdsvm':
        budget=json.loads((args.reference/'budget_certificate.json').read_text())
        if budget['quality_unattainable']:
            # No finite time/round/byte cost to this target exists in the
            # retained model class. Do not draw a misleading zero-cost bar.
            r={'dataset':args.dataset,'seed':args.seed,'method':'bdsvm',
               'converged':False,'stop_reason':'budget_quality_unattainable',
               'rounds':None,'round_cap':args.cycles,'target':args.target,
               'reference_dual_lower':lower,'reference_primal_upper':ref['primal_upper'],
               'reference_gap':ref['reference_gap'],'budget_certificate':budget,
               'train_digest':digest,'train_samples':len(y),'features':X.shape[1],
               'fit_intercept':False,'lambda':lam,'zero_score_label':-1,
               'average_accuracy':None,'per_node_accuracy':[],'test_evaluations':0,
               'total_comm_bytes':None,'training_seconds':None,'history':[],
               'final_train_hinge':None,'max_quality_gap':None,
               'phase':'proof_preflight','execution_hardware':{'cpu_affinity':process.cpu_affinity(),'threads':args.threads}}
            write(args.out/('bdsvm_timing.json' if args.timing else 'bdsvm_metrics.json'),r)
            return
    empty=sp.csr_matrix((0,X.shape[1]));shards=[]
    for ids in np.array_split(np.arange(len(y)),10):
        shards.append({'X_csr':X[ids].tocsr(),'y':y[ids],
            'X_test':empty,'y_test':np.zeros(0),'n_local':len(ids)})
    method=args.method;history=[];params={};adjacency=[];raw_model_space_bound=None
    # Exclude Python imports and JIT warmup from every algorithm's timer.
    from sklearn.svm import LinearSVC
    from sklearn.exceptions import ConvergenceWarning
    from src.peersim_python.simulation import Simulation
    from src.bdsvm_simulation import BDSVMSimulation,params_for
    from src.fedavg_simulation import FedAvgSimulation
    from src.cocoa_simulation import CoCoASimulation
    from src.baselines.gadget.simulation import GadgetSimulation
    from src.baselines.gadget.native_cycle import FullNextCycle
    from src.linear_baseline_simulation import LinearBaselineSimulation
    started=time.perf_counter()
    if method=='centralized':
        from sklearn.svm import LinearSVC
        from sklearn.exceptions import ConvergenceWarning
        converged=False
        for tolerance in [1e-5,1e-7,1e-9]:
            clf=LinearSVC(C=1/(lam*len(y)),loss='hinge',fit_intercept=False,
                dual=True,max_iter=1000000,tol=tolerance,random_state=args.seed)
            with warnings.catch_warnings(record=True):
                warnings.simplefilter('always',ConvergenceWarning);clf.fit(X,y)
            W=clf.coef_.copy();primal,hinge=objectives(X,y,W,lam)
            if float(primal[0])-lower<=args.target:converged=True;break
        seconds=time.perf_counter()-started;rounds=None;comm=0;messages=0
        reason='quality_target' if converged else 'central_solver_limit'
        params={'solver':'LinearSVC','C':1/(lam*len(y)),'tol':tolerance,
                'iterations':int(clf.n_iter_)}
    else:
        from src.peersim_python.core import Network
        if method=='sdca':
            from src.peersim_python.simulation import Simulation
            assembly=Simulation(config,shards)
        elif method=='bdsvm':
            from src.bdsvm_simulation import BDSVMSimulation,params_for
            params=params_for(args.dataset);params['C']=1/(2*lam*len(y))
            assembly=BDSVMSimulation(config,shards,params=params,protocol_cls=LinearNoBiasBDSVM)
        elif method=='fedavg':
            from src.fedavg_simulation import FedAvgSimulation
            assembly=FedAvgSimulation(config,shards)
        elif method in ('cocoa','cocoa_plus'):
            from src.cocoa_simulation import CoCoASimulation
            assembly=CoCoASimulation(config,shards,variant=method)
        elif method=='gadget':
            from src.baselines.gadget.simulation import GadgetSimulation
            from src.baselines.gadget.native_cycle import FullNextCycle
            assembly=GadgetSimulation(config,shards).build()
            assembly.sim.runner=FullNextCycle(config.get('ACTIVATION','shuffle'))
        else:
            from src.linear_baseline_simulation import LinearBaselineSimulation
            assembly=LinearBaselineSimulation(config,shards,method)
        if assembly.sim is None:assembly.build()
        observer=CommonQualityEvaluator(X,y,lam,lower,args.target,args.cycles,started)
        assembly.sim.controls=[observer];assembly.sim.cycles=args.cycles
        rounds=assembly.sim.nextExperiment();seconds=time.perf_counter()-started
        ps=[Network.get(i).getProtocol(1) for i in range(Network.size())]
        converged=observer.reached;reason='quality_target' if converged else 'round_limit'
        primal,hinge=observer.final_objectives,observer.final_hinges
        W=np.stack([observer.weights(p) for p in ps]);history=observer.history
        comm=sum(p.comm_bytes for p in ps)
        messages=comm//(X.shape[1]*8) if method=='gadget' else sum(p.messages_sent for p in ps)
        adjacency=[[Network.get(i).getProtocol(0).getNeighbor(j).getIndex()
            for j in range(Network.get(i).getProtocol(0).degree())] for i in range(10)]
        if method=='bdsvm':params=dict(assembly.params,fit_intercept=False)
        if method=='choco':params={'initial_lr':config['CHOCO_INITIAL_LR'],
            'estimator':config['CHOCO_ESTIMATE'],'consensus_lr':.1,'batch':128,'top_fraction':.2}
    # The timer has stopped. Test predictions are never part of the stop decision.
    accuracy=[];test_count=None
    if converged:
        Xt,yt=test_data(config,X.shape[1],heldout);test_count=len(yt)
        # Match LinearSVC: score zero belongs to the negative class. A common
        # tie rule matters for datasets containing all-zero feature vectors.
        accuracy=np.mean(np.where(Xt@W.T>0,1.,-1.)==yt[:,None],axis=0).tolist()
    from threadpoolctl import threadpool_info
    config_json={k:str(v) if isinstance(v,Path) else v for k,v in config.items()}
    r={'dataset':args.dataset,'seed':args.seed,'method':method,'converged':converged,
       'stop_reason':reason,'rounds':rounds,'round_cap':args.cycles,'target':args.target,
       'objective_definition':'mean_hinge + lambda/2 * squared_weight_norm; bias=0',
       'reference_dual_lower':lower,'reference_primal_upper':ref['primal_upper'],
       'reference_gap':ref['reference_gap'],'final_train_objectives':primal.tolist(),
       'max_quality_gap':float(np.max(primal-lower)),'final_train_hinge':float(hinge.mean()),
       'train_samples':len(y),'test_samples':test_count,'features':X.shape[1],
       'train_digest':digest,'fit_intercept':False,'lambda':lam,'zero_score_label':-1,
       'average_accuracy':float(np.mean(accuracy)) if accuracy else None,
       'per_node_accuracy':accuracy,'test_evaluations':len(accuracy),
       'total_comm_bytes':comm,'messages_sent':messages,'training_seconds':seconds,
       'phase':'isolated_timing' if args.timing else 'parallel_learning',
       'execution_hardware':{'cpu_affinity':process.cpu_affinity(),'threads':args.threads,
            'threadpools':threadpool_info()},'parameters':params,'config':config_json,
       'adjacency':adjacency,'history':history,
       'local_compute_budget':{'sample_draws_per_peer_round':128 if method=='choco' else
            None if method in ('centralized','bdsvm') else 4096,
            'full_shard_IRWLS':method=='bdsvm','basis_budget':params.get('P')}}
    assert converged==bool(np.all(primal-lower<=args.target))
    assert r['test_evaluations']==((1 if method=='centralized' else 10) if converged else 0)
    path=args.out/(method+('_timing.json' if args.timing else '_metrics.json'))
    write(path,r);np.savez_compressed(path.with_suffix('.npz'),weights=W)
    print(method,reason,'rounds',rounds,'gap',r['max_quality_gap'],'seconds',seconds,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('dataset');p.add_argument('--method',choices=['reference','feasibility']+METHODS,required=True)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--target',type=float,default=1e-3)
    p.add_argument('--cycles',type=int,default=50000);p.add_argument('--timing',action='store_true')
    p.add_argument('--threads',type=int,default=3);p.add_argument('--data-dir',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    run(args)


if __name__=='__main__':main()
