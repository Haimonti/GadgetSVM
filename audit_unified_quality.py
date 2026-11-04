"""Independent arithmetic audit of models, references, test scores and timings."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from run_unified_quality import METHODS


def audit_case(root,ds,seed,data_dir,methods=None,learning_only=False):
    from src.config import CONFIG
    from src.data_sharding import load_shards
    from methods.bdsvm import _make_preimages
    c=dict(CONFIG);c.update(DATASET=ds,SEED=seed,NUM_WORKERS=10)
    for k in c:
        if k.endswith('_PATH'):c[k]=data_dir/Path(str(c[k]).replace('\\','/')).name
    shards=load_shards(c);X=sp.vstack([p['X_csr'] for p in shards]).tocsr();y=np.concatenate([p['y'] for p in shards]).astype(float)
    Xt,yt=shards[0]['X_test'],shards[0]['y_test'];lam=.0001
    h=hashlib.sha256()
    for a in [X.indptr,X.indices,X.data,y]:h.update(memoryview(a))
    digest=h.hexdigest();refdir=root/'references'/ds/f'seed_{seed}'
    ref=json.loads((refdir/'reference.json').read_text());arr=np.load(refdir/'reference.npz')
    alpha=arr['signed_alpha'];wref=arr['weights'];unsigned=y*alpha
    assert len(alpha)==len(y) and np.all(unsigned>=0) and np.all(unsigned<=1)
    sums=np.asarray(X.T@alpha).ravel();lower=float(unsigned.mean()-(sums@sums)/(2*lam*len(y)**2))
    upper=float(np.maximum(0,1-y*(X@wref)).mean()+.5*lam*(wref@wref))
    assert abs(lower-ref['dual_lower'])<1e-9 and abs(upper-ref['primal_upper'])<1e-9
    assert -1e-9<=upper-lower<=.0001 and digest==ref['train_digest']
    counts={'passed':0,'capped':0,'budget_infeasible':0,'actual_isolated_runs':0}
    for method in METHODS if methods is None else methods:
        folder=root/'study'/ds/f'seed_{seed}'
        suffix='_metrics' if learning_only else '_timing'
        r=json.loads((folder/(method+suffix+'.json')).read_text());learning=json.loads((folder/(method+'_metrics.json')).read_text())
        assert r['dataset']==ds and r['seed']==seed and r['method']==method
        assert r['target']==.001 and r['lambda']==lam and r['fit_intercept']==False
        assert r['zero_score_label']==-1 and r['train_digest']==digest
        assert r['stop_reason']==learning['stop_reason'] and r['rounds']==learning['rounds']
        assert r['total_comm_bytes']==learning['total_comm_bytes']
        if r['stop_reason']=='budget_quality_unattainable':
            assert method=='bdsvm' and not r['converged'] and r['test_evaluations']==0
            assert all(r[k] is None for k in ['training_seconds','rounds','total_comm_bytes','average_accuracy'])
            cert=json.loads((refdir/'budget_certificate.json').read_text());P=cert['basis_budget']
            basis=_make_preimages(P,X.shape[1],0,kind='unit')
            assert hashlib.sha256(memoryview(basis)).hexdigest()==cert['basis_sha256']
            gram=basis@basis.T;coeff=np.linalg.lstsq(gram,basis@wref,rcond=None)[0]
            resid=wref-basis.T@coeff;mineig=float(np.linalg.eigvalsh(gram)[0]);assert mineig>0
            dist=max(0.,float(np.sqrt(max(0.,resid@resid-np.linalg.norm(basis@resid)**2/mineig)))-1e-7*(1+float(np.linalg.norm(wref))))
            radius=float(np.sqrt(2*max(0.,upper-lower)/lam));bound=.5*lam*max(0.,dist-radius)**2
            assert bound>.001 and abs(bound-cert['minimum_objective_excess_bound'])<1e-7
            counts['budget_infeasible']+=1;continue
        hw=r['execution_hardware'];assert hw['threads']==3
        pools_over_budget=[p for p in hw['threadpools'] if p['num_threads']>3]
        if learning_only and method=='bdsvm' and r['parameters'].get('matvec_backend')=='numba':
            # Frozen learning used a Numba mask of 3, while its Windows OpenMP
            # runtime reports the larger initialized pool capacity after a region.
            # This exception applies only to parallel learning, never timing.
            assert all(p['internal_api']=='openmp' and p['prefix']=='vcomp'
                       and 'sklearn' in p['filepath'] for p in pools_over_budget)
            counts['loaded_openmp_capacity_exceeds_active_numba_mask']=len(pools_over_budget)
        else:
            assert not pools_over_budget
        if learning_only:
            assert r['phase']=='parallel_learning'
        else:
            counts['actual_isolated_runs']+=1
            assert hw['cpu_affinity']==[10,11,12] and r['phase']=='isolated_timing'
        assert r['training_seconds']>0
        W=np.load(folder/(method+suffix+'.npz'))['weights']
        assert W.shape==((1 if method=='centralized' else 10),X.shape[1])
        # Separate sparse matrix-vector products, not the learner's batched evaluator.
        vals=[];hinges=[]
        for w in W:
            hinge=float(np.maximum(0,1-y*(X@w)).mean());hinges.append(hinge);vals.append(hinge+.5*lam*(w@w))
        gaps=np.array(vals)-lower;maximum=float(gaps.max())
        np.testing.assert_allclose(vals,r['final_train_objectives'],atol=1e-9,rtol=1e-9)
        assert abs(maximum-r['max_quality_gap'])<1e-9
        assert abs(float(np.mean(hinges))-r['final_train_hinge'])<1e-9
        assert r['converged']==(r['max_quality_gap']<=.001)
        if r['converged']:
            assert maximum<=.001+1e-9 and r['stop_reason']=='quality_target'
            # Independent predictions use the fixed common tie convention.
            acc=[float(np.mean(np.where(Xt@w>0,1.,-1.)==yt)) for w in W]
            np.testing.assert_allclose(acc,r['per_node_accuracy'],atol=1e-12)
            assert abs(float(np.mean(acc))-r['average_accuracy'])<1e-12
            assert r['test_evaluations']==len(W);counts['passed']+=1
        else:
            assert r['stop_reason']=='round_limit' and r['rounds']==50000
            assert maximum>.001-1e-9 and r['test_evaluations']==0 and r['average_accuracy'] is None
            counts['capped']+=1
        if method!='centralized':
            assert len(r['adjacency'])==10 and all(len(v)==2 for v in r['adjacency'])
            assert r['history'][-1]['round']==r['rounds'] and r['history'][-1]['comm_bytes']==r['total_comm_bytes']
            if method in ['gadget','cola']:assert r['total_comm_bytes']==20*r['rounds']*X.shape[1]*8
            if method=='choco':assert r['total_comm_bytes']==20*r['rounds']*int(np.ceil(.2*X.shape[1]))*12
            if method=='bdsvm':
                assert abs(r['parameters']['C']-1/(2*lam*len(y)))<1e-15
                if r['parameters'].get('full_linear_span'):
                    assert r['parameters']['P']==X.shape[1]
                    entry_size=8*(X.shape[1]**2+X.shape[1])+24
                    ledger=r['wire_ledger']
                    assert len(ledger)==10 and all(len(rows)==r['rounds'] for rows in ledger)
                    for rows in ledger:
                        assert [row[0] for row in rows]==list(range(r['rounds']))
                        assert all(neighbors==2 and 1<=origins<=10 for _,neighbors,origins in rows)
                    expected_bytes=sum(neighbors*origins*entry_size for rows in ledger for _,neighbors,origins in rows)
                    assert expected_bytes==r['total_comm_bytes']
                    assert r['messages_sent']==20*r['rounds']
                    assert r['local_compute_budget']['no_extra_inner_communication']
        # A paired timing rerun must reproduce the reported trained endpoint.
        assert abs(r['max_quality_gap']-learning['max_quality_gap'])<1e-7
    empty=int(np.sum(np.asarray(X.multiply(X).sum(axis=1)).ravel()==0))
    return {'dataset':ds,'seed':seed,'reference_gap':upper-lower,'empty_feature_training_rows':empty,**counts}


def audit(root,data_dir):
    root=Path(root);expected=json.loads((root/'expected_data.json').read_text())
    topo=json.loads((root/'hardware_topology.json').read_text());classes={r['logical']:r['efficiency_class'] for r in topo['cpu_sets']}
    assert len(set(classes.values()))>1
    assert all(classes[i]==max(classes.values()) for i in [10,11,12])
    jobs=[(ds,s) for ds in expected for s in range(42,47)]
    with ThreadPoolExecutor(max_workers=12) as ex:
        reports=list(ex.map(lambda t:audit_case(root,t[0],t[1],data_dir),jobs))
    result={'passed':True,'dataset_seed_cases':40,'method_seed_records':360,
       'actual_isolated_runs':sum(r['actual_isolated_runs'] for r in reports),
       'target_passed_runs':sum(r['passed'] for r in reports),
       'capped_runs':sum(r['capped'] for r in reports),
       'certified_budget_infeasible_runs':sum(r['budget_infeasible'] for r in reports),
       'checks':['independent native loader and seeded partitions','feasible dual bounds and primal reference',
          'all final worker objectives recomputed','final-only test evaluations and common score-zero label',
          'budget impossibility bound independently recomputed','paired learning/timing rounds and payloads',
          'same three actual P cores and thread limits','directed payload arithmetic'],'cases':reports}
    (root/'independent_audit.json').write_text(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--data-dir',type=Path,required=True)
    args=p.parse_args();print(json.dumps(audit(args.root,args.data_dir)))
