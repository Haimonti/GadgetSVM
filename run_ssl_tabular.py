"""Run the explicitly labelled FedSSL-AMC tabular adaptation on a9a."""
import argparse,json,time,os,hashlib
from pathlib import Path


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--resume-dir',type=Path)
    ap.add_argument('--rounds',type=int,default=1500)
    ap.add_argument('--local-steps',type=int,default=10)
    ap.add_argument('--threads',type=int,default=3)
    ap.add_argument('--window',type=int,default=50)
    ap.add_argument('--rtol',type=float,default=.001)
    args=ap.parse_args()
    if args.out.exists() and any(args.out.iterdir()):ap.error('Use fresh output directory')
    if args.threads<1:ap.error('Positive thread count required')
    for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]=str(args.threads)
    os.environ['GADGETSVM_DATA_DIR']=str(args.data_dir)
    import numpy as np
    import torch,joblib
    from src.config import CONFIG
    from src.data_sharding import load_shards
    from paper_reference.ssl_tabular import TabularSSLConfig,embeddings
    from src.ssl_tabular_gossip_simulation import GossipTabularSSLSimulation
    torch.set_num_threads(args.threads)
    CONFIG.update(DATASET='a9a')
    shards=load_shards(CONFIG)
    u=[p['X_csr'] for p in shards]
    l=[(p['X_csr'],p['y']) for p in shards]
    t=[(p['X_test'],p['y_test']) for p in shards]
    cfg=TabularSSLConfig(input_dim=u[0].shape[1],seed=CONFIG['SEED'],rounds=args.rounds,
        local_steps=args.local_steps,stability_window=args.window,rtol=args.rtol)
    source_data=[dict(name=Path(CONFIG[k]).name,sha256=hashlib.sha256(Path(CONFIG[k]).read_bytes()).hexdigest()) for k in ['A9A_TRAIN_PATH','A9A_TEST_PATH']]
    if args.resume_dir:ap.error('Old flooding checkpoints cannot be resumed as gossip experiments')
    sim=GossipTabularSSLSimulation(u,l,t,cfg,CONFIG['TOPOLOGY'],CONFIG['ACTIVATION'],CONFIG['GOSSIP_K'])
    started=time.perf_counter();result=sim.run();result['training_seconds']=time.perf_counter()-started
    # Evaluate each final peer model once only after operational convergence.
    prefix=[[(c,list(d)) for c,d,size in p.communication_trace[:100]] for p in sim.protos]
    result['neighbor_schedule_first_100']=prefix
    result['neighbor_schedule_sha256_first_100']=hashlib.sha256(json.dumps(prefix).encode()).hexdigest()
    result['per_node_accuracy']=sim.evaluate_final() if result['converged'] else []
    result['average_accuracy']=float(np.mean(result['per_node_accuracy'])) if result['per_node_accuracy'] else None
    result['accuracy_status']='after_operational_plateau' if result['converged'] else 'not_evaluated_not_converged'
    # Final train hinge is on the full TRAIN set for every local frozen head.
    ytrain=np.concatenate([y for x,y in l])
    result['per_node_train_hinge']=[];feature_std=[]
    for p in sim.protos:
        ztrain=np.concatenate([embeddings(p.model,x) for x in u])
        result['per_node_train_hinge'].append(float(np.maximum(0,1-ytrain*p.head.decision_function(ztrain)).mean()))
        feature_std.append(float(ztrain.std(axis=0).mean()))
    result['final_train_hinge']=float(np.mean(result['per_node_train_hinge']))
    result['embedding_mean_feature_std']=float(np.mean(feature_std))
    result.update(method='fedssl_tabular',schedule_policy='event-hashed activation and neighbor selection, independent of learning RNG',display_name='FedSSL-AMC (tabular + gossip adaptation)',dataset='a9a',
        rounds=result['training_rounds'],train_samples=sum(len(y) for x,y in l),test_samples=len(t[0][1]),
        features=cfg.input_dim,threads=args.threads,test_evaluations=sum(p.test_evaluations for p in sim.protos),
        per_node_test_evaluations=[p.test_evaluations for p in sim.protos],seed=cfg.seed,
        configuration=result['per_node_metadata'][0]['config'],reproduction_claim=False,
        comparability='nonlinear learned features, local SVM intercept and train-only C tuning; different capacity/objective from raw linear SVMs',
        source_data=source_data)
    args.out.mkdir(parents=True,exist_ok=True)
    torch.save({i:p.model.state_dict() for i,p in enumerate(sim.protos)},args.out/'encoders.pt')
    joblib.dump([p.head for p in sim.protos],args.out/'local_linear_svms.joblib')
    (args.out/'fedssl_tabular_metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf8')
    print(json.dumps({k:result[k] for k in ['method','converged','stop_reason','rounds','average_accuracy','final_train_hinge','total_comm_bytes','training_seconds']},indent=2),flush=True)


if __name__=='__main__':main()
