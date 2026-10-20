"""Run author-source FedSSL-AMC on PeerSim, with final-only local tests."""
import argparse
from pathlib import Path
import json
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--profile',choices=['smoke','author'],default='smoke')
    parser.add_argument('--topology',choices=['ring','full','random_kout'],default='ring')
    parser.add_argument('--activation',choices=['ordered','shuffle'],default='shuffle')
    parser.add_argument('--rounds',type=int,default=10)
    parser.add_argument('--local-steps',type=int)
    parser.add_argument('--threads',type=int,default=3)
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):parser.error('Choose a fresh output directory')
    if args.threads<1:parser.error('Positive thread count required')
    import torch
    import joblib
    from paper_reference.ssl import SSLConfig
    from paper_reference.iq_data import author_synthetic_clients
    from src.fedssl_simulation import FedSSLSimulation
    torch.set_num_threads(args.threads)
    config=SSLConfig(seed=args.seed,rounds=args.rounds)
    scale=1.
    if args.profile=='smoke':
        scale=.1;config.local_steps=3;config.batch_size=8
    if args.local_steps is not None:config.local_steps=args.local_steps
    config.validate()
    args.out.mkdir(parents=True,exist_ok=True)
    u,l,t=author_synthetic_clients(args.seed,scale)
    simulation=FedSSLSimulation(u,l,t,config,args.topology,args.activation)
    started=time.perf_counter()
    result=simulation.run()
    result['training_seconds']=time.perf_counter()-started
    result['per_node_accuracy']=simulation.evaluate_final()
    result['average_accuracy']=sum(result['per_node_accuracy'])/len(u)
    result.update(method='FedSSL-AMC',runtime='PeerSim CDSimulator',profile=args.profile,
        source_profile='author_release_7cf7dc2',converged=None,stop_reason='author_fixed_round_budget',
        train_samples=[len(x) for x in u],labeled_samples=[len(y) for x,y in l],
        test_samples=[len(y) for x,y in t],test_evaluations=[p.test_evaluations for p in simulation.protos],
        topology=args.topology,activation=args.activation,reproduction_claim=False)
    torch.save(simulation.protos[0].model.state_dict(),args.out/'encoder.pt')
    joblib.dump([p.head for p in simulation.protos],args.out/'local_linear_svms.joblib')
    (args.out/'metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('per_node_metadata','history')},indent=2),flush=True)


if __name__=='__main__':main()
