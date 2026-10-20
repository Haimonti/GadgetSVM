"""Run the paper references independently of the retired approximations.

Examples:
  python run_paper_reference.py ssl --profile smoke --out results/iq_reference
  python run_paper_reference.py fdr --train a9a --test a9a.t --out results/fdr_reference
  python run_paper_reference.py fdr --transport ring --out results/fdr_toy_p2p

The ring transport performs complete per-epoch all-gather before advancing an
algorithm round. Link rounds are reported separately from algorithm rounds.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('method', choices=['fdr', 'ssl'])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--transport', choices=['server','ring'], default='server')
    parser.add_argument('--profile', choices=['smoke','author'], default='smoke')
    parser.add_argument('--train', type=Path)
    parser.add_argument('--test', type=Path)
    parser.add_argument('--rounds', type=int)
    parser.add_argument('--rho', type=float, default=.01)
    parser.add_argument('--tau', type=float, default=0.)
    parser.add_argument('--threads', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.threads < 1 or (args.rounds is not None and args.rounds < 1):
        parser.error('Positive threads and rounds required')
    if args.out.exists() and any(args.out.iterdir()):
        parser.error('Choose a fresh output directory')
    if (args.train is None) != (args.test is None):
        parser.error('--train and --test must be supplied together')
    if args.method=='ssl' and args.train:
        parser.error('SSL currently validates on synthetic I/Q; a9a adaptation is deferred')
    args.out.mkdir(parents=True, exist_ok=True)
    graph = [[(i-1)%4,(i+1)%4] for i in range(4)] if args.transport=='ring' else None
    started=time.perf_counter()
    if args.method=='fdr':
        from paper_reference.fdr import train
        if args.train:
            from sklearn.datasets import load_svmlight_files
            X,y,Xt,yt=load_svmlight_files([str(args.train),str(args.test)])
            if not np.isin(y,[-1,1]).all() or not np.isin(yt,[-1,1]).all():
                parser.error('FDR labels must be -1/+1')
            parts=np.array_split(np.random.default_rng(args.seed).permutation(len(y)),4)
            shards=[(X[ix],y[ix]) for ix in parts]
            dataset=args.train.name
        else:
            shards=[(np.array([[2+i*.1],[-2-i*.1],[1.5],[-1.5]]),
                     np.array([1.,-1.,1.,-1.])) for i in range(4)]
            Xt,yt=shards[0];dataset='toy_validation'
        result=train(shards,rounds=args.rounds or 100,rho=args.rho,tau=args.tau,adjacency=graph)
        seconds=time.perf_counter()-started
        accuracy=float(np.mean(np.where(Xt@result.weight>=0,1,-1)==yt))
        np.save(args.out/'weight.npy',result.weight)
        metadata=dict(method='FDR-SVM',paper='2410.03877v3',dataset=dataset,
            transport=args.transport,train_samples=[len(y) for X,y in shards],
            rounds=len(result.history),rho=args.rho,tau=args.tau,dual_norm=1,
            epsilon=[1/(10*len(y)) for X,y in shards],kappa=1.,alpha=[.25]*4,
            converged=None,stop_reason=result.stop_reason,final_test_accuracy=accuracy,
            test_evaluations=1,training_seconds=seconds,
            communication_bytes=result.communication_bytes,link_rounds=result.link_rounds,
            history=result.history)
    else:
        import torch
        from paper_reference.ssl import SSLConfig
        from paper_reference.iq_data import author_synthetic_clients
        from methods.fedssl_amc import run_iq_clients
        torch.set_num_threads(args.threads)
        config=SSLConfig(seed=args.seed)
        scale=1.
        if args.profile=='smoke':
            scale=.1;config.local_steps=10;config.batch_size=8
        if args.rounds:
            config.rounds=args.rounds
        u,l,t=author_synthetic_clients(args.seed,scale)
        result=run_iq_clients(u,l,t,config,graph)
        metadata=result['metadata']
        metadata.update(method='FedSSL-AMC',transport=args.transport,
            dataset='author_release_synthetic_CFO_IQ',profile=args.profile,data_scale=scale,
            train_samples=[len(x) for x in u],labeled_samples=[len(y) for x,y in l],
            test_samples=[len(y) for x,y in t],
            final_test_accuracy=result['per_client_accuracy'],test_evaluations=4,
            total_pipeline_seconds=time.perf_counter()-started,
            dataset_assumptions=['Pinned author signal functions and CFO mixtures',
                                 'Independent unlabeled/labeled/test generation'],
            reproduction_claim=False)
        torch.save(result['encoder'].state_dict(),args.out/'encoder.pt')
    metadata.update(seed=args.seed,threads=args.threads)
    (args.out/'metrics.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(json.dumps(metadata,indent=2),flush=True)


if __name__=='__main__':
    main()
