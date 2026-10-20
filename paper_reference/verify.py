"""Remote numerical checks. The I/Q run is a workflow validation, not a paper score reproduction."""
import sys
sys.stderr = sys.stdout
import json
import math
import warnings
from pathlib import Path
import numpy as np
import torch
from .fdr import train, centralized, LocalProblem
from .transport import ExactAggregation
from .ssl import (SSLConfig, contrastive_loss, triplet_indices, train_encoder,
                  encoder, fit_local_heads, evaluate_local_heads, iq_tensor)
from .iq_data import synthetic_clients


def main():
    torch.set_num_threads(3)
    warnings.filterwarnings('ignore', message='.*weight_norm.*deprecated.*')
    report = {}
    ring = [[1,3],[0,2],[1,3],[2,0]]
    # An analytic scalar quadratic reference catches incorrect constraint/sign changes.
    # y*x=2, eps=.1, kappa=100: optimum w=.5 for rho=1, center=0.
    p = LocalProblem(np.array([[2.],[-2.]]), np.array([1.,-1.]), .1,100.,1.,dual_norm=1)
    analytic = p.solve(np.zeros(1),np.zeros(1))[0][0]
    assert abs(analytic-.5)<1e-5, analytic
    # Finite label cost must change the solution; not a quadratic penalty approximation.
    flip = LocalProblem(np.array([[2.],[-2.]]),np.array([1.,-1.]),.4,.1,1.,dual_norm=1)
    flip_w = flip.solve(np.zeros(1),np.zeros(1))[0][0]
    assert abs(flip_w)<1e-5, flip_w
    shards = [(np.array([[2.+i*.1],[-2.-i*.1],[1.5],[-1.5]]),np.array([1.,-1.,1.,-1.])) for i in range(4)]
    direct, optimal = centralized(shards,tau=.18)
    a = train(shards,rounds=400,rho=.01,tau=.18)
    assert np.linalg.norm(a.weight-direct)<2e-4, (a.weight,direct)
    assert a.history[-1]['primal_residual']<1e-5,a.history[-1]
    assert max(h['weighted_dual_norm'] for h in a.history)<1e-12
    b = train(shards,rounds=6,rho=.01,tau=.18)
    gossip = train(shards,rounds=6,rho=.01,tau=.18,adjacency=ring)
    assert b.history == gossip.history
    assert gossip.link_rounds == 12
    try:
        train(shards,rounds=1,rho=1.,tau=.01)
        raise AssertionError('Invalid SC bound accepted')
    except ValueError:
        pass
    try:
        ExactAggregation([[],[]]).weighted([np.ones(1),np.ones(1)],[.5,.5])
        raise AssertionError('Disconnected graph accepted')
    except ValueError:
        pass
    # Independent accounting on a directed 3-cycle: 3 origins * 2 hops.
    counter = ExactAggregation([[1],[2],[0]],header_bytes=0)
    counter.gather([0,1,2],[8,8,8])
    assert counter.bytes==48 and counter.link_rounds==2
    nonuniform=ExactAggregation([[1],[0]]).weighted([np.array([1.]),np.array([5.])],[.25,.75])
    assert all(np.array_equal(v,np.array([4.])) for v in nonuniform)
    report['fdr'] = dict(analytic_weight=float(analytic),finite_label_cost_weight=float(flip_w),
        centralized_weight=direct.tolist(),admm_weight=a.weight.tolist(),
        centralized_objective=optimal,final_residual=a.history[-1]['primal_residual'],
        server_p2p_identical=True,server_bytes=b.communication_bytes,
        p2p_bytes=gossip.communication_bytes,p2p_link_rounds=gossip.link_rounds,
        validation='toy convex optimum and exact transport; not UCI reproduction')
    zr=torch.zeros((2,3));zp=torch.zeros((2,3));zn=torch.zeros((2,10,3))
    assert abs(float(contrastive_loss(zr,zp,zn))-11*math.log(2))<1e-6
    ref, neg=triplet_indices(3,100,10,np.random.default_rng(2))
    assert np.all(ref[:,None]!=neg)
    try:
        iq_tensor(np.ones((5,123)))
        raise AssertionError('Tabular data accepted')
    except ValueError:
        pass
    # No future signal may change the output prefix of the causal stack.
    config=SSLConfig(rounds=2,local_steps=2,batch_size=2,channels=4,
                     depth=2,reduced_size=6,embedding_dim=8,negatives=2,svm_C=1.,svm_cv=0)
    with torch.random.fork_rng():
        torch.manual_seed(4)
        m=encoder(config)
    x=torch.randn(2,2,32,dtype=torch.float64);changed=x.clone();changed[:,:,16:]+=100
    stack=m.network[0]
    assert torch.equal(stack(x)[:,:,:16],stack(changed)[:,:,:16])
    u,l,t=synthetic_clients(seed=2,scale=.001)
    u[0]=u[0][:70]  # verify dataset-size weighting with unequal local sizes
    server, meta=train_encoder(u,config)
    peer, pmeta=train_encoder(u,config,adjacency=ring)
    assert all(torch.equal(v,peer.state_dict()[k]) for k,v in server.state_dict().items())
    before={k:v.clone() for k,v in server.state_dict().items()}
    heads=fit_local_heads(server,l,config)
    assert len(heads)==4 and len({id(h) for h in heads})==4
    assert all(torch.equal(v,server.state_dict()[k]) for k,v in before.items())
    accuracy=evaluate_local_heads(server,heads,t)
    # Also exercise the full upstream architecture on I/Q, with a small compute budget.
    full=SSLConfig(rounds=1,local_steps=1,batch_size=2,svm_C=1.,svm_cv=0)
    full_model,full_meta=train_encoder(u,full)
    full_heads=fit_local_heads(full_model,l,full)
    full_accuracy=evaluate_local_heads(full_model,full_heads,t)
    report['ssl']=dict(server_p2p_identical=True,causality_verified=True,
        train_counts=[len(x) for x in u],labeled_counts=[len(y) for x,y in l],
        test_counts=[len(y) for x,y in t],tiny_accuracy=accuracy,
        full_backbone_accuracy=full_accuracy,
        full_backbone_parameters=sum(p.numel() for p in full_model.parameters()),
        full_backbone=full_meta,tiny_metadata=meta,p2p_metadata=pmeta,
        validation='I/Q synthetic workflow smoke; architecture details unresolved; NOT paper accuracy reproduction')
    out=Path('verification');out.mkdir(exist_ok=True)
    (out/'checks.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
