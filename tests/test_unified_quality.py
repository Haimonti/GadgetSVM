import json
from pathlib import Path
import tempfile
import time
import unittest
from types import SimpleNamespace
import numpy as np
import scipy.sparse as sp
from src.unified_quality import objectives,CommonQualityEvaluator,LinearNoBiasBDSVM,subspace_excess_bound
from src.peersim_python.core import Network,GeneralNode
from src.peersim_python.idle_protocol import IdleProtocol
from src.peersim_python.cdsim import CDState


class UnifiedQualityTests(unittest.TestCase):
    def test_budget_infeasibility_bound_matches_an_analytic_constrained_optimum(self):
        X=sp.eye(4,format='csr');y=np.array([1.,-1.,1.,-1.]);opt=y/4
        basis=np.array([[1.,0.,0.,0.]])
        distance,radius,bound=subspace_excess_bound(opt,basis,1.,0.)
        restricted=np.array([.25,0.,0.,0.])
        exact=float(objectives(X,y,restricted,1.)[0][0]-objectives(X,y,opt,1.)[0][0])
        self.assertEqual(radius,0.)
        self.assertLessEqual(bound,exact)
        self.assertAlmostEqual(bound,exact,places=6)
        self.assertGreater(bound,.001)

    def test_primal_matches_known_dual_optimum(self):
        X=sp.eye(4,format='csr');y=np.array([1.,-1.,1.,-1.]);w=y/4
        value,hinge=objectives(X,y,w,1.)
        self.assertAlmostEqual(value[0],.875)
        self.assertAlmostEqual(hinge[0],.75)
        lower=1.-.5*float(w@w)
        self.assertAlmostEqual(value[0],lower)

    def test_stagnant_poor_model_and_one_poor_peer_cannot_pass(self):
        X=sp.eye(4,format='csr');y=np.array([1.,-1.,1.,-1.])
        good=y/4;bad=np.zeros(4)
        def install(weights):
            nodes=[]
            for i,w in enumerate(weights):
                p=SimpleNamespace(w=w,comm_bytes=0,stopped=False)
                p.accuracy=lambda: (_ for _ in ()).throw(AssertionError('test labels accessed'))
                n=GeneralNode([IdleProtocol(),p]);n.setIndex(i);nodes.append(n)
            Network.node=nodes;Network.len=len(nodes)
        install([good,bad]);CDState.setCycle(199)
        gate=CommonQualityEvaluator(X,y,1.,.875,.001,1000,time.perf_counter(),every=1)
        self.assertFalse(gate.execute())
        install([bad,bad]);self.assertFalse(gate.execute())
        install([good,good]);self.assertTrue(gate.execute())
        self.assertTrue(all(Network.get(i).getProtocol(1).stopped for i in range(2)))

    def test_no_bias_bdsvm_has_correct_regularizer_and_payload_dimensions(self):
        from src.peersim_python.core import CommonState
        CommonState.initializeRandom(42)
        X=sp.csr_matrix([[1.,0.],[0.,1.],[1.,1.]])
        y=np.array([1.,-1.,1.]);lam=.1;n=3;C=1/(2*lam*n)
        p=LinearNoBiasBDSVM(P=2,C=C,kernel='linear',preimage='unit')
        p.set_data(X,y,X[:0],y[:0],lam);p.configure_network(0,n,1)
        self.assertEqual(p.Km.shape,(3,2));self.assertEqual(p.Kpp.shape,(2,2))
        p.beta=np.array([.2,-.3]);w=p.p.T@p.beta
        native_reg=float(p.beta@p.Kpp@p.beta)/(4*C*n)
        self.assertAlmostEqual(native_reg,.5*lam*float(w@w),places=14)
        self.assertEqual(len(p.clone().beta) if hasattr(p.clone(),'beta') else p.clone().P,2)

    def test_bdsvm_whitening_preserves_the_unridged_native_irls_model_update(self):
        from methods.bdsvm import _make_preimages,_worker_contribution
        from src.peersim_python.core import CommonState
        CommonState.initializeRandom(42)
        rng=np.random.RandomState(10);X=sp.csr_matrix(rng.randn(30,7));y=np.where(X.toarray()[:,0]>0,1.,-1.)
        C=1/(2*.1*len(y));basis=_make_preimages(5,7,0,kind='unit')
        K=np.asarray(X@basis.T);A,b=_worker_contribution(K,y,np.zeros(5),C,ey_floor=.01)
        expected=basis.T@(.5*np.linalg.solve(A+basis@basis.T,b))
        p=LinearNoBiasBDSVM(P=5,C=C,rho=.5,kernel='linear',preimage='unit',ey_floor=.01)
        p.set_data(X,y,X[:0],y[:0],.1);p.configure_network(0,len(y),1);p.local_update()
        np.testing.assert_allclose(p.p@p.p.T,np.eye(5),atol=1e-12)
        np.testing.assert_allclose(p.p.T@p.beta,expected,atol=1e-12)

    def test_gadget_keeps_all_peers_active_under_common_quality_gate(self):
        from src.baselines.gadget.simulation import GadgetSimulation
        from src.baselines.gadget.native_cycle import FullNextCycle
        from src.config import CONFIG
        rng=np.random.RandomState(1);X=sp.csr_matrix(rng.randn(40,4));y=np.where(X.toarray()[:,0]>0,1.,-1.)
        shards=[dict(X_csr=X[ix],y=y[ix],X_test=X[:0],y_test=y[:0],n_local=len(ix)) for ix in np.array_split(np.arange(40),10)]
        c=dict(CONFIG);c.update(NUM_WORKERS=10,SEED=42,TOPOLOGY='ring',GOSSIP_K=2,
            LAMBDA=.1,SDCA_LOCAL_STEPS=4,OBJ_THRESHOLD=.01,STOP_ON_THRESHOLD=False,
            STOP_ON_PLATEAU=False)
        sim=GadgetSimulation(c,shards).build().sim;sim.runner=FullNextCycle('shuffle')
        gate=CommonQualityEvaluator(X,y,.1,-100.,.001,160,time.perf_counter(),every=10)
        sim.controls=[gate];sim.cycles=160;rounds=sim.nextExperiment()
        self.assertEqual(rounds,160);self.assertFalse(gate.reached)
        ps=[Network.get(i).getProtocol(1) for i in range(10)]
        self.assertTrue(all(p.t-1==640 and not p.stopped for p in ps))
        self.assertEqual(sum(p.comm_bytes for p in ps),20*160*4*8)


if __name__=='__main__':unittest.main()
