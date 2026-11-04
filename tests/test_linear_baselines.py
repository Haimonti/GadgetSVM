import unittest
import time
import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize
from p2p.linear_baselines import metropolis_matrix, signed_coordinate_solve, gadget_local_step
from src.linear_baseline_simulation import LinearBaselineSimulation
from src.peersim_python.core import Network
from src.train_convergence import TrainConvergenceEvaluator


class NewLinearBaselinesTests(unittest.TestCase):
    def build(self, method, cycles=5):
        rng=np.random.default_rng(11)
        X=sp.csr_matrix(rng.normal(size=(24,4)));y=np.tile([-1.,1.],12)
        data=[dict(X_csr=X[ix],y=y[ix],X_test=object(),y_test=object(),n_local=len(ix))
              for ix in np.array_split(np.arange(24),3)]
        cfg=dict(NUM_WORKERS=3,SEED=42,LAMBDA=.2,GOSSIP_K=2,TOPOLOGY='ring',
                 SDCA_LOCAL_STEPS=40,ACTIVATION='shuffle',EVAL_EVERY=1,
                 CONVERGENCE_RULE='adjacent_windows',CONVERGENCE_STABILITY_WINDOW=50)
        sim=LinearBaselineSimulation(cfg,data,method).build().sim
        sim.controls=[TrainConvergenceEvaluator(1,cfg,cycles,time.perf_counter())]
        sim.cycles=cycles
        return sim

    def test_mixing_is_nonnegative_symmetric_and_doubly_stochastic(self):
        W=metropolis_matrix([[1],[0,2],[1]])
        np.testing.assert_allclose(W,W.T)
        np.testing.assert_allclose(W.sum(axis=0),1)
        self.assertTrue((W>=0).all())
        with self.assertRaises(ValueError):metropolis_matrix([[1],[]])
        with self.assertRaises(ValueError):metropolis_matrix([[],[]])

    def test_cola_coordinate_solver_matches_independent_box_qp(self):
        rng=np.random.default_rng(9);X=rng.normal(size=(7,4));y=np.array([-1,1,-1,1,-1,1,1.])
        b=rng.normal(size=7);curv=.7
        objective=lambda a:.5*curv*np.linalg.norm(X.T@a)**2-(y+b)@a
        grad=lambda a:curv*X@(X.T@a)-y-b
        opt=minimize(objective,np.zeros(7),jac=grad,bounds=[(-1,0) if v<0 else (0,1) for v in y],
                     method='L-BFGS-B',options={'ftol':1e-14,'gtol':1e-10,'maxiter':5000})
        got=signed_coordinate_solve(sp.csr_matrix(X),y,np.zeros(7),b,curv,20000,np.random.RandomState(3))
        self.assertLess(abs(objective(got)-objective(opt.x)),1e-8)
        self.assertTrue(np.all((y*got>=0)&(y*got<=1)))

    def test_cola_barriers_preserve_paper_average_invariant(self):
        sim=self.build('cola');sim.nextExperiment()
        ps=[Network.get(i).getProtocol(1) for i in range(3)]
        np.testing.assert_allclose(np.mean([p.v for p in ps],axis=0),
                                   np.sum([p.local_product for p in ps],axis=0),atol=1e-12)
        self.assertEqual(sum(p.messages_sent for p in ps),5*3*2)
        self.assertEqual(sum(p.comm_bytes for p in ps),5*3*2*4*8)
        self.assertTrue(all('accuracy' not in h for p in ps for h in p.metrics))

    def test_gadget_projection_pre_shrink_margin_and_no_aliasing(self):
        # margin before shrinking is above 1: only shrink, no hinge update.
        np.testing.assert_allclose(gadget_local_step(np.array([2.,0]),np.array([1.,0]),1,.2,2),[1,0])
        out=gadget_local_step(np.zeros(2),np.array([1.,2]),-1,.2,2)
        self.assertAlmostEqual(np.linalg.norm(out),1/np.sqrt(.2))
        sim=self.build('gadget',2);sim.nextExperiment()
        ps=[Network.get(i).getProtocol(1) for i in range(3)]
        self.assertEqual(sum(p.messages_sent for p in ps),2*3*2)
        self.assertEqual(sum(p.comm_bytes for p in ps),2*3*2*4*8)
        self.assertFalse(np.shares_memory(ps[0].w,ps[1].w))

    def test_choco_neighbor_hat_copies_and_sparse_wire_size(self):
        sim=self.build('choco');sim.nextExperiment()
        ps=[Network.get(i).getProtocol(1) for i in range(3)]
        for i,p in enumerate(ps):
            for j in range(3):np.testing.assert_allclose(p.hats[j],ps[j].hats[j])
            self.assertTrue(np.isfinite(p.w).all())
        self.assertEqual(sum(p.messages_sent for p in ps),5*3*2)
        # d=4, ceil(.2*d)=1; uint32 index plus float64 value per packet.
        self.assertEqual(sum(p.comm_bytes for p in ps),5*3*2*12)

    def test_choco_full_compression_matches_author_update_order(self):
        sim=self.build('choco',1)
        for init in sim.initializers:init.execute()
        sim.initializers=[]
        ps=[Network.get(i).getProtocol(1) for i in range(3)]
        expected=[]
        for i,p in enumerate(ps):
            p.fraction=1.;p.w=np.arange(4,dtype=float)*(.01+i*.02)
            for j in range(3):p.hats[j]=np.ones(4)*(.2+j*.1)
            rng=np.random.RandomState();rng.set_state(p.rng.get_state())
            ids=rng.randint(p.n,size=p.batch);Xb,yb=p.X[ids],p.y[ids]
            grad=p.lambda_reg*p.w-np.asarray(Xb.T@(yb*(yb*(Xb@p.w)<1))).ravel()/p.batch
            expected.append(p.w-p.lr*grad+p.gamma*sum(p.W[j]*(p.hats[j]-p.hats[i]) for j in range(3)))
        sim.nextExperiment()
        for p,e in zip(ps,expected):np.testing.assert_allclose(p.w,e,atol=1e-14)


if __name__=='__main__':unittest.main()
