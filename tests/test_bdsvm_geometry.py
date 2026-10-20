import unittest
from unittest.mock import patch
import numpy as np
import scipy.sparse as sp
from src.bdsvm_simulation import BDSVMSimulation,params_for
from src.peersim_python.core import Network


class BDSVMGeometryTests(unittest.TestCase):
    def fixture(self):
        shards=[]
        for scale in [.1,1.,10.]:
            x=sp.csr_matrix(scale*np.array([[1.,2.],[2.,1.],[1.,1.],[-1.,-1.]]))
            y=np.array([1.,-1.,1.,-1.])
            shards.append(dict(X_csr=x,y=y,X_test=x,y_test=y,n_local=len(y)))
        cfg=dict(DATASET='a9a',NUM_WORKERS=3,SEED=2,LAMBDA=.01,GOSSIP_K=2,
            TOPOLOGY='full',EVAL_EVERY=1,STOP_ON_THRESHOLD=False,ACTIVATION='ordered',ROUNDS=8)
        return cfg,shards

    def test_current_comparison_is_linear_without_rbf_or_gamma_estimation(self):
        cfg,data=self.fixture()
        self.assertEqual(params_for('a9a')['kernel'],'linear')
        with patch('p2p.bdsvm_gossip._rbf',side_effect=AssertionError('RBF used')), \
             patch('p2p.bdsvm_gossip._median_gamma',side_effect=AssertionError('gamma inferred')):
            sim=BDSVMSimulation(cfg,data)
            sim.run(3)
            ps=[Network.get(i).getProtocol(1) for i in range(3)]
            self.assertTrue(all(p.kernel=='linear' and p.gamma is None for p in ps))
            self.assertTrue(all(p.P<=data[0]['X_csr'].shape[1] for p in ps))

    def test_rbf_uses_one_common_kernel_after_p2p_calibration(self):
        cfg,data=self.fixture()
        params={**params_for('a9a'),'kernel':'rbf','P':3}
        sim=BDSVMSimulation(cfg,data,params=params).build()
        sim.sim.controls=[]  # calibration/optimizer check without test scoring
        sim.run(8)
        ps=[Network.get(i).getProtocol(1) for i in range(3)]
        self.assertTrue(all(p.kernel_ready for p in ps))
        self.assertEqual(len(set(p.gamma for p in ps)),1)
        candidates=[ps[0].geometry[i][0] for i in range(3)]
        self.assertGreater(max(candidates)/min(candidates),100)
        self.assertEqual(ps[0].gamma,float(np.median(candidates)))
        for p in ps:
            np.testing.assert_array_equal(p.Kpp,ps[0].Kpp)
            self.assertEqual(len(p.contributions),3)
            self.assertGreater(p.comm_bytes,0)

    def test_server_benchmark_default_is_also_linear(self):
        from methods.bdsvm import run
        cfg,data=self.fixture()
        x=sp.vstack([d['X_csr'] for d in data]);y=np.concatenate([d['y'] for d in data])
        indices=np.array_split(np.arange(len(y)),3)
        with patch('methods.bdsvm._rbf',side_effect=AssertionError('RBF used')), \
             patch('methods.bdsvm._median_gamma',side_effect=AssertionError('gamma inferred')):
            result=run(x,y,x,y,indices,n_rounds=3,P=3,C=1.)
        self.assertEqual(result['kernel'],'linear')
        self.assertIsNone(result['gamma'])

    def test_empty_client_does_not_block_rbf_calibration_or_origin_completeness(self):
        cfg,data=self.fixture()
        data.append(dict(X_csr=sp.csr_matrix((0,2)),y=np.array([]),
            X_test=data[0]['X_test'],y_test=data[0]['y_test'],n_local=0))
        cfg.update(NUM_WORKERS=4,GOSSIP_K=3)
        sim=BDSVMSimulation(cfg,data,params={**params_for('a9a'),'kernel':'rbf','P':3}).build()
        sim.sim.controls=[]
        sim.run(8)
        ps=[Network.get(i).getProtocol(1) for i in range(4)]
        self.assertEqual(len(set(p.gamma for p in ps)),1)
        self.assertTrue(all(len(p.contributions)==4 for p in ps))
        self.assertTrue(all(np.isfinite(p.beta).all() for p in ps))


if __name__=='__main__':unittest.main()
