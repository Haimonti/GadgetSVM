import os
import unittest
from unittest.mock import patch
import numpy as np
import scipy.sparse as sp
from p2p.linear_baselines import signed_coordinate_solve
from p2p.fedavg_gossip import FedAvgGossipProtocol
from src.peersim_python.core import CommonState
from src.peersim_python.sdca_protocol import SDCAProtocol
from p2p.cocoa_gossip import CoCoAGossipProtocol,CoCoAPlusGossipProtocol


class CompiledLoopTests(unittest.TestCase):
    def test_zero_feature_duals_include_the_constant_hinge_loss(self):
        from p2p._linear_fast import dual_steps
        X=sp.csr_matrix((2,3));y=np.array([1.,-1.])
        a,w=dual_steps(X.indptr,X.indices,X.data,y,np.zeros(2),np.zeros(3),
            np.array([0,1],dtype=np.int64),1.,1.,1.)
        np.testing.assert_array_equal(a,y);np.testing.assert_array_equal(w,np.zeros(3))
        self.assertEqual(float((a*y).mean()),1.)

    def test_zero_feature_pegasos_sample_still_shrinks_weights(self):
        X=sp.csr_matrix((1,2));y=np.array([1.])
        for backend in ('0','1'):
            CommonState.initializeRandom(42)
            p=FedAvgGossipProtocol(local_steps=1)
            p.set_data(X,y,X,y,.2);p.configure_network(0,1,1,1)
            p.w[:]=[2.,-1.];p.steps_done=3
            with patch.dict(os.environ,{'SVM_COMPILED_LOCAL':backend}):p.local_update()
            np.testing.assert_allclose(p.w,[1.5,-.75],atol=1e-14)

    def test_zero_feature_cola_coordinates_reach_the_correct_box_endpoint(self):
        X=sp.csr_matrix((3,2));y=np.array([1.,-1.,1.]);b=np.array([0.,0.,-2.])
        for backend in ('0','1'):
            with patch.dict(os.environ,{'SVM_COMPILED_LOCAL':backend}):
                a=signed_coordinate_solve(X,y,np.zeros(3),b,1.,200,np.random.RandomState(42))
            np.testing.assert_array_equal(a,[1.,-1.,0.])

    def test_compiled_cola_agrees_with_author_sampler_and_python_objective(self):
        r=np.random.RandomState(9);X=sp.csr_matrix(r.normal(size=(13,8)));y=np.where(r.rand(13)<.5,-1.,1.)
        a=y*r.rand(13);b=r.normal(size=13)
        with patch.dict(os.environ,{'SVM_COMPILED_LOCAL':'0'}):
            expected=signed_coordinate_solve(X,y,a,b,.7,200,np.random.RandomState(42))
        with patch.dict(os.environ,{'SVM_COMPILED_LOCAL':'1'}):
            got=signed_coordinate_solve(X,y,a,b,.7,200,np.random.RandomState(42))
        np.testing.assert_allclose(got,expected,atol=1e-12,rtol=1e-12)

    def test_compiled_pegasos_reuses_the_same_random_draws_across_rounds(self):
        r=np.random.default_rng(8);X=sp.csr_matrix(r.normal(size=(17,7)));y=np.tile([-1.,1.],9)[:17]
        ps=[]
        for backend in ('0','1'):
            CommonState.initializeRandom(42)
            p=FedAvgGossipProtocol(local_steps=211)
            p.set_data(X,y,X,y,.01);p.configure_network(0,17,1,211)
            with patch.dict(os.environ,{'SVM_COMPILED_LOCAL':backend}):
                for _ in range(3):p.local_update()
            ps.append(p)
        np.testing.assert_allclose(ps[0].w,ps[1].w,atol=1e-12,rtol=1e-12)
        self.assertEqual(ps[0].steps_done,ps[1].steps_done)

    def test_compiled_sdca_cocoa_and_cocoa_plus_match_python(self):
        r=np.random.default_rng(8);X=sp.csr_matrix(r.normal(size=(17,7)));y=np.tile([-1.,1.],9)[:17]
        for cls in [SDCAProtocol,CoCoAGossipProtocol,CoCoAPlusGossipProtocol]:
            results=[]
            for backend in ('0','1'):
                CommonState.initializeRandom(42);p=cls()
                p.set_data(X,y,X,y,.01);p.configure_network(0,51,3,137,1.)
                with patch.dict(os.environ,{'SVM_COMPILED_LOCAL':backend}):
                    for _ in range(3):p.local_update()
                results.append(p)
            np.testing.assert_allclose(results[0].alpha,results[1].alpha,atol=1e-11,rtol=1e-11)
            np.testing.assert_allclose(results[0].w,results[1].w,atol=1e-11,rtol=1e-11)


if __name__=='__main__':unittest.main()
