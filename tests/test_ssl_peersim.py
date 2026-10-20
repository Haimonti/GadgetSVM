import unittest
import warnings
import json
import hashlib
import math
from pathlib import Path
import numpy as np
import torch
from paper_reference.ssl import (SSLConfig,new_encoder,train_encoder,fit_local_heads,
    architecture_summary,TripletLoss,aggregate_states)
from src.fedssl_simulation import FedSSLSimulation


class SSLPeerSimTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(3)
        warnings.filterwarnings('ignore',message='.*weight_norm.*deprecated.*')

    def fixture(self,n=3):
        rng=np.random.default_rng(21)
        u=[rng.normal(size=(8+i,2,16)) for i in range(n)]
        l=[(rng.normal(size=(12,2,16)),np.tile([0,1],6)) for _ in range(n)]
        t=[(rng.normal(size=(8,2,16)),np.tile([0,1],4)) for _ in range(n)]
        return u,l,t

    def config(self,**kwargs):
        return SSLConfig(rounds=2,local_steps=2,batch_size=3,channels=4,depth=1,
            reduced_size=6,embedding_dim=4,negatives=2,svm_C=1.,svm_cv=0,**kwargs)

    def test_pinned_author_sources_and_defaults(self):
        root=Path(__file__).resolve().parents[1]/'paper_reference/author_ssl'
        source=json.loads((root/'SOURCE.json').read_text())
        for filename in ['networks/causal_cnn.py','losses/triplet_loss.py','default_hyperparameters.json']:
            self.assertEqual(hashlib.sha256((root/filename).read_bytes()).hexdigest(),source['hashes'][filename])
        cfg=SSLConfig();released=json.loads((root/'default_hyperparameters.json').read_text())
        self.assertEqual((cfg.depth,cfg.channels,cfg.local_steps,cfg.batch_size),
                         tuple(released[k] for k in ['depth','channels','nb_steps','batch_size']))
        summary=architecture_summary(new_encoder(cfg))
        self.assertEqual(summary['main_convolutions'],16)
        self.assertEqual(summary['projection_convolutions'],2)

    def test_author_negative_normalization_is_not_paper_equation_sum(self):
        class Zero(torch.nn.Module):
            def forward(self,x):return torch.zeros((len(x),3),dtype=torch.float64)
        x=torch.zeros((4,2,8),dtype=torch.float64)
        value=TripletLoss(None,10,1,Sim_CSE=False)(x,Zero(),x)
        self.assertAlmostEqual(float(value),2*math.log(2),places=12)

    def test_released_uniform_aggregation_and_optional_sample_weights(self):
        records=[({'w':torch.tensor([1.],dtype=torch.float64)},1,2.),
                 ({'w':torch.tensor([9.],dtype=torch.float64)},3,6.)]
        state,loss=aggregate_states(records,self.config())
        self.assertEqual(float(state['w'][0]),5.)
        self.assertEqual(loss,4.)
        state,loss=aggregate_states(records,self.config(aggregation='sample_weighted'))
        self.assertEqual(float(state['w'][0]),7.)
        self.assertEqual(loss,5.)

    def test_actual_peersim_matches_server_across_topology_activation_and_weights(self):
        u,l,t=self.fixture()
        for topology,activation,aggregation in [('ring','ordered','uniform'),
                ('ring','shuffle','sample_weighted'),('full','shuffle','uniform')]:
            with self.subTest(topology=topology,activation=activation,aggregation=aggregation):
                cfg=self.config(aggregation=aggregation)
                server,metadata=train_encoder(u,cfg)
                expected_heads=fit_local_heads(server,l,cfg)
                sim=FedSSLSimulation(u,l,t,cfg,topology,activation)
                result=sim.run()
                self.assertEqual(result['link_cycles'],len(u)*cfg.rounds)
                for p,h in zip(sim.protos,expected_heads):
                    self.assertEqual(p.test_evaluations,0)
                    for key,value in server.state_dict().items():self.assertTrue(torch.equal(value,p.model.state_dict()[key]),key)
                    np.testing.assert_allclose(p.head.coef_,h.coef_,atol=1e-12)
                    self.assertEqual([r['train_triplet_loss'] for r in p.metrics],
                                     [r['train_triplet_loss'] for r in metadata['history']])
                self.assertEqual(len(sim.evaluate_final()),len(u))
                with self.assertRaisesRegex(RuntimeError,'already'):sim.evaluate_final()

    def test_two_node_wire_accounting(self):
        cfg=self.config();u,l,t=self.fixture(2)
        sim=FedSSLSimulation(u,l,t,cfg,'full','ordered')
        result=sim.run()
        size=sum(v.numel()*v.element_size() for v in sim.protos[0].model.state_dict().values())
        # First node sends its record; second relays two. Header is 32/record.
        self.assertEqual(result['total_comm_bytes'],3*(size+32)*cfg.rounds)

    def test_disconnected_graph_fails_without_returning_partial_encoder(self):
        cfg=self.config();u,l,t=self.fixture()
        sim=FedSSLSimulation(u,l,t,cfg).build()
        sim.sim.initializers=sim.sim.initializers[1:]  # no edges
        with self.assertRaisesRegex(RuntimeError,'incomplete'):sim.run()

    def test_training_only_cv_retains_local_linear_heads(self):
        cfg=self.config();cfg.svm_cv=5
        u,l,t=self.fixture(2)
        model,_=train_encoder(u,cfg)
        heads=fit_local_heads(model,l,cfg)
        from sklearn.svm import LinearSVC
        self.assertTrue(all(isinstance(h,LinearSVC) for h in heads))
        self.assertNotEqual(id(heads[0]),id(heads[1]))


if __name__=='__main__':unittest.main()
