import unittest,math
import numpy as np
import torch
from paper_reference import ssl_tabular as tab
from src.fedssl_tabular_simulation import TabularFedSSLSimulation


class TabularSSLTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(3)

    def fixture(self):
        rng=np.random.default_rng(8)
        u=[rng.normal(size=(16+i,4)) for i in range(3)]
        l=[(x,np.where(np.arange(len(x))%2,1,-1)) for x in u]
        # Deliberately unusable tests: training must never read these objects.
        t=[(object(),object()) for _ in u]
        cfg=tab.TabularSSLConfig(input_dim=4,hidden_dim=8,embedding_dim=3,
            rounds=2,local_steps=2,batch_size=4,negatives=2,monitor_samples=8,
            svm_C=1,svm_cv=0,seed=19)
        return u,l,t,cfg

    def test_views_preserve_shape_and_local_feature_support(self):
        u,l,t,cfg=self.fixture();x=tab.input_tensor(u[0]);rng=np.random.default_rng(1)
        a,p,n=tab.sample_views(x,cfg,rng)
        self.assertEqual(p.shape,a.shape)
        self.assertEqual(n.shape,(cfg.batch_size*cfg.negatives,4))
        for j in range(4):self.assertTrue(torch.isin(p[:,j],x[:,j]).all())

    def test_released_mean_negative_loss(self):
        class Zero(torch.nn.Module):
            def forward(self,x):return torch.zeros((len(x),3),dtype=torch.float64)
        x=torch.zeros((4,4),dtype=torch.float64)
        self.assertAlmostEqual(float(tab.loss_on_views(Zero(),(x,x,x.repeat(2,1)),2)),2*math.log(2),places=12)

    def test_training_changes_weights_and_is_deterministic_without_labels(self):
        u,l,t,cfg=self.fixture();x=tab.input_tensor(u[0]);state=tab.new_encoder(cfg).state_dict()
        a,loss=tab.train_local(state,x,cfg,0,0);b,_=tab.train_local(state,x,cfg,0,0)
        self.assertTrue(np.isfinite(loss))
        self.assertTrue(any(not torch.equal(a[k],state[k]) for k in state))
        self.assertTrue(all(torch.equal(a[k],b[k]) for k in state))
        self.assertEqual(tab.monitor_loss(tab.new_encoder(cfg),x,cfg,0),tab.monitor_loss(tab.new_encoder(cfg),x,cfg,0))

    def test_peersim_matches_reference_aggregation_with_no_test_access(self):
        for topology,activation in [('ring','ordered'),('ring','shuffle'),('full','shuffle')]:
            with self.subTest(topology=topology,activation=activation):
                u,l,t,cfg=self.fixture();model=tab.new_encoder(cfg)
                for epoch in range(cfg.rounds):
                    updates=[(*tab.train_local(model.state_dict(),tab.input_tensor(x),cfg,i,epoch),len(x)) for i,x in enumerate(u)]
                    records=[(s,n,loss) for s,loss,n in updates]
                    model.load_state_dict(tab.aggregate_states(records,cfg)[0])
                sim=TabularFedSSLSimulation(u,l,t,cfg,topology,activation);result=sim.run()
                self.assertFalse(result['converged'])
                self.assertEqual(result['stop_reason'],'max_training_rounds')
                for p in sim.protos:
                    self.assertEqual(p.test_evaluations,0)
                    self.assertTrue(all(torch.equal(p.model.state_dict()[k],v) for k,v in model.state_dict().items()))
                self.assertGreater(result['total_comm_bytes'],0)

    def test_operational_stop_and_final_test_exactly_once(self):
        u,l,t,cfg=self.fixture();cfg.rounds=10;cfg.stability_window=1;cfg.rtol=100
        t=[(x,y) for x,y in l]
        sim=TabularFedSSLSimulation(u,l,t,cfg);result=sim.run()
        self.assertTrue(result['converged']);self.assertEqual(result['training_rounds'],3)
        self.assertEqual([p.test_evaluations for p in sim.protos],[0,0,0])
        self.assertEqual(len(sim.evaluate_final()),3)
        with self.assertRaisesRegex(RuntimeError,'already'):sim.evaluate_final()

    def test_resume_preserves_encoder_monitor_and_byte_accounting(self):
        from dataclasses import replace
        u,l,t,cfg=self.fixture()
        first=TabularFedSSLSimulation(u,l,t,cfg,activation='ordered')
        prior=first.run()
        state={k:v.clone() for k,v in first.protos[0].model.state_dict().items()}
        resumed=TabularFedSSLSimulation(u,l,t,replace(cfg,rounds=4),activation='ordered',resume=(state,prior))
        result=resumed.run()
        full=TabularFedSSLSimulation(u,l,t,replace(cfg,rounds=4),activation='ordered')
        reference=full.run()
        self.assertEqual(result['history'],reference['history'])
        self.assertEqual(result['total_comm_bytes'],reference['total_comm_bytes'])
        self.assertEqual(result['link_cycles'],12)
        for k,v in full.protos[0].model.state_dict().items():
            self.assertTrue(torch.equal(v,resumed.protos[0].model.state_dict()[k]))

    def test_invalid_domains_fail(self):
        with self.assertRaises(ValueError):tab.input_tensor(np.ones((4,2,8)))
        with self.assertRaises(ValueError):tab.input_tensor([[float('nan')],[1.]])


if __name__=='__main__':unittest.main()
