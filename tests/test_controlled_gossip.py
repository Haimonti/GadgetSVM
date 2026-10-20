import unittest
import numpy as np
import torch
from src.peersim_python.core import Network,GeneralNode,CommonState
from src.peersim_python.idle_protocol import IdleProtocol
from src.peersim_python.dynamics import WireKOut
from src.peersim_python.cdsim import CDSimulator
from src.peersim_python.gossip_protocol import GossipProtocol
from src.peersim_python.communication import numeric_payload_nbytes,activation_order
from paper_reference.ssl_tabular import TabularSSLConfig
from p2p.fedssl_tabular_gossip import TabularSSLGossipProtocol
from src.ssl_tabular_gossip_simulation import GossipTabularSSLSimulation


class Keep:
    def aggregate(self,current,peers):return current


class Dummy(GossipProtocol):
    def __init__(self,noise=0):super().__init__(3,Keep());self.noise=noise
    def clone(self):return Dummy(self.noise)
    def current_state(self):return {'w':np.zeros(2,dtype=np.float64),'n':8}
    def set_state(self,state):pass
    def local_update(self):
        for _ in range(self.noise):CommonState.r.random()


class ControlledGossipTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(3)

    def dummy(self,noise):
        CommonState.initializeRandom(42);Network.reset(10,GeneralNode([IdleProtocol(),Dummy(noise)]))
        CDSimulator(20,[WireKOut(0,3,undir=True)],activation='shuffle').nextExperiment()
        return [Network.get(i).getProtocol(1) for i in range(10)]

    def test_neighbor_schedule_is_independent_of_learning_rng_draws(self):
        a=self.dummy(0);b=self.dummy(103)
        self.assertEqual([p.communication_trace for p in a],[p.communication_trace for p in b])
        self.assertEqual(sum(p.messages_sent for p in a),20*10*3)
        self.assertEqual(sum(p.comm_bytes for p in a),20*10*3*24)
        self.assertTrue(all(len(set(dest))==3 for p in a for cycle,dest,size in p.communication_trace))

    def test_activation_schedule_does_not_consume_learning_rng(self):
        CommonState.initializeRandom(42)
        expected=activation_order(10,42,7,'shuffle')
        for _ in range(1000):CommonState.r.random()
        self.assertEqual(expected,activation_order(10,42,7,'shuffle'))
        self.assertEqual(sorted(expected),list(range(10)))

    def test_numeric_wire_policy_includes_origin_id_and_metadata(self):
        payload={7:(3,np.ones(4,dtype=np.float64),.5)}
        self.assertEqual(numeric_payload_nbytes(payload),8+8+32+8)
        self.assertEqual(numeric_payload_nbytes({'w':torch.zeros(3,dtype=torch.float64),'n':7}),32)

    def fixture(self):
        rng=np.random.default_rng(5)
        u=[rng.normal(size=(16,4)) for _ in range(3)]
        l=[(x,np.tile([-1,1],8)) for x in u]
        cfg=TabularSSLConfig(input_dim=4,hidden_dim=8,embedding_dim=3,rounds=3,
            local_steps=2,batch_size=4,negatives=2,monitor_samples=8,svm_C=1,svm_cv=0)
        return u,l,cfg

    def test_ssl_uses_exact_same_transport_and_single_cycle_round(self):
        self.assertIs(TabularSSLGossipProtocol.nextCycle,GossipProtocol.nextCycle)
        self.assertIs(TabularSSLGossipProtocol._gossip_push,GossipProtocol._gossip_push)
        u,l,cfg=self.fixture()
        sim=GossipTabularSSLSimulation(u,l,[(object(),object())]*3,cfg,topology='full',gossip_k=2)
        r=sim.run()
        self.assertEqual(r['link_cycles'],3);self.assertEqual(r['training_rounds'],3)
        self.assertEqual(r['messages_sent'],3*3*2)
        params=sum(p.numel() for p in sim.protos[0].model.parameters())
        self.assertEqual(r['total_comm_bytes'],3*3*2*(params*8+8))
        self.assertEqual([p.test_evaluations for p in sim.protos],[0]*3)
        self.assertEqual(set(sim.protos[0].outgoing_payload()),{'state','n'})

    def test_ssl_final_test_once_after_all_peer_stop(self):
        u,l,cfg=self.fixture();cfg.rounds=10;cfg.stability_window=1;cfg.rtol=100
        sim=GossipTabularSSLSimulation(u,l,l,cfg,topology='full',gossip_k=2)
        r=sim.run();self.assertTrue(r['converged']);self.assertEqual(r['rounds'],3)
        self.assertEqual(len(sim.evaluate_final()),3)
        with self.assertRaisesRegex(RuntimeError,'already'):sim.evaluate_final()


if __name__=='__main__':unittest.main()
