import unittest


class PaperEntrypointTests(unittest.TestCase):
    def test_old_fdr_protocol_is_not_silently_selected(self):
        from p2p.fdr_svm_gossip import FDRSVMGossipProtocol
        with self.assertRaisesRegex(RuntimeError, 'retired'):
            FDRSVMGossipProtocol()

    def test_ssl_protocol_rejects_tabular_data(self):
        from p2p.fedssl_gossip import FedSSLGossipProtocol
        import numpy as np
        with self.assertRaisesRegex(ValueError, 'I/Q'):
            FedSSLGossipProtocol().set_iq_data(np.ones((5,123)),None,None,0,1)

    def test_tabular_ssl_is_deferred(self):
        from methods.fedssl_amc import run
        with self.assertRaisesRegex(ValueError, 'I/Q'):
            run()

    def test_obsolete_fdr_assembly_has_migration_message(self):
        from src.fdr_svm_simulation import FDRSVMSimulation
        with self.assertRaisesRegex(RuntimeError, 'retired'):
            FDRSVMSimulation({}, []).build()


if __name__ == '__main__':
    unittest.main()
