"""FedSSL-AMC I/Q reference and explicitly named neural tabular adaptation.

Pinned author release and PeerSim port: see paper_reference/SSL_PEERSIM.md.
"""

def run_iq_clients(unlabeled_clients, labeled_clients, test_clients,
                   config=None, adjacency=None):
    from paper_reference.ssl import (SSLConfig, train_encoder, fit_local_heads,
                                     evaluate_local_heads)
    config = config or SSLConfig()
    if not (len(unlabeled_clients) == len(labeled_clients) == len(test_clients)):
        raise ValueError("One unlabeled, labeled, and test set per client required")
    model, metadata = train_encoder(unlabeled_clients, config, adjacency)
    heads = fit_local_heads(model, labeled_clients, config)
    accuracy = evaluate_local_heads(model, heads, test_clients)
    return dict(encoder=model, local_classifiers=heads, per_client_accuracy=accuracy,
                metadata=metadata)


def run(*args, **kwargs):
    raise ValueError("FedSSL-AMC requires I/Q time-series and separate local labeled sets. "
                     "Use run_iq_clients, or run_tabular_clients / run_ssl_tabular.py for the explicitly labelled tabular adaptation.")


def run_tabular_clients(unlabeled_clients,labeled_clients,test_clients,config=None,
                       topology='random_kout',activation='shuffle'):
    """Run the a9a-style adaptation on PeerSim; never silently replace AMC."""
    from paper_reference.ssl_tabular import TabularSSLConfig
    from src.ssl_tabular_gossip_simulation import GossipTabularSSLSimulation
    config=config or TabularSSLConfig(input_dim=unlabeled_clients[0].shape[1])
    simulation=GossipTabularSSLSimulation(unlabeled_clients,labeled_clients,test_clients,
        config,topology,activation)
    result=simulation.run()
    result['per_node_accuracy']=simulation.evaluate_final() if result['converged'] else []
    result['method']='fedssl_tabular'
    result['reproduction_claim']=False
    result['simulation']=simulation
    return result
