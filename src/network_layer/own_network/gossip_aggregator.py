import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import numpy as np
from p2pfl.learning.aggregators.aggregator import Aggregator, NoModelsToAggregateError
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel
from p2pfl.management.logger import logger


class GossipAggregator(Aggregator):
    """
    Age-weighted pairwise merge — Algorithm 2 from:
    'Gossip Learning as a Decentralized Alternative to Federated Learning'

    Key difference from FedAvg:
    - FedAvg:          w = Σ(n_i · w_i) / Σ n_i  using fixed local data size
    - GossipAggregator: w = Σ(t_i · w_i) / Σ t_i  using accumulated model age t

    Model age t starts at local sample count and grows with each merge, so
    models that have participated in more rounds carry more weight. This
    eliminates the pseudo-central aggregator bottleneck — any node can
    merge with any peer directly, no fixed coordinator needed.
    """

    SUPPORTS_PARTIAL_AGGREGATION: bool = True

    def aggregate(self, models: list[P2PFLModel]) -> P2PFLModel:
        if not models:
            raise NoModelsToAggregateError("GossipAggregator: no models to merge")

        # t_i = accumulated model age (num_samples grows after each merge)
        ages       = [m.get_num_samples() for m in models]
        total_age  = sum(ages)

        # w_merged = Σ(t_i · w_i) / Σ t_i  — Algorithm 2 merge
        first_params = models[0].get_parameters()
        accum = [np.zeros_like(p) for p in first_params]

        for m, age in zip(models, ages):
            for i, param in enumerate(m.get_parameters()):
                accum[i] = np.add(accum[i], param * age)

        merged = [np.divide(p, total_age) for p in accum]

        contributors: list[str] = []
        for m in models:
            contributors += m.get_contributors()

        logger.info(
            "gossip",
            f"Merged {len(models)} models — ages={ages} total_age={total_age}",
        )

        # total_age carried forward so next merge weights this model by its history
        return models[0].build_copy(
            params=merged,
            num_samples=total_age,
            contributors=contributors,
        )
