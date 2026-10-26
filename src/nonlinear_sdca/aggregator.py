"""Support-vector accumulation rule for the non-linear gossip tenant.

The kernel analogue of `VersionedContributionAggregator`. Where linear SDCA
gossips one ``d``-vector per origin and *sums* them to rebuild ``w``, the kernel
model has no ``w``: each origin instead contributes a **block of projected support
vectors with their dual coefficients**, and a node scores points against the
*union* of all blocks it has heard of (accumulate, not average).

State is ``origin -> (version, X_hat_sv, alpha_sv, y_sv)``:
  - ``version``  : monotonically increasing per origin (freshness tag)
  - ``X_hat_sv`` : that origin's projected support vectors, shape (k, m)
  - ``alpha_sv`` : their signed dual coefficients, shape (k,)
  - ``y_sv``     : their labels, shape (k,)

Gossip delivers the same block along several paths and out of order, so the merge
depends only on *which* versions arrived, never on how many times or in what
sequence: keep the freshest block per origin. This makes duplicate and
out-of-order delivery idempotent, exactly as the linear aggregator does.
"""

from src.peersim_python.aggregator import Aggregator


class SupportVectorAccumulator(Aggregator):
    """Keep the freshest projected-support-vector block per origin."""

    def aggregate(self, current, peers):
        merged = dict(current)
        for peer_table in peers:
            for origin, entry in peer_table.items():
                previous = merged.get(origin)
                if previous is None or entry[0] > previous[0]:
                    merged[origin] = entry
        return merged
