"""Pluggable aggregation rules for the gossip layer.

An Aggregator decides how a node folds what it received (its inbox) into its own
state. It is deliberately algorithm-agnostic: it operates on plain numpy vectors
and knows nothing about SDCA, so any gossip learner can reuse it. Swap the rule
without touching the protocol.

Two rules are provided:
  - PlainAverageAggregator : average absolute weights (model averaging). Simple,
    but for SDCA it breaks the primal-dual invariant and diverges on long runs.
  - IncrementAggregator     : CoCoA-style — ADD the received per-round increments
    (delta-w) onto the current weight instead of averaging absolutes. Because the
    increments shrink to zero as training converges, the weight stops moving at
    convergence, so the invariant is preserved and it does not run away.
"""

import numpy as np


class Aggregator:
    """Interface: fold received payloads into the node's current state.

    aggregate(current, peers) -> new_state, where `current` is the node's own
    base vector and `peers` is the list of payloads received this round.
    """
    def aggregate(self, current, peers):
        raise NotImplementedError


class PlainAverageAggregator(Aggregator):
    """Equal-weight mean of the node's own weight and every received weight."""
    def aggregate(self, current, peers):
        accum = current.astype(np.float64)
        count = 1
        for p in peers:
            accum += p.astype(np.float64)
            count += 1
        return (accum / count).astype(current.dtype)


class IncrementAggregator(Aggregator):
    """CoCoA-style: add the received increments (delta-w) onto the current weight.

    `peers` are per-round weight *increments*, not absolute weights. The node's
    own increment is already in `current` (applied during its local step), so we
    only add the neighbours' contributions, scaled by `gamma`. Default gamma =
    1/(#received + 1) averages the neighbour increments (conservative, stable);
    pass an explicit gamma (e.g. 1/K for K workers, the CoCoA default) to override.
    """
    def __init__(self, gamma=None):
        self.gamma = gamma

    def aggregate(self, current, peers):
        if not peers:
            return current
        g = self.gamma if self.gamma is not None else 1.0 / (len(peers) + 1)
        accum = current.astype(np.float64)
        for p in peers:
            accum += g * p.astype(np.float64)
        return accum.astype(current.dtype)
