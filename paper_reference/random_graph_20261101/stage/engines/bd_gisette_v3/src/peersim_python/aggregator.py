"""Pluggable aggregation rules for the gossip layer.

An Aggregator decides how a node folds what it received (its inbox) into its own
state. It is deliberately algorithm-agnostic: it operates on plain payloads and
knows nothing about SDCA, so any gossip learner can reuse it.

Three rules are provided:
  - PlainAverageAggregator : average absolute weights (model averaging).
  - IncrementAggregator    : add received per-round increments onto the state.
  - VersionedContributionAggregator : keep the freshest entry per origin.

Only the last one is sound for SDCA, and the reason is worth stating because it
is the bug that made the earlier runs degrade. SDCA's coordinate step is only
valid while the primal and the dual describe the same model:

    w = X.T @ alpha / (lambda * n)

The first two rules fold a neighbour's work into `w` alone; each node's dual
block `alpha` is untouched. From the first exchange onward `w` no longer matches
`alpha`, and every later step computes a margin from a `w` that its own `alpha`
does not explain. Adding increments does not repair this: an increment applied
twice (the same peer drawn twice in one cycle) or never (a peer that is not
drawn) shifts `w` by an amount no dual variable accounts for, and because each
node draws a different random subset of neighbours, no two nodes accumulate the
same `w` — so they cannot reach consensus even in principle.

Keeping identified, versioned per-origin contributions fixes both halves: `w` is
*reconstructed* from the contributions a node knows rather than accumulated, so
duplicate and out-of-order delivery are idempotent, and each contribution stays
tied to the dual block that produced it.
"""

import numpy as np


class Aggregator:
    """Interface: fold received payloads into the node's current state.

    aggregate(current, peers) -> new_state, where `current` is the node's own
    base state and `peers` is the list of payloads received this round.
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
    """Add the received increments (delta-w) onto the current weight.

    Kept for gossip learners whose state genuinely is an accumulator. It is NOT
    suitable for SDCA — see the module docstring. Note also that the default
    gamma of 1/(#received + 1) makes a node's effective step depend on how many
    messages happened to arrive that cycle, which adds variance rather than
    removing it.
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


class VersionedContributionAggregator(Aggregator):
    """Keep the freshest contribution per origin; ignore stale or repeated ones.

    A state is ``origin -> (version, primal_contribution, alpha_sum)``. Gossip
    delivers the same entry along several paths and out of order, so the merge
    must depend only on *which* entries arrived — never on how many times or in
    what sequence. Comparing versions gives exactly that.
    """

    def aggregate(self, current, peers):
        merged = dict(current)
        for peer_table in peers:
            for origin, entry in peer_table.items():
                previous = merged.get(origin)
                if previous is None or entry[0] > previous[0]:
                    version, contribution, alpha_sum = entry
                    merged[origin] = (int(version), contribution, float(alpha_sum))
        return merged
