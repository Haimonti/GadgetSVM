"""Exact synchronous all-gather over links; not one-hop consensus averaging.

Each epoch starts with one immutable contribution per origin. Delta flooding
finishes before aggregation. Payload cost counts every directed link copy of
every contribution; optional headers default to 16 bytes per record.
"""
import numpy as np


class ExactAggregation:
    def __init__(self, adjacency=None, header_bytes=16):
        self.adjacency = adjacency
        self.header_bytes = header_bytes
        self.bytes = 0
        self.link_rounds = 0
        self.epochs = 0

    def gather(self, records, payload_sizes):
        n = len(records)
        if n < 1 or len(payload_sizes) != n:
            raise ValueError('One record and payload size per client required')
        self.epochs += 1
        if self.adjacency is None:
            self.bytes += sum(payload_sizes)  # uploads only
            return [list(records) for _ in records]
        if len(self.adjacency) != n:
            raise ValueError('Graph size does not match clients')
        graph = [set(v) for v in self.adjacency]
        if any(j < 0 or j >= n or j == i for i, v in enumerate(graph) for j in v):
            raise ValueError('Invalid graph edge')
        known = [{i} for i in range(n)]
        sent = {(i, j): set() for i in range(n) for j in graph[i]}
        while any(len(v) != n for v in known):
            incoming = [set() for _ in records]
            for (i, j), delivered in sent.items():
                delta = known[i] - delivered
                self.bytes += sum(payload_sizes[k] + self.header_bytes for k in delta)
                delivered.update(delta)
                incoming[j].update(delta)
            updated = [a | b for a, b in zip(known, incoming)]
            self.link_rounds += 1
            if updated == known:
                raise ValueError('Graph cannot disseminate all contributions')
            known = updated
        # Canonical origin order also makes floating-point summation reproducible.
        return [[records[k] for k in range(n)] for _ in records]

    def broadcast(self, payload_size, clients):
        if self.adjacency is None:
            self.bytes += clients * payload_size

    def weighted(self, values, weights):
        values = [np.asarray(v) for v in values]
        weights = np.asarray(weights, dtype=float)
        if (weights.shape != (len(values),) or not np.isfinite(weights).all()
                or (weights <= 0).any() or not np.isclose(weights.sum(), 1)):
            raise ValueError('Positive weights summing to one required')
        # Include each weight in its transmitted contribution.
        copies = self.gather(list(zip(values, weights)), [v.nbytes + 8 for v in values])
        return [sum(w * v for v, w in peer) for peer in copies]
