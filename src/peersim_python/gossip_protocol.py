"""Generic gossip transport as a PeerSim CDProtocol.

The reusable communication substrate, extracted out of any specific learner so
that ANY algorithm can gossip. Per cycle a node:
  1. drains its inbox and folds the received payloads into its state via a
     pluggable Aggregator,
  2. runs one local update on its own state,
  3. pushes its outgoing payload to `gossip_k` random neighbours (via Linkable),
  4. records per-round metrics.

A concrete learner supplies these hooks:
  - current_state()   : the node's own base vector the aggregator folds into
  - set_state(v)      : adopt the aggregated result
  - outgoing_payload(): what to gossip — the absolute state (model averaging) OR
                        the per-round increment delta-w (CoCoA); defaults to
                        current_state()
  - local_update(), record(), payload_nbytes(), ready()
It never re-implements the inbox, the send, or the merge — those live here.
See `SDCAProtocol` for an example tenant.
"""

from src.peersim_python.cdsim.cd_protocol import CDProtocol
from src.peersim_python.core.common_state import CommonState
from src.peersim_python.aggregator import PlainAverageAggregator


class GossipProtocol(CDProtocol):
    """Reusable gossip loop (transport + pluggable aggregation), learner-agnostic."""

    LINKABLE_PID = 0  # protocol id of the IdleProtocol holding neighbours

    def __init__(self, gossip_k=1, aggregator=None):
        self.gossip_k = gossip_k
        self.aggregator = aggregator or PlainAverageAggregator()
        self.inbox: list = []   # received payloads — the async mailbox
        self.comm_bytes = 0

    # ---- the per-cycle gossip loop (fixed; shared by every tenant) ----------
    def nextCycle(self, node, pid):
        if not self.ready():
            return
        self._drain_and_merge()   # 1. receive + fold neighbours' payloads in
        self.local_update()       # 2. local training step (may set the increment)
        self._gossip_push(node, pid)  # 3. send outgoing payload to neighbour(s)
        self.record()             # 4. log per-round metrics

    def _drain_and_merge(self):
        if not self.inbox:
            return
        merged = self.aggregator.aggregate(self.current_state(), self.inbox)
        self.set_state(merged)
        self.inbox = []

    def _gossip_push(self, node, pid):
        link = node.getProtocol(self.LINKABLE_PID)
        deg = link.degree()
        if deg == 0:
            return
        payload = self.outgoing_payload()
        # Draw neighbours WITHOUT replacement. Drawing with replacement let one
        # peer be picked twice in a cycle, so it received the same payload twice
        # — harmless for an idempotent merge, but it double-charged the
        # communication counter and, for an additive merge, applied the same
        # update twice.
        for peer_index in CommonState.r.sample(range(deg), min(self.gossip_k, deg)):
            peer = link.getNeighbor(peer_index)
            peer.getProtocol(pid).inbox.append(dict(payload))
            self.comm_bytes += self.payload_nbytes()

    # ---- hooks a concrete learner MUST implement ----------------------------
    def ready(self):
        """True once the node is set up (data assigned, etc.). Default: always."""
        return True

    def current_state(self):
        """The node's own base vector (e.g. the weight w) the aggregator folds into."""
        raise NotImplementedError

    def set_state(self, merged):
        """Adopt the aggregated result."""
        raise NotImplementedError

    def outgoing_payload(self):
        """What to gossip. Default: the current state (model averaging).

        Increment-based schemes override this to return the per-round change
        (delta-w) instead of the absolute state.
        """
        return self.current_state()

    def local_update(self):
        """Run one local training step on this node's own data."""
        raise NotImplementedError

    def record(self):
        """Append this cycle's metrics. No-op by default.

        A learner whose metrics only mean something network-wide (a global
        duality gap, say) leaves this alone and lets a Control measure it.
        """
        return None

    def payload_nbytes(self):
        """Bytes counted per gossip send (for comm-cost accounting)."""
        raise NotImplementedError
