# FedSSL-AMC a9a adaptation: tabular + one-hop gossip

**Current implementation:** `run_ssl_tabular.py`,
`src/ssl_tabular_gossip_simulation.py`, `p2p/fedssl_tabular_gossip.py`.
This is a domain and deployment adaptation, not a paper reproduction.

## What the original implementation does

The author release uses a central server to aggregate client encoders and send
updated weights back. It does not use flooding. The earlier local PeerSim port
used flooding to reproduce exact global aggregation without a server. That was
our deployment choice; its 100.38912 GB result is superseded for this comparison.
It must not be attributed to the paper or reused as a gossip result.

Author source: https://github.com/usmanakua927/FedSSL-AMC_Github
Pinned commit: 7cf7dc290aa208ef525c56b4591a501e547b8abf.
Corruption inspiration: SCARF, https://arxiv.org/abs/2106.15147.
Neither AMC-paper accuracy nor a SCARF reproduction is claimed.

## Current algorithm

- Double-precision 123 -> 128 ReLU -> 64 MLP replaces the causal I/Q CNN.
- Positive pairs: original row vs local empirical-marginal feature corruption
  (probability .3 per column). Corrupted one-hot groups need not remain valid;
  these are augmentation views, not claimed real examples.
- Ten negatives from other local training rows, excluding anchor row ID.
- Released positive logistic + MEAN-negative logistic objective; local Adam
  .001, batch 128, ten steps per link cycle. Optimizer resets each local round.
- Shared initial encoder seed; each local update has an isolated client/round
  sampling seed, with no labels or test samples accepted by pretraining.
- Use the SAME `GossipProtocol.nextCycle` and `_gossip_push` as SDCA, CoCoA,
  CoCoA+, BDSVM and FedAvg: drain inbox, merge, local update, push current encoder
  to at most three selected neighbors. No all-origin encoder tables or flooding.
- Encoder aggregation is the uniform local-neighborhood mean (author uniform
  weighting retained locally). FedAvg's own sample weighting is intrinsic to
  its algorithm. Asynchronous neighborhood means are not exact global uniform
  averaging and inherit no original server-convergence guarantee.
- Freeze each peer's OWN final encoder, fit its own LinearSVC with five-fold
  training-only C selection; all existing local training labels are used.
  This is not a low-label-data experiment. Head iteration cap 100000.

## Controlled communication conditions

- Same a9a loader, seed-42 ten training shards, original shared test file.
- Same random-k-out topology (k=3, undirected), activation mode and network seed.
- Event-hashed communication schedule keyed by network seed/cycle/node ID,
  separated from learning RNG consumption. Corresponding rounds use the same
  activation and neighbor selections despite different learners.
- One PeerSim cycle = one local update per node plus at most three sends.
- One shared `numeric-v1` wire-size function: real array/tensor dtype bytes;
  numeric scalar values and numeric origin IDs eight bytes; fixed schema string
  keys, Python container memory and physical network headers excluded uniformly.
- Record actual directed sends once. Same policy counts relayed SDCA/CoCoA/BDSVM
  contribution tables, as those tables are algorithm-specific payloads.
- Record topology, message count and first-100-round neighbor-schedule hash in
  every result; figure generation checks equality across P2P methods.
- No SDCA warm start in the controlled comparison; no hidden pretraining pass.
- Equal three-thread sequential timing budget. Local steps, parameter counts,
  solver work and model capacity remain algorithm-specific and must be disclosed.

Each SSL encoder has 24128 float64 parameters. A gossip payload carries the
current encoder plus an int64 local sample count: 193032 bytes. For ten nodes
sending to three neighbors, each cycle sends 30 payloads = 5790960 bytes.
Actual stopped round count determines the cumulative total; do not multiply by
an extra flooding factor or by ten link cycles per encoder round.

## Stopping and evaluation

G=50 adjacent-window means, relative tolerance .001, absolute tolerance 1e-8,
two consecutive stable comparisons for EVERY peer. The controlled SVM comparison
uses the same window/check policy on its train objectives. SSL monitors fixed
local training anchors/corruption masks/negatives. Objectives are different;
matching relative stationarity does not establish equal optimality or quality.

Offline monitoring CPU time is included. Hypothetical distributed monitor
traffic is excluded consistently from algorithm communication costs.
Test scores are evaluated once per final peer model only after operational
convergence. Capped runs report N/A test accuracy. Final train hinge evaluates
each peer's own encoder/head over the complete train set. Contrastive loss is
not hinge loss. All final peer encoders and corresponding local heads are saved.

The earlier flooding pilot had a separate capped diagnostic test observation.
It is historical, was not used to tune this gossip adaptation, and is excluded
from the controlled comparison. Old flooding checkpoints cannot be resumed as
new gossip experiments.

## Comparison limits

Nonlinear feature learning and train-CV local heads differ from raw-feature
linear SVMs. BDSVM also has its own budget/regularization. Single-host time ratios
are implementation timing, not an equal-quality distributed speedup proof.

Run: `python run_ssl_tabular.py --data-dir <processed> --out <fresh-output>`.
