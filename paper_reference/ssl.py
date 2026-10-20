"""FedSSL-AMC, pinned author release 7cf7dc2 (paper v1 repository).

Reuse the author's encoder and TripletLoss. Released defaults differ from
paper prose: depth=7/channels=80, mean negative loss, LinearSVC with train CV.
These are explicit source choices, not a claimed reproduction of paper v2.
"""
from dataclasses import dataclass, asdict
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.exceptions import ConvergenceWarning
from .author_ssl.networks.causal_cnn import CausalCNNEncoder
from .author_ssl.losses.triplet_loss import TripletLoss
from .transport import ExactAggregation


@dataclass
class SSLConfig:
    rounds: int = 10
    local_steps: int = 2500
    batch_size: int = 50
    learning_rate: float = .001
    negatives: int = 10
    embedding_dim: int = 320
    kernel_size: int = 3
    # Actual released default_hyperparameters.json, not inferred from prose.
    channels: int = 80
    depth: int = 7
    reduced_size: int = 160
    svm_C: float | None = None  # None -> author's C grid, train-only CV
    svm_cv: int = 5
    aggregation: str = 'uniform'  # author's weights_agg; equal-sized paper clients
    compared_length: int | None = 192  # author's server CLI default
    seed: int = 0

    def validate(self):
        for name in ('rounds', 'local_steps', 'batch_size', 'negatives',
                     'embedding_dim', 'channels', 'depth', 'reduced_size'):
            v = getattr(self, name)
            if not isinstance(v, int) or v <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if self.kernel_size < 2 or self.learning_rate <= 0 or (self.svm_C is not None and self.svm_C <= 0):
            raise ValueError('Invalid SSL hyperparameters')
        if self.aggregation not in ('uniform', 'sample_weighted') or self.svm_cv not in (0, 5):
            raise ValueError('Use uniform/sample_weighted aggregation and svm_cv=5 (or 0 for fixed-C smoke checks)')
        if self.svm_cv == 0 and self.svm_C is None:
            raise ValueError('Fixed C required when CV is disabled for a smoke check')
        if self.compared_length is not None and self.compared_length < 1:
            raise ValueError('compared_length must be positive or None')


def iq_tensor(data):
    x = torch.as_tensor(np.asarray(data), dtype=torch.float64)
    if x.ndim != 3 or x.shape[1] != 2 or x.shape[0] < 2 or x.shape[2] < 2:
        raise ValueError('Expected I/Q array (sequences>=2, 2, time>=2); tabular input is unsupported')
    if not torch.isfinite(x).all():
        raise ValueError('I/Q data contains nonfinite samples')
    return x


def triplet_indices(n, batch_size, negatives, rng):
    reference = rng.integers(n, size=batch_size)
    # Draw from n-1 alternatives; remap across the excluded reference ID.
    negative = rng.integers(n-1, size=(batch_size, negatives))
    negative += negative >= reference[:, None]
    return reference, negative


def sample_triplet(x, batch_size, negatives, rng):
    """Paper-equation diagnostic sampler; not the author-release training path.

    No labels accepted. Positive is contained in the reference sequence.

    Crop lengths are sampled uniformly. This is an explicit engineering
    choice; the AMC paper does not specify its crop-length distribution.
    """
    ref, neg = triplet_indices(len(x), batch_size, negatives, rng)
    length = int(rng.integers(1, x.shape[-1]+1))
    starts = rng.integers(x.shape[-1]-length+1, size=batch_size)
    positive = torch.stack([x[i, :, start:start+length] for i, start in zip(ref, starts)])
    return x[ref], positive, x[neg.reshape(-1)]


def contrastive_loss(reference, positive, negative):
    """Eq. (1), batch mean of positive plus SUM over K negatives."""
    positive_dot = (reference * positive).sum(-1)
    negative_dot = (reference[:, None, :] * negative).sum(-1)
    return (F.softplus(-positive_dot) + F.softplus(negative_dot).sum(-1)).mean()


def encoder(config):
    config.validate()
    return CausalCNNEncoder(2, config.channels, config.depth,
                           config.reduced_size, config.embedding_dim,
                           config.kernel_size).double()


def architecture_summary(model):
    convolutions = [m for m in model.modules() if isinstance(m, torch.nn.Conv1d)]
    return dict(total_conv1d=len(convolutions),
                main_convolutions=sum(m.kernel_size != (1,) for m in convolutions),
                projection_convolutions=sum(m.kernel_size == (1,) for m in convolutions),
                parameters=sum(p.numel() for p in model.parameters()),
                layer_interpretation='author released depth semantics; see SOURCE.json')


def new_encoder(config):
    with torch.random.fork_rng():
        torch.manual_seed(config.seed)
        return encoder(config)


def train_local(state, x, config, client_id, round_id):
    """Shared server/PeerSim local solver using the unmodified author loss.

    Isolate RNGs so node activation order does not change a client's samples.
    The original subprocess launcher did not specify reproducible seeds.
    """
    seed = int(np.random.SeedSequence([config.seed, client_id, round_id]).generate_state(1)[0])
    numpy_state = np.random.get_state()
    try:
        np.random.seed(seed)
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            model = encoder(config)
            model.load_state_dict(state)
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
            criterion = TripletLoss(config.compared_length, config.negatives, 1, Sim_CSE=False)
            loader = torch.utils.data.DataLoader(x, batch_size=config.batch_size, shuffle=True)
            iterator = iter(loader)
            total = 0.
            for _ in range(config.local_steps):
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(loader)
                    batch = next(iterator)
                optimizer.zero_grad()
                loss = criterion(batch, model, x)
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite author triplet loss')
                loss.backward()
                optimizer.step()
                total += float(loss.detach())
            return {k: v.detach().clone() for k, v in model.state_dict().items()}, total/config.local_steps
    finally:
        np.random.set_state(numpy_state)


def aggregate_states(records, config):
    """Canonical origin order; records=(encoder state, unlabeled count, loss)."""
    weights = np.array([r[1] for r in records], dtype=float)
    if config.aggregation == 'uniform':
        weights[:] = 1.
    weights /= weights.sum()
    state = {key: sum(float(w)*record[0][key] for record, w in zip(records, weights))
             for key in records[0][0]}
    return state, float(weights @ np.array([r[2] for r in records]))


def source_metadata(config, model):
    return dict(config=asdict(config), architecture=architecture_summary(model),
        source_repository='https://github.com/usmanakua927/FedSSL-AMC_Github',
        source_commit='7cf7dc290aa208ef525c56b4591a501e547b8abf',
        validation_status='author_release_adaptation', paper_version='2510.04927v1 repository',
        paper_prose_differences=['released depth=7/channels=80', 'negative terms averaged',
            'negative sampler can choose reference sequence', 'released batch=50',
            'uniform encoder aggregation'],
        deployment_changes=['shared deterministic initialization', 'isolated per-client/round RNGs',
            'final-only test evaluation', 'CPU-compatible runner; CV n_jobs=1'])


def train_encoder(unlabeled_clients, config=None, adjacency=None):
    config = config or SSLConfig()
    config.validate()
    clients = [iq_tensor(x) for x in unlabeled_clients]
    if not clients:
        raise ValueError('At least one client required')
    # Fork RNG keeps this reference from changing unrelated caller randomness.
    model = new_encoder(config)
    transport = ExactAggregation(adjacency)
    history = []
    for round_id in range(config.rounds):
        state = model.state_dict()
        size = sum(v.numel()*v.element_size() for v in state.values())
        transport.broadcast(size, len(clients))
        updates = []
        for client_id, x in enumerate(clients):
            updated, loss = train_local(state, x, config, client_id, round_id)
            updates.append((updated, len(x), loss))
        sizes = [sum(v.numel()*v.element_size() for v in r[0].values()) + 16 for r in updates]
        gathered = transport.gather(updates, sizes)
        # Each peer computes the same result from its own received contributions.
        aggregates = [aggregate_states(peer, config)[0] for peer in gathered]
        if any(not torch.equal(aggregates[0][k], a[k]) for a in aggregates[1:] for k in state):
            raise RuntimeError('Peers disagree after exact aggregation')
        model.load_state_dict(aggregates[0])
        history.append(dict(round=round_id+1, train_triplet_loss=aggregate_states(updates, config)[1]))
    # Algorithm 1, line 11: final encoder reaches all clients for local heads.
    transport.broadcast(size, len(clients))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    metadata = source_metadata(config, model)
    metadata.update(communication_bytes=transport.bytes,
        link_rounds=transport.link_rounds, history=history,
        stop_reason='author_fixed_round_budget', converged=None)
    return model, metadata


def embeddings(model, x, batch_size=128):
    x = iq_tensor(x)
    with torch.no_grad():
        return torch.cat([model(x[i:i+batch_size]) for i in range(0, len(x), batch_size)]).numpy()


def fit_local_heads(model, labeled_clients, config):
    """One independently fitted SVM per client; no head communication."""
    if any(p.requires_grad for p in model.parameters()):
        raise ValueError('Freeze encoder before fitting SVMs')
    heads = []
    for x, y in labeled_clients:
        if len(x) != len(y) or len(np.unique(y)) < 2:
            raise ValueError('Each local labeled set needs aligned samples and >=2 classes')
        head = LinearSVC(random_state=0, tol=1e-5, max_iter=10000)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', ConvergenceWarning)
            features = embeddings(model, x)
            if config.svm_cv:
                if np.unique(y, return_counts=True)[1].min() < config.svm_cv:
                    raise ValueError('Author 5-fold SVM CV requires at least 5 labeled samples per class')
                values = [config.svm_C] if config.svm_C is not None else [1e-4,1e-3,.01,.1,1,10,100,1000,10000]
                search = GridSearchCV(head, {'C': values}, cv=5, n_jobs=1, error_score='raise')
                search.fit(features, np.asarray(y))
                head = search.best_estimator_
            else:
                head.set_params(C=config.svm_C)
                head.fit(features, np.asarray(y))
            if head.n_iter_ >= head.max_iter:
                raise RuntimeError('Selected local LinearSVC did not meet the author solver iteration budget')
        heads.append(head)
    return heads


def evaluate_local_heads(model, heads, test_clients):
    if len(heads) != len(test_clients):
        raise ValueError('One test set per local classifier required')
    return [float(head.score(embeddings(model, x), y))
            for head, (x, y) in zip(heads, test_clients)]
