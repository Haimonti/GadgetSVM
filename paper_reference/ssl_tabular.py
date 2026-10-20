"""Explicit tabular adaptation of FedSSL-AMC, not a paper reproduction.

MLP + SCARF-inspired marginal feature corruption replaces causal CNN/crops.
Retains released logistic positive + mean-negative objective, fresh local Adam,
uniform encoder aggregation and frozen encoder with local train-CV LinearSVC.
"""
from dataclasses import dataclass, asdict
import warnings
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.exceptions import ConvergenceWarning
from .ssl import triplet_indices,aggregate_states


@dataclass
class TabularSSLConfig:
    input_dim:int=123
    hidden_dim:int=128
    embedding_dim:int=64
    rounds:int=1500
    local_steps:int=10
    batch_size:int=128
    negatives:int=10
    learning_rate:float=.001
    corruption_rate:float=.3
    aggregation:str='uniform'
    svm_C:float|None=None
    svm_cv:int=5
    seed:int=42
    monitor_samples:int=256
    stability_window:int=50
    stability_checks:int=2
    rtol:float=.001

    def validate(self):
        for name in ('input_dim','hidden_dim','embedding_dim','rounds','local_steps','batch_size',
                     'negatives','monitor_samples','stability_window','stability_checks'):
            if not isinstance(getattr(self,name),int) or getattr(self,name)<1:
                raise ValueError(f'{name} must be a positive integer')
        if not 0<self.corruption_rate<1 or self.learning_rate<=0 or self.rtol<0:
            raise ValueError('Invalid corruption, learning rate or stability tolerance')
        if self.aggregation not in ('uniform','sample_weighted') or self.svm_cv not in (0,5):
            raise ValueError('Invalid aggregation or CV mode')
        if self.svm_C is not None and self.svm_C<=0:raise ValueError('Positive SVM C required')
        if self.svm_cv==0 and self.svm_C is None:raise ValueError('Fixed C required without CV')


def input_tensor(x):
    if hasattr(x,'toarray'):x=x.toarray()
    x=torch.as_tensor(np.asarray(x),dtype=torch.float64)
    if x.ndim!=2 or len(x)<2 or x.shape[1]<1 or not torch.isfinite(x).all():
        raise ValueError('Finite table (rows>=2, features>=1) required')
    return x


def new_encoder(config):
    config.validate()
    with torch.random.fork_rng():
        torch.manual_seed(config.seed)
        return nn.Sequential(nn.Linear(config.input_dim,config.hidden_dim),nn.ReLU(),
            nn.Linear(config.hidden_dim,config.embedding_dim)).double()


def sample_views(x,config,rng,batch_size=None):
    """Only local feature values, never labels or test data, enter the sampler.

    Reference and corrupt reference are positive; negatives exclude reference
    row (explicit change from the author's potentially self-negative sampler).
    Different columns use independently drawn local donor rows. One-hot group
    consistency is not guaranteed; these are augmentation views, not new data.
    """
    size=batch_size or config.batch_size
    ref,neg=triplet_indices(len(x),size,config.negatives,rng)
    donors=rng.integers(len(x),size=(size,x.shape[1]))
    mask=torch.from_numpy(rng.random((size,x.shape[1]))<config.corruption_rate)
    positive=x[ref].clone()
    replacement=x[donors,np.arange(x.shape[1])[None,:]]
    positive[mask]=replacement[mask]
    return x[ref],positive,x[neg.reshape(-1)]


def loss_on_views(model,views,negatives):
    ref,pos,neg=views
    a,b,c=model(ref),model(pos),model(neg)
    c=c.reshape(len(a),negatives,-1)
    # Released loss: positive logistic plus MEAN over K negative terms.
    return (F.softplus(-(a*b).sum(-1))+F.softplus((a[:,None,:]*c).sum(-1)).mean(-1)).mean()


def train_local(state,x,config,client_id,round_id):
    seed=int(np.random.SeedSequence([config.seed,client_id,round_id,137]).generate_state(1)[0])
    rng=np.random.default_rng(seed)
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        model=new_encoder(config);model.load_state_dict(state);model.train()
        optimizer=torch.optim.Adam(model.parameters(),lr=config.learning_rate)
        total=0.
        for _ in range(config.local_steps):
            optimizer.zero_grad()
            loss=loss_on_views(model,sample_views(x,config,rng),config.negatives)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite tabular contrastive loss')
            loss.backward();optimizer.step();total+=float(loss.detach())
        return {k:v.detach().clone() for k,v in model.state_dict().items()},total/config.local_steps


def monitor_loss(model,x,config,client_id):
    # Fixed anchors, donors, masks and negatives in every monitoring call.
    rng=np.random.default_rng(np.random.SeedSequence([config.seed,client_id,271]))
    with torch.no_grad():
        return float(loss_on_views(model,sample_views(x,config,rng,config.monitor_samples),config.negatives))


def embeddings(model,x,batch_size=512):
    x=input_tensor(x)
    with torch.no_grad():
        return torch.cat([model(x[i:i+batch_size]) for i in range(0,len(x),batch_size)]).numpy()


def fit_local_heads(model,labeled_clients,config):
    if any(p.requires_grad for p in model.parameters()):raise ValueError('Freeze encoder first')
    heads=[]
    for x,y in labeled_clients:
        features=embeddings(model,x);y=np.asarray(y)
        if len(y)!=len(features) or len(np.unique(y))!=2:raise ValueError('Aligned binary labels required')
        head=LinearSVC(random_state=config.seed,tol=1e-5,max_iter=100000)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',ConvergenceWarning)
            if config.svm_cv:
                if np.unique(y,return_counts=True)[1].min()<5:raise ValueError('Need >=5 rows/class for CV')
                values=[config.svm_C] if config.svm_C is not None else [1e-4,1e-3,.01,.1,1,10,100,1000,10000]
                search=GridSearchCV(head,{'C':values},cv=5,n_jobs=1,error_score='raise')
                search.fit(features,y);head=search.best_estimator_
            else:head.set_params(C=config.svm_C);head.fit(features,y)
        if head.n_iter_>=head.max_iter:raise RuntimeError('Selected tabular head hit solver budget')
        heads.append(head)
    return heads


def evaluate_local_heads(model,heads,test_clients):
    if len(heads)!=len(test_clients):raise ValueError('One test set per classifier required')
    return [float(h.score(embeddings(model,x),y)) for h,(x,y) in zip(heads,test_clients)]


def source_metadata(config,model):
    return dict(config=asdict(config),parameters=sum(p.numel() for p in model.parameters()),
        validation_status='tabular_adaptation_not_paper_reproduction',
        source_repository='https://github.com/usmanakua927/FedSSL-AMC_Github',
        source_commit='7cf7dc290aa208ef525c56b4591a501e547b8abf',
        augmentation_reference='https://arxiv.org/abs/2106.15147',
        retained=['released positive/mean-negative logistic objective','uniform encoder aggregation',
            'fresh local Adam per round','frozen encoder, local train-CV LinearSVC'],
        changed=['MLP instead of causal CNN','local marginal feature corruption instead of crops',
            'exclude anchor row from negatives','fixed train-only monitor and operational plateau stop',
            'all training labels for final local heads; 100000 head iteration cap'],
        reproduction_claim=False)
