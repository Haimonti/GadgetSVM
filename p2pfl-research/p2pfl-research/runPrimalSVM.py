from primalSVM import LinearSVM
from p2pfl.learning.frameworks.pytorch.lightning_model import LightningModel

def model_build_fn(*args, **kwargs):
    compression = kwargs.pop("compression", None)
    return LightningModel(LinearSVM(*args, **kwargs), compression=compression)