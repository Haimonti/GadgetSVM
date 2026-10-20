"""Synthetic baseband generator following FedSSL-AMC Eq. (34)-(35).

No pulse shaping is specified in the paper. This generator uses independent,
unit-power constellation symbols; this assumption is recorded in the report.
Smoke scaling preserves class ratios but is not the published dataset budget.
"""
import numpy as np

COUNTS = np.array([[60000,60000,10000,10000], [10000,60000,60000,10000],
                   [10000,10000,60000,60000], [60000,10000,10000,60000]])


def signals(counts, rng, length=100):
    grid = np.array([-3., -1., 1., 3.])
    qam = (grid[:, None] + 1j*grid[None, :]).reshape(-1) / np.sqrt(10)
    constellations = [np.exp(2j*np.pi*np.arange(m)/m) for m in (2,4,8)] + [qam]
    xs, ys = [], []
    for label, (count, constellation) in enumerate(zip(counts, constellations)):
        symbols = rng.choice(constellation, size=(int(count), length))
        gain = np.minimum(rng.rayleigh(size=(int(count),1)), 1.)
        phase = rng.uniform(0, np.pi/16, size=(int(count),1))
        clean = gain*np.exp(1j*(phase+2*np.pi*.01*np.arange(length)/length))*symbols
        snr = rng.uniform(-10, 10, size=(int(count),1))
        noise = rng.normal(size=clean.shape)+1j*rng.normal(size=clean.shape)
        # Normalize realized power to match Eq. (35)'s sampled SNR.
        noise *= np.sqrt(np.sum(abs(clean)**2,axis=1,keepdims=True)
                         /np.sum(abs(noise)**2,axis=1,keepdims=True)/10**(snr/10))
        received = clean+noise
        xs.append(np.stack([received.real, received.imag],axis=1).astype(np.float32))
        ys.append(np.full(int(count),label,dtype=np.int64))
    x, y = np.concatenate(xs), np.concatenate(ys)
    order = rng.permutation(len(y))
    return x[order], y[order]


def synthetic_clients(seed=0, scale=1.):
    if not 0 < scale <= 1:
        raise ValueError('scale must lie in (0,1]')
    unlabeled, labeled, tests = [], [], []
    streams = np.random.SeedSequence(seed).spawn(12)
    for c in range(4):
        for dest, counts, stream in zip((unlabeled,labeled,tests),
               (COUNTS[c], COUNTS[c]//50, COUNTS[c]//500), streams[3*c:3*c+3]):
            data = signals(np.maximum(1, np.rint(counts*scale).astype(int)), np.random.default_rng(stream))
            dest.append(data[0] if dest is unlabeled else data)
    return unlabeled, labeled, tests


def author_synthetic_clients(seed=0, scale=1.):
    """Released CFO heterogeneous fixture using original signal functions.

    Only counts are scaled: sequence length 384, Rayleigh scale .35 and
    per-client CFO mixtures follow the pinned author's generator.
    """
    from .author_ssl.signal_functions import Generate_symbols
    if not 0 < scale <= 1:
        raise ValueError('scale must lie in (0,1]')
    counts=np.roll(COUNTS//10,-1,axis=0)
    probabilities=np.array([[.4,.4,.1,.1],[.4,.1,.4,.1],[.1,.4,.4,.1],[.1,.1,.4,.4]])
    low=np.array([0,.01,.1,1]);high=np.array([.01,.1,1,20])
    old=np.random.get_state()
    unlabeled,labeled,tests=[],[],[]
    try:
        np.random.seed(seed)
        for c in range(4):
            for dest,budget in zip((unlabeled,labeled,tests),(counts[c],counts[c]//5,counts[c]//50)):
                xs,ys=[],[]
                for label,count in enumerate(np.maximum(1,np.rint(budget*scale).astype(int))):
                    for _ in range(int(count)):
                        index=np.random.choice(4,p=probabilities[c])
                        cfo=np.random.uniform(low[index],high[index])
                        z=Generate_symbols(label,num_symbols=384,rayleigh_scale=.35,sanity_check=False,del_f=cfo)
                        xs.append(np.stack([z.real,z.imag]));ys.append(label)
                order=np.random.permutation(len(ys));x=np.asarray(xs)[order];y=np.asarray(ys)[order]
                dest.append(x if dest is unlabeled else (x,y))
        return unlabeled,labeled,tests
    finally:
        np.random.set_state(old)
