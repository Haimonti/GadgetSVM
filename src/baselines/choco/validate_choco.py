"""Validate the CHOCO-SGD port against the official epfml/ChocoSGD convex code.

Runs the *official* `LogisticDecentralizedSGD.fit` (method="choco", ring, full
compression) with its logistic factor swapped for the hinge indicator — the
module-level `sigmoid(-y a.x)` becomes `[y a.x < 1]`, which turns their
`minus_grad` into exactly our hinge subgradient — and checks that our
`ChocoSimulation` reproduces every node's x after one epoch (to 1e-10), given
the same x0, ring W and per-step sample sequence.

    # download https://github.com/epfml/ChocoSGD (zip) to /some/dir/ChocoSGD (needs networkx)
    python src/baselines/choco/validate_choco.py /some/dir/ChocoSGD/convex_code

Exits 0 on PASS, 1 on mismatch, 2 if the official code cannot be imported.
"""

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

K, D, N_PER, LAM, SEED_X0, SEED_RUN = 4, 20, 25, 1e-2, 3, 11
LR, TAU, CONS = 0.1, 20.0, 0.7


def _problem():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(K * N_PER, D))
    y = np.where(A @ rng.normal(size=D) >= 0, 1.0, -1.0)
    return A, y


def official(code_dir, A, y):
    sys.path.insert(0, str(code_dir))
    try:
        import logistic
        from parameters import Parameters
    except Exception as e:                       # noqa: BLE001
        print(f"cannot import the official CHOCO code from {code_dir}: {e}")
        sys.exit(2)
    logistic.sigmoid = lambda z: (np.asarray(z) > -1.0).astype(float)   # hinge indicator
    p = Parameters(num_epoch=1, lr_type="decay", initial_lr=LR, tau=TAU, regularizer=LAM,
                   consensus_lr=CONS, quantization="full", n_cores=K, topology="ring",
                   method="choco", distribute_data=True, split_data_strategy="naive",
                   random_seed=SEED_RUN, split_data_random_seed=0, estimate="final",
                   name="validate")
    m = logistic.LogisticDecentralizedSGD(p)
    np.random.seed(SEED_X0)
    m.fit(A, y)
    np.random.seed(SEED_X0)                      # replay their x0 draw
    x0 = np.random.normal(0, logistic.INIT_WEIGHT_STD, size=(D,))
    rs = np.random.RandomState(SEED_RUN)         # replay their sample sequence
    picks = [[] for _ in range(K)]
    for _ in range(N_PER):
        for k in range(K):
            picks[k].append(int(rs.choice(np.arange(k * N_PER, (k + 1) * N_PER))) - k * N_PER)
    return m.x.copy(), x0, picks


class _Replay:
    """Stands in for a node's np.random.Generator: returns the official samples."""
    def __init__(self, seq):
        self.seq = list(seq)

    def choice(self, n, size):
        return np.array([self.seq.pop(0) for _ in range(size)])


def ours(A, y, x0, picks):
    from src.config import CONFIG
    from src.peersim_python.core import Network
    from src.baselines.choco.simulation import ChocoSimulation
    shards = []
    for k in range(K):
        sl = slice(k * N_PER, (k + 1) * N_PER)
        shards.append({"X_csr": sp.csr_matrix(A[sl]), "y": y[sl], "X_test": sp.csr_matrix(A[:5]),
                       "y_test": y[:5], "n_local": N_PER})
    cfg = dict(CONFIG, NUM_WORKERS=K, TOPOLOGY="ring", LAMBDA=LAM, SEED=0,
               CHOCO_COMPRESSOR="full", CHOCO_CONSENSUS_LR=CONS, CHOCO_LR_TYPE="decay",
               CHOCO_INITIAL_LR=LR, CHOCO_TAU=TAU, CHOCO_BATCH=1,
               STOP_ON_THRESHOLD=False, STOP_ON_PLATEAU=False, EVAL_EVERY=10 ** 9)
    sim = ChocoSimulation(cfg, shards).build()

    class Inject:
        """After the data initialiser: impose the official x0 and sample sequence."""
        def execute(self):
            for k in range(K):
                p = Network.get(k).getProtocol(ChocoSimulation.CHOCO_PID)
                p.x = x0.copy()
                p.x_hat = {j: x0.copy() for j in p.x_hat}
                p.rng = _Replay(picks[k])
            return False
    sim.sim.initializers.append(Inject())
    sim.run(N_PER)
    return np.stack([Network.get(k).getProtocol(ChocoSimulation.CHOCO_PID).x
                     for k in range(K)], axis=1)


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    A, y = _problem()
    ref_x, x0, picks = official(Path(sys.argv[1]), A, y)
    our_x = ours(A, y, x0, picks)
    err = float(np.max(np.abs(ref_x - our_x)))
    ok = err < 1e-10
    print(f"CHOCO-SGD port vs official: {N_PER} steps x {K} nodes, max |x_ours - x_official| "
          f"= {err:.3e} -> {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
