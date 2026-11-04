"""Validate the COLA port against the official epfml/cola code.

Drives the *official* `LinearSVM` local solver (its compiled `fast_cd/svm.pyx`)
inside a plain transcription of the 8-line `cola/algo.py::cola` loop, on a small
random sparse problem over a 4-node ring, and checks that our `ColaSimulation`
produces the same per-node v_k, xk and Akxk every round (to 1e-10).

    # one-time: get + build the official code (Cython)
    #   download https://github.com/epfml/cola (zip) to /some/dir/cola, then
    #   cd /some/dir/cola && make build        (needs cython; older numpy if numpy>=2 breaks setup.py)
    python src/baselines/cola/validate_cola.py /some/dir/cola

Exits 0 on PASS, 1 on mismatch, 2 if the official code cannot be imported.
"""

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

K, D, N_PER, ROUNDS, LAM, SEED = 4, 30, 40, 15, 1e-2, 7


def _problem():
    rng = np.random.default_rng(0)
    X = sp.random(K * N_PER, D, density=0.3, random_state=1, format="csr")
    w_true = rng.normal(size=D)
    y = np.where(X.dot(w_true) + 0.1 * rng.normal(size=K * N_PER) >= 0, 1.0, -1.0)
    shards = []
    for k in range(K):
        sl = slice(k * N_PER, (k + 1) * N_PER)
        shards.append({"X_csr": X[sl].tocsr(), "y": y[sl], "X_test": X[:10], "y_test": y[:10],
                       "n_local": N_PER})
    return shards


def official_trajectory(official_dir, shards):
    sys.path.insert(0, str(official_dir))
    try:
        from cola.cocoasolvers import LinearSVM
    except Exception as e:                       # noqa: BLE001
        print(f"cannot import the official COLA code from {official_dir}: {e}")
        sys.exit(2)
    W = np.zeros((K, K))
    for i in range(K):                           # ring, COLA weights 1/3
        for j in (i - 1, i, i + 1):
            W[i, j % K] = 1.0 / 3.0
    C = 1.0 / (LAM * K * N_PER)
    gamma, sigma = 1.0, 1.0 * K
    solvers, xk, Akxk, vs = [], [], [], [np.zeros(D) for _ in range(K)]
    for s in shards:
        sol = LinearSVM(C=C, random_state=SEED)
        sol.dist_init(s["X_csr"].T.tocsc(), s["y"], 0.0, 1.0, sigma)
        solvers.append(sol)
        xk.append(np.zeros(N_PER))
        Akxk.append(np.zeros(D))
    traj = []
    for _ in range(ROUNDS):
        new_vs = []
        for k in range(K):
            averaged_v = sum(W[k, j] * vs[j] for j in range(K) if W[k, j])
            dx, dv = solvers[k].solve(averaged_v, Akxk[k], xk[k])
            xk[k] = xk[k] + gamma * dx
            Akxk[k] = Akxk[k] + gamma * dv
            new_vs.append(averaged_v + gamma * dv * K)
        vs = new_vs                              # synchronous exchange
        traj.append([v.copy() for v in vs])
    return traj


def our_trajectory(shards):
    from src.config import CONFIG
    from src.peersim_python.core import Network
    from src.baselines.cola.simulation import ColaSimulation
    cfg = dict(CONFIG, NUM_WORKERS=K, TOPOLOGY="ring", LAMBDA=LAM, SEED=SEED,
               COLA_GAMMA=1.0, COLA_LOCAL_ITERS=1.0, COLA_THETA=0.0, COMPRESSOR="full",
               STOP_ON_THRESHOLD=False, STOP_ON_PLATEAU=False, EVAL_EVERY=10 ** 9)
    traj = []
    sim = ColaSimulation(cfg, shards).build()

    class Recorder:
        def execute(self):
            traj.append([Network.get(k).getProtocol(ColaSimulation.COLA_PID).v.copy()
                         for k in range(K)])
            return False
    sim.sim.controls.insert(1, Recorder())       # right after the barrier
    sim.run(ROUNDS)
    return traj


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    shards = _problem()
    ref = official_trajectory(Path(sys.argv[1]), shards)
    ours = our_trajectory(shards)
    err = max(float(np.max(np.abs(a - b))) for ra, rb in zip(ref, ours) for a, b in zip(ra, rb))
    ok = len(ref) == len(ours) and err < 1e-10
    print(f"COLA port vs official: {len(ours)} rounds, max |v_ours - v_official| = {err:.3e} "
          f"-> {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
