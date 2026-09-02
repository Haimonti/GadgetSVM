"""Run Gossip-FedAvg-SVM on the PeerSim engine.

Proof-of-pipeline driver for `p2p/fedavg_protocol.py`. Deliberately small:
it wires one topology, hands out shards, runs the cycle-driven simulation, and
reports per-node accuracy plus the consensus error between nodes.

It reuses Sreekar's engine and his DataInitializer unchanged — the protocol was
written to his set_data signature specifically so that no file under
src/network_layer/peersim_python/ needs touching.

    python run_fedavg_peersim.py                       # defaults below
    python run_fedavg_peersim.py --cycles 30 --nodes 10 --topology ring

Once the remaining five protocols land, this grows into the full grid runner;
for now it exists to show the pipeline end to end on real data.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.datasets import load_svmlight_file

import benchmark as data_lib

from src.network_layer.peersim_python.core import Network, GeneralNode, CommonState
from src.network_layer.peersim_python.idle_protocol import IdleProtocol
from src.network_layer.peersim_python.cdsim import CDSimulator
from src.network_layer.peersim_python.dynamics import (
    WireKOut, WireRing, WireFull, WireStar, WireMesh,
)
from src.network_layer.peersim_python.observers import DataInitializer
from src.network_layer.peersim_python.logger import logger

from p2p import PROTOCOLS

LINKABLE_PID = 0
ALGO_PID = 1


def topology(name, k):
    wires = {
        "random_kout": lambda: WireKOut(LINKABLE_PID, k, undir=True),
        "ring":        lambda: WireRing(LINKABLE_PID, undir=True),
        "full":        lambda: WireFull(LINKABLE_PID, undir=True),
        "star":        lambda: WireStar(LINKABLE_PID, undir=True),
        "mesh":        lambda: WireMesh(LINKABLE_PID, undir=True),
    }
    if name not in wires:
        raise ValueError(f"Unknown topology '{name}'. Choose: {sorted(wires)}")
    return wires[name]()


def to_pm1(y):
    """Map a two-class label vector to {-1, +1} (larger class value -> +1)."""
    y = np.asarray(y, dtype=np.float32)
    vals = np.unique(y)
    if len(vals) == 2:
        return np.where(y == vals.max(), 1.0, -1.0).astype(np.float32)
    return np.sign(y).astype(np.float32)


def overlay_components(n_nodes):
    """Number of connected components in the wired overlay.

    Worth recording: WireKOut with k=1 gives each node a single out-edge, and
    an undirected random 1-out graph is very often disconnected — at n=10,
    seed 42 it splits 7+3. Contributions cannot cross a component boundary, so
    any protocol whose aggregation needs to see every node is capped by this
    no matter how many cycles or how much bandwidth it is given.
    """
    adj = {i: set() for i in range(n_nodes)}
    for i in range(n_nodes):
        lk = Network.get(i).getProtocol(LINKABLE_PID)
        for j in range(lk.degree()):
            t = lk.getNeighbor(j).getIndex()
            adj[i].add(t)
            adj[t].add(i)
    seen, comps = set(), 0
    for s0 in range(n_nodes):
        if s0 in seen:
            continue
        comps += 1
        stack, comp = [s0], set()
        while stack:
            u = stack.pop()
            if u in comp:
                continue
            comp.add(u)
            stack += [v for v in adj[u] if v not in comp]
        seen |= comp
    return comps


def parse_scheme(label):
    """'dirichlet_0.1' -> ('dirichlet', {'alpha': 0.1});  'iid' -> ('iid', {})."""
    if label.startswith("dirichlet"):
        return "dirichlet", {"alpha": float(label.split("_")[1])}
    return label, {}


def load_shards(path, n_nodes, seed, test_fraction, max_samples, scheme="iid"):
    """Load a LIBSVM file, hold out a test split, and shard the rest per node.

    Every node is evaluated on the *same* held-out test set, so per-node accuracy
    is comparable across nodes and against the server baseline. (Sreekar's
    peersim_run.py splits the test set across nodes instead — that difference is
    open question #8 on the merge checklist.)
    """
    X, y = load_svmlight_file(str(path))
    y = to_pm1(y)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(X.shape[0])
    X, y = X[perm].tocsr(), y[perm]

    if max_samples and X.shape[0] > max_samples:
        X, y = X[:max_samples].tocsr(), y[:max_samples]

    n_test = int(test_fraction * X.shape[0])
    X_test, y_test = X[:n_test].tocsr(), y[:n_test]
    X_tr, y_tr = X[n_test:].tocsr(), y[n_test:]
    logger.info("data", f"{X.shape[0]} samples, {X.shape[1]} features -> "
                        f"{X_tr.shape[0]} train / {n_test} test")

    # Same partitioner as run_benchmark.py, so a P2P run and its server
    # baseline see identical shards for a given (dataset, scheme, K, seed).
    sch, kw = parse_scheme(scheme)
    parts = data_lib.partition(y_tr, n_nodes, sch, seed=seed, **kw)
    logger.info("data", f"scheme={scheme} shard sizes={[len(q) for q in parts]}")
    return [{
        "X_csr":   X_tr[p].tocsr(),
        "y":       y_tr[p],
        "X_test":  X_test,
        "y_test":  y_test,
        "n_local": len(p),
    } for p in parts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="fedavg_svm",
                    help="fedavg_svm | cocoa | cocoa_plus | bdsvm | fdr_svm | fedssl_amc")
    ap.add_argument("--data", default="data/covtype.libsvm.binary.scale")
    ap.add_argument("--nodes", type=int, default=10)
    ap.add_argument("--cycles", type=int, default=30)
    ap.add_argument("--topology", default="random_kout")
    ap.add_argument("--gossip-k", type=int, default=1)
    ap.add_argument("--local-steps", type=int, default=100)
    ap.add_argument("--lambda-reg", type=float, default=1e-4)
    ap.add_argument("--t-global", action="store_true",
                    help="textbook Pegasos (t runs across the whole run) instead of "
                         "restarting t each round as methods/fedavg_svm.py does")
    ap.add_argument("--scheme", default="iid",
                    help="iid | dirichlet_<alpha> | label_skew")
    ap.add_argument("--test-fraction", type=float, default=0.2)
    ap.add_argument("--max-samples", type=int, default=50_000)
    ap.add_argument("--P", type=int, default=None,
                    help="bdsvm only: budget — number of pre-image vectors")
    ap.add_argument("--gamma", type=float, default=None,
                    help="bdsvm only: RBF width (default: median heuristic)")
    ap.add_argument("--preimage", default="uniform",
                    help="bdsvm only: uniform | unit (unit-norm, for sparse "
                         "high-dimensional data such as rcv1)")
    ap.add_argument("--gossip-entries", type=int, default=2,
                    help="bdsvm only: table entries carried per message")
    ap.add_argument("--budget", type=int, default=200,
                    help="bdsvm only: samples kept per cycle")
    ap.add_argument("--csv", default=None,
                    help="append one row per node to this CSV")
    ap.add_argument("--tag", default="",
                    help="free-form label recorded in the CSV (e.g. dataset)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not Path(args.data).exists():
        raise SystemExit(f"{args.data} not found — download a LIBSVM binary dataset there first.")

    CommonState.initializeRandom(args.seed)
    shards = load_shards(args.data, args.nodes, args.seed,
                         args.test_fraction, args.max_samples, args.scheme)

    prototype_algo = PROTOCOLS[args.method]()
    if hasattr(prototype_algo, "n_local_steps"):
        prototype_algo.n_local_steps = args.local_steps
    if args.P is not None and hasattr(prototype_algo, "P"):
        prototype_algo.P = args.P
    if args.gamma is not None and hasattr(prototype_algo, "gamma"):
        prototype_algo.gamma = args.gamma
    if hasattr(prototype_algo, "preimage"):
        prototype_algo.preimage = args.preimage
    if hasattr(prototype_algo, "gossip_entries"):
        prototype_algo.gossip_entries = args.gossip_entries
    if hasattr(prototype_algo, "n_nodes"):
        prototype_algo.n_nodes = args.nodes
    if hasattr(prototype_algo, "budget"):
        prototype_algo.budget = args.budget
    if hasattr(prototype_algo, "encoder_cycles"):
        prototype_algo.encoder_cycles = max(args.cycles // 3, 1)
    if hasattr(prototype_algo, "t_global"):
        prototype_algo.t_global = args.t_global
    if hasattr(prototype_algo, "n_global"):
        # CoCoA's increments only share a scale if every node divides by the
        # same global sample count (see p2p/cocoa_protocol.py).
        prototype_algo.n_global = sum(s["n_local"] for s in shards)
    Network.reset(args.nodes, GeneralNode([IdleProtocol(), prototype_algo]))
    logger.info("network", f"{args.nodes} nodes (protocol 0=Linkable, 1={args.method})")

    sim = CDSimulator(
        cycles=args.cycles,
        initializers=[
            topology(args.topology, args.gossip_k),
            # t0_fraction is part of SDCAProtocol's signature; FedAvg ignores it.
            DataInitializer(ALGO_PID, shards, args.lambda_reg, 0.5, args.gossip_k),
        ],
        controls=[],
        activation="shuffle",
    )
    logger.info("main", f"method={args.method}, topology={args.topology}, k={args.gossip_k}, "
                        f"cycles={args.cycles}, lambda={args.lambda_reg}, "
                        f"local_steps={args.local_steps}, t_global={args.t_global}")
    # Must be read AFTER nextExperiment: the Wire* control is an initializer
    # and only runs inside it, so the overlay does not exist before that.
    sim.nextExperiment()
    n_comp = overlay_components(args.nodes)
    logger.info("network", f"overlay has {n_comp} connected component(s)")

    protos = [Network.get(i).getProtocol(ALGO_PID) for i in range(args.nodes)]
    accs = np.array([p.accuracy() for p in protos])
    # Protocols name their parameter vector differently (w for the primal
    # methods, beta for the kernel one), so pick whichever this one carries.
    def _params(pr):
        v = getattr(pr, "w", None)
        return np.asarray(pr.beta if v is None else v, dtype=np.float64)

    W = np.stack([_params(p) for p in protos])
    consensus = float(np.mean(np.linalg.norm(W - W.mean(axis=0), axis=1)))
    total_mb = sum(p.comm_bytes for p in protos) / 1e6

    print(f"\n{'node':>5} {'n_local':>9} {'test_acc':>10} {'hinge':>10} {'MB sent':>9}")
    print("-" * 48)
    for i, p in enumerate(protos):
        m = p.metrics[-1]
        print(f"{i:>5} {p.n:>9} {accs[i]:>10.4f} {m['hinge_loss']:>10.4f} "
              f"{p.comm_bytes / 1e6:>9.3f}")
    print("-" * 48)
    print(f"mean accuracy      {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
    print(f"consensus error    {consensus:.6f}   (mean ||w_i - mean(w)||)")
    print(f"overlay components {n_comp}" + ("   <-- DISCONNECTED" if n_comp > 1 else ""))
    print(f"total sent         {total_mb:.2f} MB over {args.cycles} cycles")

    if args.csv:
        path = Path(args.csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        new = not path.exists()
        with open(path, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=[
                "tag", "method", "topology", "scheme", "nodes", "cycles",
                "gossip_k", "lambda_reg", "seed", "node", "n_local",
                "test_acc", "hinge_loss", "duality_gap", "consensus_err",
                "comm_bytes", "components",
            ])
            if new:
                w.writeheader()
            for i, p_ in enumerate(protos):
                m = p_.metrics[-1] if p_.metrics else {}
                w.writerow(dict(
                    tag=args.tag, method=args.method, topology=args.topology,
                    scheme=args.scheme, nodes=args.nodes, cycles=args.cycles,
                    gossip_k=args.gossip_k, lambda_reg=args.lambda_reg,
                    seed=args.seed, node=i, n_local=p_.n, test_acc=accs[i],
                    hinge_loss=m.get("hinge_loss", float("nan")),
                    duality_gap=m.get("duality_gap", float("nan")),
                    consensus_err=consensus, comm_bytes=p_.comm_bytes,
                    components=n_comp,
                ))
        print(f"appended to        {path}")


if __name__ == "__main__":
    main()
