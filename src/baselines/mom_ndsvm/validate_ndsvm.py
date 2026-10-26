"""Validator for the MoM-NDSVM baseline (Forero, Cano & Giannakis, JMLR 2010).

Standalone (the repo has no test framework): prints PASS/FAIL per check and exits
non-zero if any fails.

    python src/baselines/mom_ndsvm/validate_ndsvm.py            # all checks
    python src/baselines/mom_ndsvm/validate_ndsvm.py V3 V4      # selected checks
    python src/baselines/mom_ndsvm/validate_ndsvm.py report gis # paper Table-3 report

Checks
------
V1  box-QP solver for (35) vs scipy L-BFGS-B (objective + KKT; lambda is not unique)
V2  per-iteration invariants of Algorithm 3: anchor identity (E2), local primal-dual
    gap of every node's subproblem (E1 + E2), sum_j alpha_j = 0 and sum_j beta_j = 0,
    and the ||w||^2 identity against an explicit K(X,X)
V3  linear kernel + full-rank Gamma: (23) == (1) (paper Sec. 4.1), so the run must
    reach sklearn SVC(kernel="linear"); the printed-sign variant of (35) must not
V4  exact RBF oracles: Gamma containing every training point makes (23) == (2), so the
    run must reach SVC(kernel="rbf"); for L < N, (23) solved centrally with SLSQP
V5  identical setup: all four tenants wire the same graph from CONFIG; shards are
    deterministic
V7  the engine is Algorithm 3: shuffle vs ordered activation and the plain loop give
    bit-identical iterates; anchors and communication bytes are exactly accounted
V8  E2 in the transient: the readout built with f~(t) satisfies the anchor identity,
    the one built with f~(t+1) does not, and the two agree at the fixed point
report  paper Table 3: centralised SVC vs local-only SVCs vs the saved MoM-NDSVM run
"""

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from src.baselines.mom_ndsvm.core import LocalProblem, f_tilde, h_scalar, dual_ascent
from src.baselines.mom_ndsvm.qp import BoxQPSolver, kkt_violation
from src.nonlinear_sdca.kernel import rbf_gram


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def linear_kernel(A, B):
    return np.asarray(A, dtype=np.float64) @ np.asarray(B, dtype=np.float64).T


def rbf(gamma):
    return lambda A, B: rbf_gram(A, B, gamma)


def toy_data(J, n_per, p, seed, nonlinear=False):
    """J shards of a 2-class problem (+ a test set) with labels in {-1,+1}."""
    rng = np.random.default_rng(seed)
    def draw(n):
        X = rng.normal(size=(n, p))
        if nonlinear:
            score = X[:, 0] ** 2 + X[:, 1] ** 2 - 1.2 + 0.3 * rng.normal(size=n)
        else:
            score = X @ np.linspace(1.0, -0.5, p) + 0.4 * rng.normal(size=n)
        return X, np.where(score >= 0.0, 1.0, -1.0)
    shards = [draw(n_per) for _ in range(J)]
    return shards, draw(200)


RING_PLUS = {4: [[1, 3, 2], [0, 2], [1, 3, 0], [2, 0]],
             3: [[1, 2], [0, 2], [0, 1]]}


def run_algorithm3(shards, adj, Gamma, kernel, eta, JC, iters, seed=0,
                   sign=+1, qp_tol=1e-12, on_node=None, on_round=None, init=None):
    """Algorithm 3 in a plain synchronous loop (no engine).

    ``sign=-1`` reproduces (35) exactly as printed (E1 uncorrected), for the
    negative control. ``on_node(t, j, ctx)`` runs after node j's phase A,
    ``on_round(t, state)`` after the dual step. ``init`` optionally fixes each
    node's ``(w~_j(0), b_j(0))``.
    """
    J = len(shards)
    K_GG = kernel(Gamma, Gamma)
    probs = [LocalProblem(X, y, Gamma, K_GG, kernel, eta, len(adj[j]), JC)
             for j, (X, y) in enumerate(shards)]
    solvers = [BoxQPSolver(P.Q, JC, tol=qp_tol, max_passes=200000) for P in probs]
    L = Gamma.shape[0]
    if init is None:
        rng = np.random.default_rng(seed)
        init = [(rng.standard_normal(L), float(rng.standard_normal())) for _ in range(J)]
    w = [np.array(w0, dtype=np.float64) for w0, _ in init]
    b = [float(b0) for _, b0 in init]
    alpha = [np.zeros(L) for _ in range(J)]
    beta = [0.0] * J
    readout = [None] * J
    for t in range(iters):
        pending = []
        for j, (P, S) in enumerate(zip(probs, solvers)):
            f = f_tilde(alpha[j], w[j], [w[i] for i in adj[j]], eta)
            h = h_scalar(beta[j], b[j], [b[i] for i in adj[j]], eta)
            p = P.lin_term(f, h) if sign > 0 else 1.0 - P.y * (P.A_GX.T @ f + h / P.c0)
            lam = S.solve(p).copy()
            a, c, w_new, b_new = P.iterate(lam, f, h)
            readout[j] = (a, c, b_new)
            pending.append((w_new, b_new))
            if on_node is not None:
                on_node(t, j, dict(P=P, lam=lam, Qlam=S.Qlam.copy(), f=f, h=h,
                                   a=a, c=c, w_new=w_new, b_new=b_new))
        w = [pw for pw, _ in pending]                      # broadcast
        b = [pb for _, pb in pending]
        for j in range(J):                                 # (38), (39)
            alpha[j], beta[j] = dual_ascent(alpha[j], beta[j], w[j], b[j],
                                            [w[i] for i in adj[j]],
                                            [b[i] for i in adj[j]], eta)
        if on_round is not None:
            on_round(t, dict(w=w, b=b, alpha=alpha, beta=beta, probs=probs,
                             readout=readout))
    return dict(probs=probs, readout=readout, w=w, b=b, alpha=alpha, beta=beta)


def decision(kernel, X_j, Gamma, readout_j, Z):
    a, c, b = readout_j
    return kernel(Z, X_j) @ a + kernel(Z, Gamma) @ c + b


def svm_objective(norm_sq, scores, y, C):
    return 0.5 * norm_sq + C * np.maximum(0.0, 1.0 - y * scores).sum()


RESULTS = []


def report(name, ok, msg):
    RESULTS.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {msg}", flush=True)


# --------------------------------------------------------------------------- #
# V1
# --------------------------------------------------------------------------- #
def check_v1():
    rng = np.random.default_rng(1)
    worst_obj, worst_kkt = 0.0, 0.0
    for trial in range(6):
        n = 60
        B = rng.normal(size=(n, n // 2 if trial % 2 else n))      # odd trials: singular Q
        Q = B @ B.T + 0.05 * np.eye(n)
        p = rng.normal(size=n) * 3
        upper = 1.5
        S = BoxQPSolver(Q, upper, tol=1e-12, max_passes=100000)
        lam = S.solve(p).copy()
        f = lambda l: (0.5 * l @ Q @ l - p @ l, Q @ l - p)
        ref = minimize(f, np.zeros(n), jac=True, method="L-BFGS-B",
                       bounds=[(0, upper)] * n,
                       options=dict(ftol=1e-16, gtol=1e-12, maxiter=100000, maxcor=50))
        ours = f(lam)[0]
        worst_obj = max(worst_obj, ours - ref.fun)                 # ours must not be worse
        worst_kkt = max(worst_kkt, kkt_violation(lam, Q @ lam - p, upper).max())
        # warm start: perturb p, re-solve from the cache, compare with a cold solve
        p2 = p + 0.1 * rng.normal(size=n)
        warm = S.solve(p2).copy()
        cold = BoxQPSolver(Q, upper, tol=1e-12, max_passes=100000).solve(p2)
        worst_obj = max(worst_obj, abs(f(warm)[0] - f(cold)[0]))
    ok = worst_obj < 1e-8 and worst_kkt < 1e-10
    report("V1 QP solver", ok,
           f"max(obj_ours - obj_LBFGSB, warm/cold diff)={worst_obj:.2e}, max KKT viol={worst_kkt:.2e}")


# --------------------------------------------------------------------------- #
# V2
# --------------------------------------------------------------------------- #
def check_v2():
    gamma = 0.5
    k = rbf(gamma)
    shards, _ = toy_data(4, 25, 2, seed=2, nonlinear=True)
    rng = np.random.default_rng(3)
    X_all = np.vstack([X for X, _ in shards])
    Gamma = rng.uniform(X_all.min(0), X_all.max(0), size=(6, 2))
    stats = dict(anchor=0.0, gap=0.0, norm=0.0, cons_a=0.0, cons_b=0.0)
    Kxx = [k(X, X) for X, _ in shards]

    def on_node(t, j, ctx):
        P, a, c = ctx["P"], ctx["a"], ctx["c"]
        anchor = np.abs(P.K_XG.T @ a + P.K_GG @ c - ctx["w_new"]).max()
        gap = abs(P.local_gap(ctx["lam"], ctx["Qlam"], ctx["f"], ctx["h"]))
        direct = a @ Kxx[j] @ a + 2 * a @ (P.K_XG @ c) + c @ P.K_GG @ c
        norm = abs(P.norm_sq(ctx["lam"], ctx["Qlam"], a, c) - direct) / max(1.0, abs(direct))
        stats["anchor"] = max(stats["anchor"], anchor)
        stats["gap"] = max(stats["gap"], gap)
        stats["norm"] = max(stats["norm"], norm)

    def on_round(t, s):
        stats["cons_a"] = max(stats["cons_a"], np.abs(np.sum(s["alpha"], axis=0)).max())
        stats["cons_b"] = max(stats["cons_b"], abs(sum(s["beta"])))

    run_algorithm3(shards, RING_PLUS[4], Gamma, k, eta=1.0, JC=4.0, iters=60,
                   on_node=on_node, on_round=on_round)
    ok = (stats["anchor"] < 1e-10 and stats["gap"] < 1e-7 and stats["norm"] < 1e-9
          and stats["cons_a"] < 1e-9 and stats["cons_b"] < 1e-9)
    report("V2 invariants", ok,
           f"anchor id {stats['anchor']:.1e}, local gap {stats['gap']:.1e}, "
           f"||w||^2 id {stats['norm']:.1e}, |sum alpha| {stats['cons_a']:.1e}, "
           f"|sum beta| {stats['cons_b']:.1e}")
    # negative control: the printed sign leaves a large local gap
    worst = [0.0]

    def on_node_minus(t, j, ctx):
        worst[0] = max(worst[0], abs(ctx["P"].local_gap(ctx["lam"], ctx["Qlam"], ctx["f"], ctx["h"])))
    run_algorithm3(shards, RING_PLUS[4], Gamma, k, eta=1.0, JC=4.0, iters=5,
                   sign=-1, on_node=on_node_minus)
    report("V2 E1 negative control", worst[0] > 1e-3,
           f"printed sign of (35): local gap {worst[0]:.2e} (must be large)")


# --------------------------------------------------------------------------- #
# V3
# --------------------------------------------------------------------------- #
def converge_to_svc(shards, adj, Gamma, kernel, svc, C, J, eta, iters, sign=+1):
    out = run_algorithm3(shards, adj, Gamma, kernel, eta=eta, JC=J * C,
                         iters=iters, sign=sign)
    X_all = np.vstack([X for X, _ in shards])
    y_all = np.concatenate([y for _, y in shards])
    ref_scores = svc.decision_function(X_all)
    if svc.kernel == "linear":
        ref_norm = float(svc.coef_.ravel() @ svc.coef_.ravel())
    else:
        dc = svc.dual_coef_.ravel()
        ref_norm = float(dc @ kernel(svc.support_vectors_, svc.support_vectors_) @ dc)
    ref_obj = svm_objective(ref_norm, ref_scores, y_all, C)
    worst_dec, worst_obj = 0.0, 0.0
    for j, (X, _) in enumerate(shards):
        P = out["probs"][j]
        a, c, b = out["readout"][j]
        scores = decision(kernel, X, Gamma, out["readout"][j], X_all)
        nrm = a @ kernel(X, X) @ a + 2 * a @ (P.K_XG @ c) + c @ P.K_GG @ c
        obj = svm_objective(nrm, scores, y_all, C)
        worst_dec = max(worst_dec, np.abs(scores - ref_scores).max())
        worst_obj = max(worst_obj, abs(obj - ref_obj) / abs(ref_obj))
    return worst_dec, worst_obj, ref_obj


def check_v3():
    J, C = 4, 0.5
    shards, _ = toy_data(J, 30, 5, seed=4)
    X_all = np.vstack([X for X, _ in shards])
    y_all = np.concatenate([y for _, y in shards])
    rng = np.random.default_rng(5)
    Gamma = rng.uniform(X_all.min(0), X_all.max(0), size=(8, 5))   # L=8 >= p=5, full rank
    svc = SVC(kernel="linear", C=C, tol=1e-12).fit(X_all, y_all)
    dec, obj, ref = converge_to_svc(shards, RING_PLUS[4], Gamma, linear_kernel, svc, C, J,
                                    eta=0.5, iters=1500)
    report("V3 linear-kernel oracle", dec < 1e-4 and obj < 1e-6,
           f"max |g_j - g_SVC| = {dec:.2e}, max rel objective diff = {obj:.2e} (SVC obj {ref:.4f})")
    dec_m, obj_m, _ = converge_to_svc(shards, RING_PLUS[4], Gamma, linear_kernel, svc, C, J,
                                      eta=0.5, iters=1500, sign=-1)
    report("V3 E1 negative control", dec_m > 1e-2,
           f"printed sign: max |g_j - g_SVC| = {dec_m:.2e}, rel objective diff = {obj_m:.2e} (must not converge)")


# --------------------------------------------------------------------------- #
# V4
# --------------------------------------------------------------------------- #
def check_v4():
    gamma, J, C = 0.7, 3, 1.0
    k = rbf(gamma)
    shards, _ = toy_data(J, 10, 2, seed=6, nonlinear=True)
    X_all = np.vstack([X for X, _ in shards])
    y_all = np.concatenate([y for _, y in shards])
    svc = SVC(kernel="rbf", gamma=gamma, C=C, tol=1e-12).fit(X_all, y_all)
    # Gamma = all training points makes K(G,G) ill-conditioned (cond ~7e5), so ADMM's
    # tail is slow. The fixed point does not depend on eta (Prop. 3: any eta > 0), so
    # this oracle uses eta=2 and enough iterations rather than a looser tolerance.
    dec, obj, ref = converge_to_svc(shards, RING_PLUS[3], X_all.copy(), k, svc, C, J,
                                    eta=2.0, iters=6000)
    report("V4a RBF oracle (Gamma = all training points)", dec < 1e-4 and obj < 1e-6,
           f"max |g_j - g_SVC| = {dec:.2e}, rel objective diff = {obj:.2e} (SVC obj {ref:.4f})")

    # L < N: solve (23) centrally in representer variables, compare node by node.
    shards, (X_te, _) = toy_data(J, 8, 2, seed=7, nonlinear=True)
    rng = np.random.default_rng(8)
    X_all = np.vstack([X for X, _ in shards])
    Gamma = rng.uniform(X_all.min(0), X_all.max(0), size=(4, 2))
    oracle = solve_23_centrally(shards, Gamma, k, J * C)
    out = run_algorithm3(shards, RING_PLUS[3], Gamma, k, eta=0.5, JC=J * C, iters=4000)
    worst = 0.0
    for j, (X, _) in enumerate(shards):
        ours = decision(k, X, Gamma, out["readout"][j], X_te)
        theirs = decision(k, X, Gamma, oracle[j], X_te)
        worst = max(worst, np.abs(ours - theirs).max())
    report("V4b exact (23) oracle, L < N (SLSQP)", worst < 1e-6,
           f"max |g_j - g_j^(23)| over test points = {worst:.2e}")


def solve_23_centrally(shards, Gamma, kernel, JC):
    """Problem (23) in representer variables (Theorem 4): per node a_j, c_j, a common
    b (b_j = b_i on a connected graph), slacks xi; consensus K(G,X_j)a_j + K(G,G)c_j
    equal across nodes."""
    J, L = len(shards), Gamma.shape[0]
    Ns = [len(y) for _, y in shards]
    K_GG = kernel(Gamma, Gamma)
    Kjj = [kernel(X, X) for X, _ in shards]
    KjG = [kernel(X, Gamma) for X, _ in shards]
    na = sum(Ns)
    off_a = np.cumsum([0] + Ns)
    off_c = na + L * np.arange(J + 1)
    i_b = na + L * J
    off_x = i_b + 1
    nv = off_x + na

    def unpack(v):
        return ([v[off_a[j]:off_a[j + 1]] for j in range(J)],
                [v[off_c[j]:off_c[j + 1]] for j in range(J)], v[i_b], v[off_x:])

    def obj(v):
        a, c, _, xi = unpack(v)
        val, grad = JC * xi.sum(), np.zeros(nv)
        grad[off_x:] = JC
        for j in range(J):
            Ka, Gc = Kjj[j] @ a[j] + KjG[j] @ c[j], KjG[j].T @ a[j] + K_GG @ c[j]
            val += 0.5 * (a[j] @ Ka + c[j] @ Gc)
            grad[off_a[j]:off_a[j + 1]] = Ka
            grad[off_c[j]:off_c[j + 1]] = Gc
        return val, grad

    A_ineq, A_eq = [], []
    for j, (X, y) in enumerate(shards):
        for n in range(Ns[j]):
            row = np.zeros(nv)
            row[off_a[j]:off_a[j + 1]] = y[n] * Kjj[j][n]
            row[off_c[j]:off_c[j + 1]] = y[n] * KjG[j][n]
            row[i_b] = y[n]
            row[off_x + off_a[j] + n] = 1.0
            A_ineq.append(row)
    for j in range(1, J):
        for l in range(L):
            row = np.zeros(nv)
            row[off_a[j]:off_a[j + 1]] = KjG[j][:, l]
            row[off_c[j]:off_c[j + 1]] = K_GG[l]
            row[off_a[0]:off_a[1]] -= KjG[0][:, l]
            row[off_c[0]:off_c[1]] -= K_GG[l]
            A_eq.append(row)
    A_ineq, A_eq = np.array(A_ineq), np.array(A_eq)
    cons = [dict(type="ineq", fun=lambda v: A_ineq @ v - 1.0, jac=lambda v: A_ineq),
            dict(type="eq", fun=lambda v: A_eq @ v, jac=lambda v: A_eq)]
    bounds = [(None, None)] * off_x + [(0.0, None)] * na
    res = minimize(obj, np.zeros(nv), jac=True, method="SLSQP", constraints=cons,
                   bounds=bounds, options=dict(ftol=1e-15, maxiter=5000))
    a, c, b, _ = unpack(res.x)
    return [(a[j], c[j], float(b)) for j in range(J)]


# --------------------------------------------------------------------------- #
# V8
# --------------------------------------------------------------------------- #
def check_v8():
    gamma = 0.5
    k = rbf(gamma)
    shards, _ = toy_data(4, 20, 2, seed=9, nonlinear=True)
    rng = np.random.default_rng(10)
    X_all = np.vstack([X for X, _ in shards])
    Gamma = rng.uniform(X_all.min(0), X_all.max(0), size=(5, 2))
    adj, eta = RING_PLUS[4], 1.0
    early, late = {}, {}

    def on_round(t, s):
        if t not in (0, 2, 2999):
            return
        worst_alt, worst_ok, worst_diff = 0.0, 0.0, 0.0
        for j, P in enumerate(s["probs"]):
            a, c, _ = s["readout"][j]
            f_next = f_tilde(s["alpha"][j], s["w"][j], [s["w"][i] for i in adj[j]], eta)
            c_alt = P.c0 * np.linalg.solve(np.eye(P.L) + P.c0 * P.K_GG,
                                           P.K_GG @ f_next - P.K_XG.T @ a) - f_next
            w_t1 = s["w"][j]
            worst_ok = max(worst_ok, np.abs(P.K_XG.T @ a + P.K_GG @ c - w_t1).max())
            worst_alt = max(worst_alt, np.abs(P.K_XG.T @ a + P.K_GG @ c_alt - w_t1).max())
            worst_diff = max(worst_diff, np.abs(c_alt - c).max())
        (early if t < 2999 else late)[t] = (worst_ok, worst_alt, worst_diff)

    run_algorithm3(shards, adj, Gamma, k, eta=eta, JC=4.0, iters=3000, on_round=on_round)
    e_ok = max(v[0] for v in early.values())
    e_alt = min(v[1] for v in early.values())
    l_diff = late[2999][2]
    ok = e_ok < 1e-10 and e_alt > 1e-3 and l_diff < 1e-6
    report("V8 E2 transient", ok,
           f"early: anchor id with f~(t) {e_ok:.1e}, with f~(t+1) {e_alt:.1e}; "
           f"at t=3000 the two readouts differ by {l_diff:.1e}")


# --------------------------------------------------------------------------- #
# V5 / V7 — the engine
# --------------------------------------------------------------------------- #
def _engine_run(activation, iters, seed=11):
    """Run NDSVMSimulation on a small synthetic network; return its protocols."""
    import scipy.sparse as sp
    from src.config import CONFIG
    from src.peersim_python.core import Network
    from src.baselines.mom_ndsvm.simulation import NDSVMSimulation
    J = 6
    shards, (X_te, y_te) = toy_data(J, 20, 3, seed=12, nonlinear=True)
    worker_data = [dict(X_csr=sp.csr_matrix(X), y=y, X_test=sp.csr_matrix(X_te),
                        y_test=y_te, n_local=len(y)) for X, y in shards]
    cfg = dict(CONFIG)
    cfg.update(NUM_WORKERS=J, TOPOLOGY="random_kout", GOSSIP_K=2, SEED=seed,
               LAMBDA=0.01, RBF_GAMMA=0.5, NDSVM_L=7, NDSVM_ETA=1.0,
               NDSVM_QP_TOL=1e-12, NDSVM_QP_MAX_SWEEPS=200000, NDSVM_FEATURES="raw",
               ACTIVATION=activation, EVAL_EVERY=1,
               STOP_ON_THRESHOLD=False, STOP_ON_PLATEAU=False)
    NDSVMSimulation(cfg, worker_data).run(iters)
    protos = [Network.get(i).getProtocol(NDSVMSimulation.NDSVM_PID) for i in range(J)]
    return protos, shards, cfg


def check_v7():
    iters = 40
    shuf, shards, cfg = _engine_run("shuffle", iters)
    snap = [(p.a.copy(), p.c.copy(), p.b_read, p.w_tilde.copy(), p.alpha.copy(), p.beta)
            for p in shuf]
    adj = [list(p.neighbours) for p in shuf]
    Gamma = shuf[0].Gamma.copy()
    comm = [p.comm_bytes for p in shuf]
    ordered, _, _ = _engine_run("ordered", iters)
    diff_order = max(max(np.abs(s[0] - p.a).max(), np.abs(s[1] - p.c).max(),
                         abs(s[2] - p.b_read), np.abs(s[3] - p.w_tilde).max(),
                         np.abs(s[4] - p.alpha).max(), abs(s[5] - p.beta))
                     for s, p in zip(snap, ordered))
    report("V7 activation order irrelevant", diff_order == 0.0,
           f"shuffle vs ordered: max |difference| = {diff_order:.1e} (must be exactly 0)")

    # The engine must be Algorithm 3: replay it in the plain loop from the same start.
    J, L, gamma = len(shards), Gamma.shape[0], cfg["RBF_GAMMA"]
    n_global = sum(len(y) for _, y in shards)
    JC = J / (cfg["LAMBDA"] * n_global)
    init = []
    for j in range(J):
        rng = np.random.default_rng([cfg["SEED"], 1, j])
        init.append((rng.standard_normal(L), float(rng.standard_normal())))
    out = run_algorithm3(shards, adj, Gamma, rbf(gamma), eta=cfg["NDSVM_ETA"], JC=JC,
                         iters=iters, qp_tol=cfg["NDSVM_QP_TOL"], init=init)
    diff_plain = max(max(np.abs(out["readout"][j][0] - snap[j][0]).max(),
                         np.abs(out["readout"][j][1] - snap[j][1]).max(),
                         abs(out["readout"][j][2] - snap[j][2]),
                         np.abs(out["alpha"][j] - snap[j][4]).max())
                     for j in range(J))
    report("V7 engine == Algorithm 3", diff_plain == 0.0,
           f"PeerSim run vs plain synchronous loop: max |difference| = {diff_plain:.1e}")

    # Anchors: max/min consensus + shared seed reproduce the global box draw.
    from src.baselines.mom_ndsvm.anchors import draw_anchors, check_graph
    X_all = np.vstack([X for X, _ in shards])
    ref = draw_anchors(X_all.min(0), X_all.max(0), L, cfg["SEED"])
    diam = check_graph(adj)
    p_dim = X_all.shape[1]
    expected = [diam * len(adj[j]) * 2 * p_dim * 8 + (iters + 1) * len(adj[j]) * (L + 1) * 8
                for j in range(J)]
    ok = np.array_equal(ref, Gamma) and comm == expected
    report("V7 anchors + communication accounting", ok,
           f"Gamma == global-box draw: {np.array_equal(ref, Gamma)}; comm_bytes == "
           f"flood(diameter={diam}) + (1 init + {iters}) broadcasts of (L+1) floats: {comm == expected}")


def check_v5():
    """All four tenants wire the identical graph from the real CONFIG, and the shard
    loader is deterministic."""
    from src.config import CONFIG
    from src.peersim_python.core import Network
    from src.linear_sdca.simulation import Simulation
    from src.baselines.gadget.simulation import GadgetSimulation
    from src.nonlinear_sdca.simulation import KernelSDCASimulation
    from src.baselines.mom_ndsvm.simulation import NDSVMSimulation
    graphs = {}
    for name, cls in (("sdca", Simulation), ("gadget", GadgetSimulation),
                      ("nonlinear", KernelSDCASimulation), ("ndsvm", NDSVMSimulation)):
        sim = cls(dict(CONFIG), []).build()
        sim.sim.initializers[0].execute()                       # the topology control
        graphs[name] = [sorted(peer.getIndex() for peer in Network.get(i).getProtocol(0).neighbors)
                        for i in range(Network.size())]
    same = all(g == graphs["sdca"] for g in graphs.values())
    degrees = [len(n) for n in graphs["ndsvm"]]
    report("V5 identical graph", same,
           f"sdca/gadget/nonlinear/ndsvm adjacency identical: {same}; degrees {degrees}")

    from src.scripts.data_sharding import load_shards
    cfg = dict(CONFIG, DATASET="a9a")
    if Path(cfg["A9A_TRAIN_PATH"]).exists():
        s1, s2 = load_shards(cfg), load_shards(cfg)
        same_shards = all((a["X_csr"] != b["X_csr"]).nnz == 0 and np.array_equal(a["y"], b["y"])
                          for a, b in zip(s1, s2))
        report("V5 deterministic shards", same_shards,
               f"two load_shards(a9a) calls give identical shards: {same_shards}")

# --------------------------------------------------------------------------- #
# V6 report — paper Table 3: centralised vs local-only vs MoM-NDSVM
# --------------------------------------------------------------------------- #
def table3_report(keyword):
    """Same shards, same C, same RBF gamma: a centralised SVC on all training data, an
    SVC on each node's shard alone, and the MoM-NDSVM run saved by
    ``src/scripts/main.py -m ndsvm``. Objectives are normalised as in the evaluator:
    ``(lambda/2)||w||^2 + mean hinge`` over all training data."""
    import json
    from src.config import CONFIG, CODE_DIR
    from src.scripts.dataset_keywords import select_dataset
    from src.scripts.data_sharding import load_shards
    select_dataset(keyword)
    shards = load_shards(CONFIG)
    X = [s["X_csr"].toarray() for s in shards]
    y = [np.asarray(s["y"], dtype=np.float64) for s in shards]
    X_te = shards[0]["X_test"].toarray()
    y_te = np.asarray(shards[0]["y_test"], dtype=np.float64)
    X_all, y_all = np.vstack(X), np.concatenate(y)
    lam, gamma = CONFIG["LAMBDA"], CONFIG["RBF_GAMMA"]
    C = 1.0 / (lam * len(y_all))

    def normalised_objective(svc):
        dc = svc.dual_coef_.ravel()
        sv = svc.support_vectors_
        norm_sq = float(dc @ rbf_gram(sv, sv, gamma) @ dc)
        hinge = np.maximum(0.0, 1.0 - y_all * svc.decision_function(X_all)).mean()
        return 0.5 * lam * norm_sq + hinge

    central = SVC(kernel="rbf", gamma=gamma, C=C, tol=1e-6).fit(X_all, y_all)
    acc_central = float(np.mean(central.predict(X_te) == y_te))
    obj_central = normalised_objective(central)
    acc_local = [float(np.mean(SVC(kernel="rbf", gamma=gamma, C=C, tol=1e-6)
                               .fit(Xj, yj).predict(X_te) == y_te)) for Xj, yj in zip(X, y)]
    path = CODE_DIR / "results" / "json" / f"ndsvm_{CONFIG['DATASET']}.json"
    print(f"\nTable-3-style report — {CONFIG['DATASET']} (C={C:.4g}, gamma={gamma}, "
          f"{len(shards)} nodes)")
    print(f"  centralised SVC (rbf):      accuracy {acc_central:.4f}   objective {obj_central:.5f}")
    print(f"  local-only SVCs (mean):     accuracy {np.mean(acc_local):.4f}   "
          f"(min {min(acc_local):.4f}, max {max(acc_local):.4f})")
    if path.exists():
        data = json.loads(path.read_text())
        last = [m[-1] for m in data["metrics"] if m]
        print(f"  MoM-NDSVM (L={data.get('ndsvm_L')}): accuracy {data['avg_accuracy']:.4f}   "
              f"objective {np.mean([r['objective'] for r in last]):.5f}   "
              f"(round {last[0]['round']}, max primal residual "
              f"{max(r['primal_residual'] for r in last):.2e})")
    else:
        print(f"  MoM-NDSVM: no {path.name} yet — run src/scripts/main.py -m ndsvm -d {keyword}")

CHECKS = {"V1": check_v1, "V2": check_v2, "V3": check_v3, "V4": check_v4, "V8": check_v8,
"V5": check_v5,
"V7": check_v7}


def main():
    args = sys.argv[1:]
    if args and args[0].lower() == "report":
        table3_report(args[1] if len(args) > 1 else "gis")
        return
    wanted = [a.upper() for a in args] or list(CHECKS)
    for name in wanted:
        if name not in CHECKS:
            raise SystemExit(f"Unknown check {name}. Choose from {list(CHECKS)} or 'report <dataset>'")
        CHECKS[name]()
    n_fail = RESULTS.count(False)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
