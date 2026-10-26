"""MoM-NDSVM — the nonlinear consensus-based distributed SVM baseline.

P. A. Forero, A. Cano, G. B. Giannakis, "Consensus-Based Distributed Support
Vector Machines", JMLR 11 (2010) 1663-1707, Section 4, Algorithm 3.

Nodes cannot exchange ``w`` (it lives in the RBF feature space), so they agree on
the discriminant's values at L common anchor points ``Gamma = {chi_l}``: problem
(23) adds the consensus constraints ``G w_j = G w_i`` and ``b_j = b_i`` with
``G = [phi(chi_1) .. phi(chi_L)]^T``. ADMM ("ADMoM") on (23) gives Algorithm 3.
Each round, each node solves a local box-constrained **SVM dual QP** (35) over its
own multipliers ``lambda_j`` (one per training point; ``a_j = Y_j lambda_j`` are its
support-vector coefficients). It then broadcasts the ``L+1`` numbers
``(w~_j = G w_j, b_j)`` to every neighbour and updates its consensus prices
``(alpha_j, beta_j)``. Its classifier is
``g_j(x) = sum_n a_jn K(x, x_jn) + sum_l c_jl K(x, chi_l) + b_j`` (31).
For finite ``L`` the fixed point is the solution of (23), which agrees with the
centralised SVM (2) at the anchors, not everywhere (paper Sec. 4.1).

Mapping onto the shared PeerSim engine (one engine cycle = one iteration ``t``):

    Algorithm 3                      here
    -------------------------------  ---------------------------------------------
    Require: Gamma, JC, eta, K agreed NDSVMDataInitializer: Sec. 4.1 random design
                                      (max/min consensus over diameter(G) rounds +
                                      shared seed), JC = J / (LAMBDA * n_global)
    random w~_j(0), b_j(0)            initializer + initial broadcast to neighbours
    lines 2-6   (35), (36), (37)      NDSVMProtocol.nextCycle (phase A, pending only)
    line 8      broadcast             SynchronousRoundBarrier -> deliver / commit
    lines 11-12 (38), (39)            SynchronousRoundBarrier -> finish_round
    line 13     readout (32)-(34)     taken in phase A (E2 pairing)

Errata. These are typesetting errors in the paper. The code follows the paper's
derivation (node j's ADMoM subproblem and Appendices G/H), not the printed
formulas:

E1  (35) and (82). The term ``- ( f~^T (K(G,X) - K~(G,X)) + h 1^T / (2 eta |B_j|) ) Y lambda``
    must enter with a PLUS sign. Writing node j's subproblem as
    ``1/2||w||^2 + eta|B_j| ||Gw||^2 + f~^T Gw + eta|B_j| b^2 + h b + JC 1^T xi`` gives
    the minimisers ``w(lambda) = U^{-1}(Phi^T Y lambda - G^T f~)`` (= (78)) and
    ``b(lambda) = (1^T Y lambda - h) / (2 eta |B_j|)`` (= (37)). KKT then requires
    ``dD/dlambda_n = 1 - y_n (phi_n^T w + b)``. That holds only with ``+``, which is also
    the sign in the paper's own linear-case (16). With the printed sign, lambda is not
    the multiplier of the iterate (36)/(37) builds from it. ``validate_ndsvm.py``
    measures a local duality gap of ~5e2 for the printed sign versus ~1e-10 for the
    corrected one. The printed variant also fails to reach the centralised SVM in
    the linear-kernel oracle.
E2  (32)-(34) and (81). The discriminant of the iterate ``(w_j(t+1), b_j(t+1))``
    pairs ``lambda_j(t+1)`` with ``f~_j(t)`` and ``h_j(t)``, as (78) shows. The printed
    ``lambda_j(t) / f~_j(t)`` pairing, and the ``f~_j(t+1)`` in (81), are index slips.
    They only matter during the transient: at the fixed point ``f~(t+1) = f~(t)``.
    Only the correct pairing satisfies ``K(G,X) a + K(G,G) c == w~_j(t+1)`` of (36).
E3  (79). The sum inside ``b_j(t+1)`` runs over the biases ``b_j + b_i``, not
    ``beta_j + beta_i``. This is main-text (37) with ``h_j(t)``.

Same distributed setup as the other tenants: same shards, test set, graph (same
seeded WireKOut, checked identical), engine, budget and metric definitions. The
regularised problem is also the same: ``C = 1/(LAMBDA n_global)`` makes (2)/(C n)
equal ``(lambda/2)||w||^2 + (1/n) sum hinge``. The kernel is also the same RBF with
``gamma = RBF_GAMMA``. Intrinsic to the algorithm, and therefore part of what is
measured: synchronous broadcast to *all* neighbours, an ``(L+1)``-float payload,
the bias ``b``, raw features (``NDSVM_FEATURES="dsrr"`` switches to the kernel
tenant's projection), and stopping the whole network at once.

Modules: ``core`` (engine-free math), ``qp`` (exact solver for (35)), ``anchors``
(Sec. 4.1), ``ndsvm_protocol``, ``observers``, ``simulation``, ``validate_ndsvm``.
Run with ``python src/scripts/main.py -m ndsvm -d <dataset>``.
"""
