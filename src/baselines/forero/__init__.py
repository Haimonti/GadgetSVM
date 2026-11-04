"""Forero MoM-NDSVM — the nonlinear consensus-based distributed SVM baseline.

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

Run in the paper's OWN setup, not on our PeerSim engine (10/26 meeting: "used in
its own setup, not ported"): a standalone synchronous loop over J nodes
(`algorithm.ForeroRun`), the paper's network (random connected / ring / full,
`network.py`), its equal class-balanced or biased partition, broadcast
communication accounting, and its centralised + local-only SVM benchmarks. Only
the data comes from us: `load_shards` supplies the user's dataset and test set.

    Algorithm 3                      here (algorithm.py)
    -------------------------------  ---------------------------------------------
    Require: Gamma, JC, eta, K agreed ForeroRun.build: Sec. 4.1 random design
                                      (max/min consensus over diameter(G) rounds +
                                      shared seed), JC = J / (LAMBDA * n_global)
    random w~_j(0), b_j(0)            ForeroNode init + initial broadcast
    lines 2-6   (35), (36), (37)      ForeroNode.phase_a (every node, from round-t values)
    line 8      broadcast             ForeroRun._broadcast
    lines 11-12 (38), (39)            ForeroNode.phase_c
    line 13     readout (32)-(34)     taken in phase_a (E2 pairing)

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

What is shared with our models: the dataset and test set (via `load_shards`), the
regularised problem (``C = 1/(LAMBDA n_global)`` makes (2)/(C n) equal
``(lambda/2)||w||^2 + (1/n) sum hinge``), the kernel (`make_kernel`: RBF with
``gamma = RBF_GAMMA``, or polynomial) and the metric definitions. Everything about
the distributed setup is the paper's: network, partition, synchronous broadcast
of an ``(L+1)``-float message, the bias ``b``, raw features (``NDSVM_FEATURES="dsrr"``
switches to our projection), whole-network stopping. Metrics carry both the
paper's broadcast count (`comm_bytes`) and a point-to-point count (`comm_bytes_p2p`).

Modules: ``core`` (math), ``qp`` (exact solver for (35)), ``anchors`` (Sec. 4.1),
``network`` (paper's graph + partition), ``algorithm`` (the run), ``validate_ndsvm``.
Run with ``python src/scripts/main.py -m ndsvm -d <dataset>``.
"""
