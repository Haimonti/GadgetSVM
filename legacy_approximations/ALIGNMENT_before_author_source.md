# FDR-SVM and FedSSL-AMC alignment audit — 2026-10-20

## 当前结论

- FDR 已恢复论文的鲁棒约束、精确凸子问题、全局聚合和对偶更新；原二次代理已停用。
- SSL 已恢复 I/Q CNN、对比 triplet loss、共享 encoder 和客户端独立 SVM；a9a 适配暂缓。
- 第一轮数值检查通过：FDR 与相同目标的集中式解一致；SSL I/Q 流程跑通；两者精确 P2P 聚合与服务器聚合一致。
- SSL 的层/块数量解释、裁剪采样及 SVM 超参数尚不能由论文完全确定，已逐项记录实现选择，**不能称为完整论文复现**。
- 旧图中的这两项不能继续作为论文算法的比较结果。新参考版使用固定轮数，不把预算结束标记为收敛。

PeerSim 其余方法的后续核对见 [P2P_AUDIT.md](P2P_AUDIT.md)：CoCoA/CoCoA+ 本地更新已数值核对；FedAvg 的拓扑加权及 BDSVM 的公共核参数仍有独立问题待处理。不能把此前的流程测试通过理解为所有算法已完整对齐论文。

## Status and source versions

The previous quadratic FDR and tabular SVD FedSSL results are **retired as paper comparisons**. They are preserved for audit, not reused as evidence for the corrected algorithms. `plot_convergence_compare.load_results` excludes those old entries. The a9a SSL adaptation is deferred at the user's request until I/Q verification.

Sources:

- FDR paper: https://arxiv.org/html/2410.03877v3 (Eq. 9, Algorithm 2, Theorem 3, Appendix C).
- Official FDR repository: https://github.com/mibrahim41/FDR-SVM/tree/6e44afc882575258175fca380b2c074188e2a419 . `fdr_author.py` is a read-only source snapshot, with `FDR_LICENSE` (MIT). It is not executed.
- FedSSL-AMC paper: https://arxiv.org/html/2510.04927v2 (Eq. 1, Algorithm 1, IV-A).
- Encoder cited by FedSSL-AMC: https://github.com/White-Link/UnsupervisedScalableRepresentationLearningTimeSeries/tree/4aff59229113368ace606ad76e149a7916daa6f9 . `ssl_backbone.py` is an unmodified pinned snapshot; retain `BACKBONE_LICENSE`, `BACKBONE_NOTICE`, and the embedded TCN MIT notice.

## FDR implementation mapping

| Paper / author implementation | Corrected code |
| --- | --- |
| epsilon * lambda + mean(s), s >= 0 | `fdr.robust_terms` |
| Both observed-label and flipped-label constraints | `fdr.robust_terms` |
| lambda >= dual norm(w) | `fdr.norm_epigraph` |
| rho/2 * squared norm(w - global + mu) | `fdr.LocalProblem` |
| ADMM-SC adds tau * squared norm(w) | `fdr.LocalProblem`; Theorem 3 parameter check |
| Solve local convex subproblems | CVXPY / Clarabel, checked optimal status and feasibility; replaces author's Gurobi solver |
| Global=sum alpha_g*(local_g+mu_g), then dual += local-global | `fdr.train`, after all local solves |
| UCI equal client weights; epsilon=1/(10 N_g); kappa=1 | Explicit defaults |
| UCI global initial weights zero; scaled duals ones | Explicit defaults |
| UCI author pnorm=1 | Named **dual_norm=1**, to avoid confusing ground metric and dual norm |
| Fixed number of rounds T | `paper_fixed_round_budget`; **no convergence claim** |

There is no existing lambda_reg+epsilon quadratic shortcut in the new reference. ADMM and ADMM-SC are separate objectives: tau=0 versus a nonzero tau satisfying the theorem's bound. A generic objective plateau is not the original paper's ADMM stopping rule. Residuals are logged as diagnostics, without relabeling a fixed budget as convergence.

The direct centralized validation solves the **same robust objective**, not the ordinary L2 SVM used in the older comparison. Hyperparameter selection and a published UCI experiment reproduction remain separate from algorithm verification.

## SSL implementation mapping

| Paper | Corrected code |
| --- | --- |
| Two I/Q channels, time axis | `(N,2,T)` input contract; tabular arrays rejected |
| Causal, exponentially dilated, weight-normalized CNN | Pinned cited backbone; causal prefix test |
| Logistic triplet objective Eq. 1 | `ssl.contrastive_loss`: one positive term plus **sum**, not mean, over negatives |
| Positive subsequence of reference; negatives from different sequences | `sample_triplet`, no labels accepted; exclude reference ID from every negative |
| Dataset-size weighted FedAvg of encoder | `train_encoder`, with identical downloaded state each round |
| Freeze final encoder; each client trains own SVM | `fit_local_heads`; no classifier aggregation or communication |
| 10 rounds, 15,000 local steps, batch 50, Adam 0.001, 10 negatives, embedding 320, kernel 3 | `SSLConfig` defaults |
| Synchronous experimental Algorithm 1 | Implemented without proximal term; paper's proximal-variant theorem is not claimed for this implementation |

### Remaining underspecified SSL details

The paper calls the encoder “10-layer”; the cited upstream default `depth=10` creates 11 residual blocks / 22 convolutions. Upstream hidden width 40, reduced width 160, output 320 gives **247,880 parameters**, consistent with the paper's approximate 0.247M parameter count. This supports that choice but does not prove the AMC authors used identical code. We report upstream depth semantics explicitly rather than silently changing the block count.

The paper does not fully determine hidden/pooling widths, crop-length distribution, Adam state handling between rounds, or local SVM kernel/C. Current reference uses upstream widths/max pooling, uniform positive crop lengths, full sequence references, per-round Adam reset, and explicit configurable RBF SVC C=1. These are recorded choices, not verified original hyperparameters. Kernel/C must be selected using training/validation data if tuned; never the test set.

Consequently the SSL status is **algorithm reference with unresolved experimental details**, not a complete reproduction of the published accuracy. No claim of 66.69% reproduction is made.

## I/Q data and verification

`iq_data.py` generates independent synthetic unlabeled/labeled/test splits using Eq. 34–35: four modulation classes, sequence length 100, clipped Rayleigh gain, random phase, CFO 0.01, and SNR uniform -10 to 10 dB. The full profile has the paper's four class-count distributions, 140k unlabeled / 2.8k labeled / 280 test sequences per client. Independent unit-power constellation symbols and no pulse shaping are explicit assumptions.

`run_paper_reference.py ssl --profile smoke` uses the full upstream backbone, but reduces data and local computation. It verifies training and personalization on I/Q; it does **not** establish paper accuracy or convergence. The `paper` profile restores reported data/training budgets, but still reports unresolved details and does not claim exact reproduction.

Remote verification is in `verification/integrated/verification/checks.json` after extracting the returned `validation.zip`. The first check is also retained under `verification/run1`. It checks an analytic local optimum, finite label-flip behavior, ADMM versus a direct robust optimum, the SC bound, disconnected graph rejection, actual link payload accounting, causal prefix invariance, triplet normalization, negative exclusion, frozen encoder and independent heads, and server/P2P equality.

## P2P rewrite and communication meaning

`transport.ExactAggregation` implements synchronous delta flooding with an origin table, resetting each epoch. Every client must receive all current contributions before an algorithm update. This preserves the paper's global weighted update; replacing it with a single neighbor average would change the algorithm.

This is a **P2P communication adaptation**, not a claim that either original paper proposed this topology or proved asynchronous gossip convergence. It requires a strongly connected reliable graph and global barriers. The current implementation is a deterministic link-level simulator, separate from the old PeerSim assemblies. It is not a multi-machine deployment.

Algorithm rounds and link rounds are distinct. P2P counts every transmitted contribution on every directed edge, including configurable 16-byte record headers and 8-byte weights. Server counts each client upload and each per-client encoder/model download. Final SSL encoder downloads are included; SVM heads stay local. Transport bytes describe this specified protocol and must not be mixed with the old approximation's counters. Offline diagnostics are excluded.

## Entry points and retired code

- `methods/fdr_svm.run` now calls the exact convex reference.
- `methods/fedssl_amc.run_iq_clients` exposes the I/Q pipeline; the old tabular `run` raises an explanatory error.
- `run_paper_reference.py`: runnable server/ring references; FDR accepts separate LIBSVM train/test files.
- `run_convergence_compare.py`: the previous six ordinary baseline entries remain eligible; old FDR/SSL are removed pending integration of a declared common stopping/measurement protocol.
- `src/fdr_svm_simulation.py`: obsolete approximation assembly is disabled with a clear migration message.
- `legacy_approximations/`: historical server implementations and assembly retained solely for inspection. Old P2P approximation modules are not paper references.

No a9a FedSSL results or new common convergence bars should be published from this audit. First establish the reference and declared adaptations; later integrate valid measured results using train-only stopping and final-only test accuracy.

## Re-run

Use the private Windows compute workflow; dependency versions used there are
recorded in `requirements.txt`. From a staged copy of the repository:

```text
python -m unittest discover -s tests -v
python -m paper_reference.verify
python run_paper_reference.py ssl --profile smoke --out results/iq_reference
python run_paper_reference.py fdr --transport ring --out results/fdr_toy_p2p
```

The complete paper-budget SSL profile is opt-in (`--profile paper`). It can be
expensive on a CPU and still does not resolve the missing original settings.

## Measured validation results

- Final integration suite: **15 tests passed**; actual FDR server/ring CLI histories match.
- FDR toy ADMM-SC: direct robust optimum weight `0.66666666667`; ADMM weight `0.66666666521`; final primal residual about `1.20e-9`.
- SSL full upstream backbone smoke: 10 rounds, 10 local steps per round, batch 8; four clients, each with 14,000 unlabeled / 280 labeled / 28 test signals. Initial/final training triplet loss: `244.45` / `8.32`.
- Final local test accuracies: `82.14%, 53.57%, 50.00%, 75.00%`; equal-client mean `65.18%`. The tiny test budget and reduced optimization budget make this a workflow validation, **not a reproduction or comparison to the paper's reported accuracy**.
- Total I/Q pipeline elapsed about 374 seconds on the Windows CPU environment. This includes data generation, training, heads and testing, and is not a speedup measurement.
- Machine-readable summary: `verification/SUMMARY.json`; full I/Q records and checkpoint: `verification/integrated/results/iq_smoke/`; final entrypoint checks: `verification/final/final/`.
