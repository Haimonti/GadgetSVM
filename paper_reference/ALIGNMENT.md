# FDR-SVM / SSL 对齐与 PeerSim 状态 — 2026-10-20

## 最新进展

**已找到 SSL 作者发布代码，并完成实际 PeerSim 接入；BDSVM 已默认切换为 linear，RBF 公共核宽度也已修复。**

详见 [SSL 作者代码接入 PeerSim 与 linear BDSVM](SSL_PEERSIM.md)，其中包含来源提交、原论文与发布配置差异、移植调整、通信计量和验证结果。

此前关于“未找到 SSL 作者代码”“网络配置尚不能核实”“尚未接回 PeerSim”的结论已被最新进展取代。旧版说明保留在 `../legacy_approximations/ALIGNMENT_before_author_source.md`，仅供审计。

## a9a 表格适配

新增独立的神经表格适配版，详见 [SSL_TABULAR.md](SSL_TABULAR.md)。它使用 MLP、局部边际特征替换和固定训练监测，保留编码器聚合与冻结本地 SVM 流程；明确不称为原论文复现。入口 `run_ssl_tabular.py`，协议改为与其他算法共用的 one-hop GossipProtocol；旧 flooding 结果已被取代。原论文为中心服务器聚合，flooding 是早期移植选择。

## FDR 状态

`fdr.py` 仍实现 arXiv:2410.03877v3 的 Eq.9 与 Algorithm 2：epsilon*lambda、两种标签方向的 hinge epigraph、dual-norm 约束、精确凸子问题、加权全局模型和 scaled dual 更新。ADMM-SC 单独添加 tau*||w||²，并检查 Theorem 3 的 rho 上界。

来源：https://arxiv.org/html/2410.03877v3 ，作者代码 https://github.com/mibrahim41/FDR-SVM/tree/6e44afc882575258175fca380b2c074188e2a419 。本地保留来源及 MIT 许可证。

参考版用 CVXPY/Clarabel 代替作者的 Gurobi；检查求解状态和约束可行性。UCI 默认等客户端权重、epsilon_g=1/(10*N_g)、kappa=1、w0=0、mu0=1。ADMM-SC 与直接集中式同目标解已数值核对。固定训练轮数不被标记成收敛，残差仅作为诊断。

`transport.py` 的 FDR exact-ring 仍是独立网络适配器，**FDR 尚未接回 PeerSim**。旧的二次代理/单跳 FDR 入口继续停用。

## 当前参考代码

- SSL 作者来源及校验值：`author_ssl/SOURCE.json`。
- SSL 可复用训练/分类器：`ssl.py`；PeerSim：`../src/fedssl_simulation.py` 和 `../p2p/fedssl_gossip.py`。
- SSL 信号数据：`iq_data.py` 的 `author_synthetic_clients`；原 Eq.34 数据生成函数仍保留为单独的 `synthetic_clients`，不混称作者发布数据。
- FDR：`fdr.py`；服务器/独立网络入口：`../run_paper_reference.py fdr`。
- 原始近似实现和旧结果均仅保留作历史审计；不得作为新论文复现结果。

`requirements.txt` 记录 Windows 验证依赖。SSL 采用作者发布配置，不宣称完整复现论文 v2 的文字参数或报告精度。论文 Algorithm 1 的普通编码器聚合也不自动获得其 proximal 理论变体的保证。
