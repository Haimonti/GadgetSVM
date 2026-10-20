> **后续更新：已找到 SSL 作者源码并接入 PeerSim；BDSVM 的 linear 默认与 RBF 公共核宽度已修复。以下为修复前的发现记录，最新状态见 [SSL_PEERSIM.md](SSL_PEERSIM.md)。**

# PeerSim P2P 实现核对 — 2026-10-20

## 核对标准

P2P 是本次实验的目标。核对的是原方法的目标、本地更新、关键聚合不变量，以及改写带来的假设和数值稳定措施。没有服务器本身不构成错误。原同步算法的证明不能自动用于异步 gossip；这与实现是否保留核心算法是两个判断。

此前的 15 项集成测试验证流程和指定数值性质，不是所有算法完整论文复现的认证。

## 结论

| 方法 | 已确认 | 仍需处理或明确 |
| --- | --- | --- |
| CoCoA | 本地投影 dual coordinate update、全局 lambda*N、1/K 提交比例保留 | 当前节点使用可能过期的贡献表，是异步 P2P 改写；未证明原同步收敛界仍适用 |
| CoCoA+ | gamma=1 相加、sigma=K*gamma 的本地曲率保留 | 同上；不能只因使用版本表就声称与每轮同步聚合数值一致 |
| FedAvg | 本地线性 SVM 子梯度训练、按样本数加权的邻居模型混合 | 属于 gossip FedAvg-SVM；任意拓扑上的现有混合未证明保持全局样本权重。全程递增的 Pegasos 步数也是明确的优化器选择 |
| BDSVM | 随机 pre-images、IRWLS、局部 C_m/d_m 与求和结构保留 | **发现公共核参数不一致的问题**：gamma=None 时各节点按各自数据估计 gamma。另有过期统计、a_i 权重截断和 ridge 稳定化，需明确为 P2P 改写设置 |
| FedSSL-AMC | 新 I/Q 参考版的 logistic triplet、编码器训练与加权聚合、冻结 encoder、个性化本地 SVM 流程已对照原文 | 编码器层数解释和部分实验设置仍未确认；新参考版的 exact-ring 模拟器**尚未接入 PeerSim**。旧 PeerSim SVD 实现已停用 |

## 可复查的数值证据

脚本 `verification/p2p_audit/audit.py` 在私人 Windows 主机执行，输出 `verification/p2p_audit/audit.json`。

- 用相同初始 alpha/w 和相同坐标访问序列，把 PeerSim 的 CoCoA/CoCoA+ 本地步骤与作者公式对应的 Python 参考步骤比较。两者最大误差均为 `5.55e-17`。这是对本地更新的核对，不是任意拓扑全程等价证明，也未覆盖零特征行等全部输入边界。
- BDSVM 两个不同尺度训练分片使用相同随机 pre-images 时，默认 gamma 分别为 `2.341367` 和 `0.002197`，公共 Kpp 的最大差达到 `0.95919`。公共基坐标上的系数不能同时代表两个不同核映射。后续修复应统一训练选定的 gamma，并明确初始化通信成本；不能通过改名把这个问题视为普通通信差异。
- FedAvg 等样本数三节点星形图，在一次同步邻居混合后，模型均值由 `1/3` 变成 `5/18`。这证明该混合并非一般意义上的全局均值保持操作；不等于证明当前 a9a 实验一定发散或一定达到错误解。
- SSL 默认 upstream depth=10 实际产生 11 个残差块、22 个 kernel=3 主卷积，以及 2 个 1x1 投影卷积，共 24 个 Conv1d，247,880 个参数。参数量接近原文的约 0.247M，支持这一实现选择，但不能据此证明原作者的“10 层”就是这个定义。新结果会输出实际层数。

## SSL 两个版本不能混用

| 合成 I/Q 实验设置 | v1（2025-10-06） | v2（2026-08-15，当前参考版依据） |
| --- | --- | --- |
| 无标签数据/客户端 | 14,000 | 140,000 |
| 有标签数据/客户端 | 2,800 | 2,800 |
| 通信轮数 | 10 | 10 |
| 本地步数/轮 | 2,500 | 15,000 |
| batch size | 20 | 50 |
| Adam 学习率 | 0.001 | 0.001 |
| 描述的 encoder | 10 层、kernel 3、embedding 320 | 同左 |

当前 `SSLConfig` 默认采用 v2 的已公开预算。之前的 10 轮 smoke 则每轮仅 10 步、batch 8、每客户端 14,000/280/28 条无标签/有标签/测试信号；它既不是 v1 的完整实验，也不是 v2 的完整实验。

已确认原文的两阶段流程；**仍不能确认完整网络配置与作者实验完全相同**。未在两版原文中找到公开作者实现链接。裁剪长度分布、Adam 状态跨轮处理及实验 SVM kernel/C 还需作者实现或补充材料才能最终确认。原文的硬间隔线性 SVM 理论分析本身不能证明实验采用了同一分类器设置。当前 RBF SVC C=1 是明确记录的实现选择。

## 来源

- CoCoA 作者代码（固定提交）：https://github.com/gingsmith/cocoa/blob/5363b92dc2ca1401047214af5b2e5da7540c7644/src/main/scala/solvers/CoCoA.scala
- CoCoA+ 原文：https://proceedings.mlr.press/v37/mab15.pdf
- FedAvg 原文 Algorithm 1：https://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf
- BDSVM 原文（作者大学资料库）：https://e-archivo.uc3m.es/entities/publication/a1f6862a-f23d-4901-a1e1-606de760ce17
- SSL v1：https://arxiv.org/html/2510.04927v1
- SSL v2：https://arxiv.org/html/2510.04927v2
- 引用的 encoder 默认配置：https://github.com/White-Link/UnsupervisedScalableRepresentationLearningTimeSeries/blob/4aff59229113368ace606ad76e149a7916daa6f9/default_hyperparameters.json

本轮先完成确认与记录，尚未改动四个既有 PeerSim 算法的优化/聚合规则，也未重新生成比较图。
