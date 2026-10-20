# a9a 实验结果与项目记忆

2026 年 10 月 20 日。全部计划运行已结束。CoCoA 与 CoCoA+ 按最新要求并行运行，FedAvg 继续原任务。暂不交付图。

## 新结果

通信采用十进制 MB。准确率为通过停止门限后的节点均值。

| 方法 | 状态 | 周期 | 测试准确率 | 训练 hinge | 通信 MB |
|---|---|---:|---:|---:|---:|
| 集中式 | 求解器停止 | — | 84.9702% | 0.350959 | 0 |
| SDCA | 通过门限 | 252 | 84.9874% | 0.351142 | 75.953808 |
| BDSVM linear | 通过门限 | 202 | 84.9702% | 0.350920 | 4975.336440 |
| FedAvg | 通过门限 | 2900 | 84.8246% | 0.353634 | 86.304000 |
| CoCoA | 通过门限 | 252 | 84.9518% | 0.351282 | 75.953808 |
| CoCoA+ | 通过门限 | 252 | 84.9598% | 0.351286 | 75.953808 |
| FedSSL-AMC 表格 gossip | 达到上限 未收敛 | 5000 | N/A | 0.737889 | 28954.800000 |

FedAvg、CoCoA、CoCoA+ 运行有重叠，耗时分别为 1088.293、71.782、60.265 秒。这些不是独占机器计时，不能据此计算公平的集中式加速比。相关独占计时和加速比字段已清空；原始归档里的自动顺序计时标签不作为分析依据。

## 实验口径

- 数据为 a9a，32561 条训练、16281 条测试、123 维、10 个节点，seed 42。
- 相同 random k-out 无向拓扑，k=3。各节点每周期激活一次并发送给 3 个邻居；同一事件种子调度独立于学习随机数。
- numeric-v1 载荷计量：数组按实际 dtype，数值标量和数值来源 ID 各 8 bytes；排除固定字段名、Python 内存、物理网络头与离线监测通信。
- P2P 停止采用 G=50 相邻训练窗口，rtol=.001、atol=1e-8，全部节点连续两次通过。上限 5000 周期；集中式采用求解器容差。
- 测试集只在通过停止门限后评估。SSL 本轮没有测试评估，不可填入旧准确率。
- 最终 hinge 在完整训练集上计算，各最终节点分类器取平均；SSL 使用各节点自己的编码器和 SVM。
- 平台期不证明同等最优性。模型容量、目标、本地更新预算和求解器仍有差异。

## SSL 结论与待查问题

当前 a9a 是表格适配，不是原论文性能复现：MLP 123→128→64，float64，特征替换 .3，每个 anchor 配 10 个负样本，10 个 Adam steps/轮，batch 128，lr .001。最终冻结本地编码器并训练本地线性 SVM。

当前共享真实 gossip 层，只发送自身当前编码器，不再传播全来源模型表。每条消息 24128×8+8=193032 bytes；每周期 30 条消息，即 5790960 bytes。5000 周期恰为 28954800000 bytes，即 28.9548 GB。

最后五次窗口检查稳定节点数为 1、0、0、0、0；最后一次最差变化约为容差的 644 倍。末次固定训练对比损失约 1.2424，历史最低约 .7289。未收敛原因仍需排查；跨轮优化器状态、学习率、邻域参数平均只是待验证方向。

原论文采用中心服务器。旧 flooding 是我们移植时引入的选择，旧 100.39 GB 不能代表原论文，也不能代表当前 gossip。旧 84.45% 不能充当本轮准确率。

## 已确定需求与算法状态

- 先用 a9a，每个 dataset 一张四分图。communication cost、最终 test accuracy、停止轮数用柱状图；hinge 在训练集计算；去掉 consensus error。最新指示暂停图交付。
- BDSVM 本轮为 linear，P=100、C=300、gamma=None。RBF 核宽度修复保留，不参与本轮。
- CoCoA 本地步已核对；异步 gossip 的旧贡献传播是 P2P 适配，不等同原文同步轨迹。
- FedAvg 是邻域平均与本地 Pegasos 的 P2P 变体，步数计数跨轮保留。
- FedSSL-AMC 的 I/Q 作者实现已核对并做小规模验证，未宣称复现论文完整精度。当前表格 gossip 为独立适配。
- FDR 论文 Eq.9 与 Algorithm 2 参考实现采用 CVXPY 与 Clarabel，尚未接入 PeerSim，本轮不包含。旧二次代理实现停用。
- 后续需排查 SSL 稳定性；若继续比较 speedup，应另行获取可比的独占计时；FDR 接入后再核对 ADMM 更新和残差。

## 验证与文件入口

先前 Windows 上 38 项回归检查通过。本轮独立审计通过：数据哈希、实际邻接表、独立重建的首百轮邻居调度、统一载荷规则、30 条消息每周期、SSL 停止窗口算术、测试门控、线性 BDSVM 与有效时间比。

工作区为 `/Users/riesling/SVM/FDR-SVM-Lincoln`。

- 当前指标：`results/a9a_controlled_gossip_20261020/output/comparison/*_metrics.json`
- 审计：同一 output 下 `independent_audit.json`
- SSL 全节点模型：同一 output 下 `ssl/encoders.pt` 和 `ssl/local_linear_svms.joblib`
- 并行记录：结果根目录 `parallel_results.zip`
- 关键入口：`run_ssl_tabular.py`、`run_convergence_compare.py`、`p2p/fedssl_tabular_gossip.py`、`src/ssl_tabular_gossip_simulation.py`
- 定义文档：`paper_reference/SSL_TABULAR.md`、`SSL_PEERSIM.md`、`ALIGNMENT.md`
- 远端运行器临时加入队列复用桥接，避免重复训练；未改优化步骤。源码校验记录在 `output/source_hashes.json`。
- 作者仓库 FedSSL-AMC_Github 固定提交 `7cf7dc290aa208ef525c56b4591a501e547b8abf`；表格增强借鉴 SCARF 思路。

未经授权不发送群聊、不修改 GitHub。后续不得混用已作废 flooding 结果。
