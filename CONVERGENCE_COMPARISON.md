> 2026-10-20：旧 SSL flooding 对比已弃用。当前入口 `run_ssl_tabular.py` 使用共用 one-hop gossip；所有方法通信调度独立于学习 RNG，数值载荷按 numeric-v1 统一计量，训练停止按 G=50 相邻窗口对所有节点判断。详见 `paper_reference/SSL_TABULAR.md`。旧结果不得混入新比较。

> **最新：SSL 已按作者发布代码接入 PeerSim 的 I/Q 独立入口；BDSVM 默认 linear，RBF 公共核宽度已修复。详见 [SSL_PEERSIM.md](paper_reference/SSL_PEERSIM.md)。**

> **2026-10-20 更新：本文的 FDR 二次代理和 FedSSL SVD 接入说明属于历史版本，已停用。**
> 新论文参考实现及尚未确认的细节见 [paper_reference/ALIGNMENT.md](paper_reference/ALIGNMENT.md)。
> 当前 a9a 比较入口暂时排除这两项，SSL 先在 I/Q 数据上验证。

# 单数据集比较：训练停止、最终测试与通信统计

此次使用入口 `run_convergence_compare.py`。原有 `run_compare.py` 和历史结果保留，
便于核对旧实验；新入口替换观察器后运行同一批现有训练协议。

新增比较项为 FDR-SVM 和 FedSSL，加上 SDCA、BDSVM、FedAvg、CoCoA、CoCoA+，
以及 centralized baseline。当前仅运行 a9a，Gadget 留待后续。

## 停止规则

每一训练轮结束，离线观察器对每个节点的模型计算完整 train set 上的训练目标。
取最近 10 **轮**的均值；相对历史显著最优目标改善超过 `1e-3` 时重置 patience。
当所有节点连续 `G=50` 轮未显著改善，且当前平滑目标仍接近历史最优目标时，
同时停止所有节点。后一个条件避免发散或显著恶化被误报为收敛。
参与贡献表的算法还需先收到所有来源；FedSSL 需先完成分类器聚合。
目标或平滑值非有限时不能触发收敛。

这里的“收敛”是操作性的 plateau 判据，并非达到全局最优的数学证书。
最多 5000 轮；到达上限标记 `max_rounds`，不算收敛。
FedSSL 若局部 SVM 未满足求解器容差，则单独标记 `local_solver_limit`。
测试集不参与停止、超参数选择和训练曲线；仅在满足停止条件后，每个最终节点模型
计算一次 test accuracy。没有满足条件时，accuracy 和收敛 speedup 均为 N/A。
CSV 另保留所有算法的 `runtime_ratio_at_stop`，包括跑满预算的耗时比；图中用斜线
区分这种预算截止耗时比，不把它当作达到收敛的加速。

SDCA、FedAvg、CoCoA 和 CoCoA+ 使用 mean hinge + `lambda/2 ||w||²`。
FDR 沿用现有二次鲁棒正则代理，用样本数加权的 eps 加到 lambda。
BDSVM 保留现有 RBF kernel、预算、C 和稳定化措施，监测其 IRWLS 对应的缩放目标
`mean hinge + betaᵀ Kpp beta/(4 C N)`。这不是把所有算法改成相同目标的实验，
不同模型、正则强度和求解预算会影响精度与耗时。

## FedSSL 接入

旧 `p2p/fedssl_protocol.py` 依赖已不存在的 engine 路径，且各节点冻结不同坐标的
encoder 后直接平均分类器。新 `p2p/fedssl_gossip.py` 将现有的“局部 SVD + 编码后
局部 SVM”近似接入当前 gossip engine：

1. 每个节点仅在自己的训练分片上计算 SVD。
2. 通过不可重复累计的来源表交换编码器；所有来源到齐后，按来源排序，以共同的
   第一个编码器为参考做正交 Procrustes 对齐、样本数加权平均和 QR。
3. 每个节点得到相同的冻结 encoder，训练无 intercept 的局部 hinge SVM。
4. 用来源表交换固定局部分类器，再按局部样本数聚合。

局部 SVM 使用 `C=1/(n_local * lambda)`、tol `1e-4`、最多 1,000,000 次迭代；
记录每个节点是否达到求解器容差。来源表包含编码器与分类器，实际发送的完整表
计入通信量。该实现是仓库 **SVD-based tabular adaptation** 的修复与迁移，
不是原论文神经网络 SSL-AMC 的逐项复现，不据此宣称原论文的结论。

## 图和计量口径

- 训练 hinge loss 保留曲线：每个节点模型在全部 train shards 上的 hinge，再取节点均值。
- 总 communication cost 为 bar chart：累计到实际停止轮、所有节点的发送 payload 字节。
- Test accuracy 为 bar chart：满足停止条件后，最终节点模型在相同测试集上的 accuracy 均值。
- 原 consensus error 图移除，改为实际停止轮数 bar chart；未收敛的条形加斜线。
- Speed ratio 为 bar chart：`centralized training seconds / algorithm training seconds`。

通信沿用各训练协议的数值 payload 计数，不含传输协议头和物理网络开销，
每个有向发送计一次，不另外重复加收件流量。FedSSL 计入数组及来源/权重数值。
观察器直接读取模拟内存中的训练分片，其额外部署通信未模拟；其计算时间计入耗时。

Centralized 使用全部相同训练样本、相同 lambda、无 intercept 的 hinge LinearSVC，
`C=1/(N*lambda)`，tol `1e-5`，最多 100,000 次迭代。
这修正了旧 baseline 使用 squared hinge 和 `C=1/lambda` 的口径。

结果先用 8 个独立进程并行生成，在 24 logical CPU 上每个进程 3 个 BLAS/OpenMP
线程。随后各算法使用相同 3 线程预算单独重跑计时，核对停止轮、通信量与最终精度
是否一致，避免把并行资源竞争算进 speed ratio。计时包含模型构建、初始化、
warm start、本地训练、通信模拟与 train monitoring；不含数据读盘和最终 test evaluation。
这只是**单机模拟实现的时间比**，不是实测多机并行加速或等精度到达时间。

每个算法独立记录 config、数据文件 SHA-256、训练/测试样本数、运行日志和原始曲线。
PNG 和 PDF 各输出单图及总览，CSV 提供精确值。单次 seed 42，无跨 seed 置信区间。

## 运行

### a9a 预览（2026-10-19）

本次预览目录为 `results/a9a_2026-10-19_preview`，包含 centralized 及七种
P2P 方法。用 `--jobs 1 --threads 3` 依次执行；单任务运行时直接记录该次
训练耗时为 isolated timing，不重复训练。FedAvg 改用相邻的、互不重叠的
50 轮训练目标均值：所有节点均满足相对变化小于 `1e-3`，且连续两次检查
通过后全局停止。它不再受历史偶然最低值的约束；其余 P2P 方法沿用上述
10 轮平滑与 G=50 patience 规则。两者均不能当作 duality-gap 最优性证书。
旧 FedAvg 草稿探索运行在约 2400 轮后取消，未作为最终结果使用；以新规则
单独重跑并记录实际停止轮、总通信量和最终 test accuracy。

`plot_convergence_preview.py` 输出一张总览及五张单项 PNG：最终 test accuracy、
全部节点总发送量、训练 hinge 曲线、实际停止轮数和相对 centralized 的时间比。
时间表直接列出秒数及停止状态。FedSSL-AMC 的柱形标星号注明 SVD 改编，
未满足停止条件的值保留 limit/N/A 标记。

```powershell
python remote_convergence_job.py preview
python plot_convergence_preview.py results/a9a_2026-10-19_preview
```

```powershell
python run_convergence_compare.py adult `
  --data-dir 'C:/Users/remote/GadgetSVM-Lincoln/data/processed' `
  --out results/a9a_2026-10-16 --jobs 8 --threads 3 --cycles 5000
```

使用全新输出目录，避免覆盖先前的实验。`--skip-isolated-timing` 仅供执行检查，
不会报告 speed ratio。重新绘图：

```powershell
python plot_convergence_compare.py results/a9a_2026-10-16
```
