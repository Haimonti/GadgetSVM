> a9a 当前使用表格 + 真正 one-hop gossip 适配，见 [SSL_TABULAR.md](SSL_TABULAR.md)。本页 I/Q 参考端口的 exact flooding 用于核对服务器聚合等价性，不作为当前 gossip 通信比较。原论文采用中心服务器，不是 flooding。

# SSL 作者代码接入 PeerSim 与 linear BDSVM — 2026-10-20

## 当前结果

- SSL 已接入本仓库的 **PeerSim CDSimulator / CDProtocol / Linkable**，使用真实节点收发，不再只使用独立的网络模拟器。
- 核心编码器和 triplet loss 直接复用固定版本的作者源码；本地分类器按作者生效代码改为 LinearSVC，并只用本地训练数据做 C 的五折交叉验证。
- BDSVM 的当前比较入口默认 **linear**，不调用 RBF，也不估计 gamma。a9a 已跑通并触发训练停止规则：107 轮，最终节点平均测试准确率 **84.9647%**。
- RBF 模式保留作为显式选项，已修复各客户端 gamma 不同的问题；当前 a9a 实验不使用这个模式。

## 1. 作者源码与论文文字的差异已查清

找到的作者仓库明确对应这篇论文的 v1：
https://github.com/usmanakua927/FedSSL-AMC_Github

固定提交：`7cf7dc290aa208ef525c56b4591a501e547b8abf`。
来源与逐文件校验值保存在 `author_ssl/SOURCE.json`。编码器文件与先前引用的 Franceschi 实现完全相同；现在配置和训练行为以作者仓库的**实际生效代码**为依据。

| 项目 | 作者发布版的实际设置 |
| --- | --- |
| 编码器 | channels=80，depth=7，reduced_size=160，embedding=320，kernel=3，float64 |
| 实际结构 | 8 个残差块，16 个主卷积，2 个 1x1 投影，共 18 个 Conv1d，432,880 个参数 |
| 本地预算 | 2,500 步/轮，batch=50，Adam lr=0.001，10 个负样本 |
| 裁剪 | 直接使用作者 `TripletLoss`；server 的 compared_length 默认 192 |
| 负项 | 按负样本数**平均**，不同于论文 Eq.1 的求和文字 |
| 负样本来源 | 作者采样代码允许抽到与 anchor 相同的序列；按发布代码保留 |
| 编码器聚合 | 作者 `weights_agg` 的等权平均；另支持显式 sample_weighted 选项 |
| 本地分类器 | 生效代码为 **LinearSVC**，不是其旧注释写的 RBF SVC |
| C | 本地训练集五折 CV，候选 1e-4 到 1e4；CPU 移植将 n_jobs 设为 1 |

因此这里采用的是**固定作者发布版的移植**。它不是对论文 v2 所有文字/数值的逐项复现，也不能再把 `depth` 直接解释为 Conv1d 模块总数。作者发布配置与论文中的“10 层”、v2 的 15,000 本地步数并不一致；这些差异现在是明确来源，不再是猜测。

只复用必要的 encoder/loss；原始服务器多 GPU 启动脚本没有被执行。原信号生成脚本会在导入时写文件，因而只逐字提取其中的纯信号函数用于数据生成。原文件和已有的上游版权声明均保留。

为满足本实验要求，移植版使用共同的确定性初始编码器、每个客户端/轮独立 RNG，并取消作者脚本中的逐轮 test 评估。它们属于明确记录的部署/评估调整。

## 2. PeerSim P2P 更新

入口：`run_ssl_peersim.py`；组装：`src/fedssl_simulation.py`；节点协议：`p2p/fedssl_gossip.py`。

每个训练轮采用固定同步时段：

1. 每个节点从自己上一轮得到的共享编码器开始，仅用本地无标签 I/Q 训练一次。
2. 通过 Linkable 邻居转发当前 epoch 的不可变来源记录；重复来源不重复累计，旧 epoch 不污染新轮。
3. 预留 N-1 个转发周期，之后每个节点各自聚合全部来源的编码器。
4. 重复固定训练轮数；最后冻结编码器，各节点拟合自己的 LinearSVC。
5. 全部训练结束后，每个本地测试集仅评估一次。SVM 权重不交换、不聚合。

没有由 Control 聚合模型，也不从 Network 读取别人的模型来代替通信。Control 只负责初始化和检测全部节点完成。该实现需要可靠且强连通的网络，以及每周期每节点激活一次（ordered/shuffle）；不支持 getpair_rand 的遗漏/重复激活、节点失效或无界延迟。

训练轮数和 PeerSim 链路周期分开记录。当前固定同步预算每轮 N 个 PeerSim 周期（N-1 次转发预算 + 1 次本地聚合时刻）；已有异步算法的一个 cycle 与它不具有相同工作量，后续比较必须保留这个区别。

通信记录每次有向发送的实际来源表张量和每条记录 32 字节元数据。共享种子/配置视为预先已知；不含物理网络开销。模型初始化由共同种子完成。最终各节点已自行得到聚合编码器，无需服务器下载；本地 SVM 不产生通信。

停止状态是 `author_fixed_round_budget`，`converged=null`。不会把固定轮数跑完宣称为数学收敛。

## 3. linear BDSVM 与 RBF 修复

`BDSVMSimulation`、`BDSVMGossipProtocol`、`methods.bdsvm.run` 和比较入口的默认模式均为 linear。线性模型使用共享随机基向量，保留论文的预算 IRWLS 框架；它仍与 SDCA 的求解器、正则参数不同，不能仅凭都线性就宣称优化目标/容量完全相同。

a9a 本轮参数：P=100、d=123、C=300、rho=0.5、ey_floor=0.01。线性预算的有效模型在该共享基张成的子空间内，秩不超过 P。此次只切换核类型，保留原先的训练参数作为预览，尚未重新调优 linear 的 C。

显式选择 RBF 且 gamma 未指定时，节点先传播各自训练分片的候选 gamma（连同来源和样本数），收到所有来源后采用候选值的共同中位数，再生成 Kpp 和训练矩阵。这个值是**客户端候选的中位数**，不是所有原始距离的集中式中位数。各节点核矩阵一致已验证。校准阶段的通信计入总量；校准完成前不允许训练统计或收敛判据生效。

IRWLS 的异步旧统计、ey_floor 和 ridge 稳定化依然是本 P2P 改写的设置，不据此声称原同步论文的收敛证明已覆盖它们。

## 4. 验证与结果

- 最终 **26 项测试通过**，包括来源校验、聚合权重解析检查、server linear 路径和空客户端校准回归；结果见 `verification/ssl_peersim_final/`。
- 实际 PeerSim 与服务器参考版在 ring/full、ordered/shuffle、uniform/sample_weighted 下逐参数一致；本地 SVM 系数一致。
- 检查断网拒绝、作者源码校验值、作者负项归一化、两节点通信字节数、最终仅评估一次测试集，以及 linear 完全不调用 RBF/gamma。
- 10 轮 I/Q smoke：4 客户端，每轮仅 1 个本地训练步，batch=8；每客户端 1,400 条无标签、280 条有标签、28 条测试序列。训练 triplet loss 从 31.84 降至 1.52；40 个 PeerSim 周期；平均测试准确率 52.68%。这只是移植和通信流程检查，**不是作者完整预算或论文精度复现**。
- I/Q smoke 原始结果：`verification/ssl_peersim/validation/ssl_peersim/metrics.json`；编码器和四个本地 LinearSVC 也保存在该目录。
- a9a linear BDSVM 原始结果：`../results/a9a_linear_20261020/a9a_linear/bdsvm_metrics.json`。通信约 2.6226 GB；训练时间约 3.37 秒。本次与独立 SSL 检查并行运行，因此不据此重算严格隔离条件下的 speedup。它使用原有训练 plateau 规则，非最优性证书。
- a9a 训练子进程正常结束；第一版远程包装缺少绘图模块，报错发生在训练后的自动绘图阶段，已修正后续作业的打包清单。数值结果已经取回并核对，本轮不交付新的比较图。

复跑（按约定在 Windows 计算主机上执行）：

```text
python -m unittest discover -s tests -v
python -m paper_reference.verify
python run_ssl_peersim.py --profile smoke --rounds 10 --local-steps 1 --out results/iq_peersim
python run_ssl_peersim.py --profile author --out results/iq_author_budget
python run_convergence_compare.py adult --data-dir DATA_DIRECTORY --only bdsvm --jobs 1 --threads 3 --out results/a9a_linear
```

SSL 仍先使用 I/Q；没有把 a9a 表格强行变成时序，也没有恢复旧 SVD 替代品。
