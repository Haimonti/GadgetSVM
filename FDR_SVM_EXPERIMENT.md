# FDR-SVM 与最新 Lincoln SDCA 的 PeerSim 对比

## 实验来源与口径

- SDCA 源码基于 GitHub `Haimonti/GadgetSVM` 的 Lincoln 分支提交 `9c319782dffbd421d7cf64bc00dbb722cd31d8b4`（2026-10-06 核对）。本包 `code/` 保存此次运行的完整 PeerSim 相关源码与驱动。
- 8 个数据集分别是 covtype、gisette、real-sim、rcv1、ijcnn1、a9a、w8a、webspam。输入文件与该提交 `linear_data_manifest.json` 的字节数和 SHA-256 一致；covtype 与 gisette 在发现远程旧缓存不一致后，已使用清单匹配文件重跑，最终表格采用重跑结果。
- 每个数据集的 SDCA 和 FDR-SVM 在同一个进程中调用一次 `load_shards`，共享相同的训练分片和测试集。8 个数据集同时作为独立进程运行。10 个节点，seed 42，random_kout 无向网络、k=3，lambda=1e-4，最多 5000 轮，每 10 轮评估。SDCA 每轮 4096 次坐标更新，FDR-SVM 每轮 100 次本地 Pegasos 更新。两者本地计算预算不同，因为保留了各自已有实现的默认值。
- 准确率为 10 个节点在同一测试集上的准确率均值。通信量是所有节点累计发送的 payload 字节，未计协议头、接收流量和物理网络开销。SDCA 发送带版本的贡献表；FDR-SVM 向每个邻居发送 float32 的 `w+u` 和分片样本数，因此二者消息结构不同。

## 结果

| 数据集 | SDCA 准确率 | FDR-SVM 准确率 | FDR−SDCA | SDCA 轮次 | FDR 轮次 | SDCA MB | FDR MB |
|---|---:|---:|---:|---:|---:|---:|---:|
| a9a | 0.8501 | 0.8398 | -0.0102 | 1161 | 5000 | 180.7 | 130.0 |
| covtype | 0.7622 | 0.6903 | -0.0719 | 161 | 5000 | 18.7 | 58.2 |
| gisette | 0.9700 | 0.6859 | -0.2841 | 641 | 5000 | 7414.5 | 5202.1 |
| ijcnn1 | 0.9195 | 0.9050 | -0.0145 | 111 | 5000 | 4.2 | 25.0 |
| rcv1 | 0.9620 | 0.9198 | -0.0422 | 51 | 5000 | 3423.8 | 49127.5 |
| real-sim | 0.9680 | 0.7956 | -0.1724 | 61 | 5000 | 3025.8 | 21798.4 |
| w8a | 0.9041 | 0.8962 | -0.0079 | 171 | 5000 | 117.4 | 314.1 |
| webspam | 0.9149 | 0.7830 | -0.1318 | 61 | 5000 | 36.9 | 266.2 |

8 个数据集均成功完成。FDR-SVM 的准确率均低于本次 SDCA；FDR-SVM 在全部 8 项都达到 5000 轮上限，未触发参数停滞停止条件，因此 5000 轮不表示收敛。SDCA 在表中轮次结束，停止依据是 duality gap 或相邻评估窗口的参数变化。逐轮曲线见 `plots/`，逐节点原始指标和日志见 `results/`。

## FDR-SVM 算法如何实现

每个节点 i 只持有自己的 CSR 训练分片 `(X_i, y_i)`，样本数 `n_i`。设局部模糊半径 `eps_i = eps_scale / sqrt(n_i)`，本次 `eps_scale=1`。源码按原项目 FDR-SVM 的**二次正则化代理目标**实现局部鲁棒项，局部曲率是 `lambda + eps_i`。每个节点保存权重 `w_i`、缩放对偶变量 `u_i` 和邻居共识估计 `z_i`，初值为零。

每轮在 PeerSim `CDSimulator` 中按洗牌顺序激活节点。节点把自己与已收到邻居的 `(w_j+u_j, n_j)` 依样本数加权，得到 `z_i`；更新 `u_i <- u_i + w_i - z_i`，再以 `v_i=z_i-u_i` 为中心做 100 步稀疏 Pegasos 近端更新。每一步随机抽取一条本地样本，步长 `eta_t=1/((lambda+eps_i+rho)t)`，其中 `rho=1`；当 margin 小于 1 时添加 hinge 子梯度。求解器将权重写成 `s*u+c*v_i`，只在样本非零特征上更新，避免高维数据每一步遍历整条稠密向量。然后向拓扑中的每位邻居发送 `(w_i+u_i,n_i)`。收件箱在下次激活时合并。

测试准确率、训练 hinge、二次代理目标、共识误差和通信字节由观察器每 10 轮记录。FDR-SVM 无可用 duality gap 证书；当相邻评估窗口的相对参数变化低于 `1e-3` 时单个节点可停止。本次各数据集均未满足全部节点的停止条件。

**数学解释边界：** 原仓库把 `eps_i` 描述为 Wasserstein 半径，但代码实际优化的是 `(lambda+eps_i)||w||²/2` 的代理形式。一般线性 hinge 损失的 Wasserstein 鲁棒项是与 `||w||` 成正比的项；因此本实验不能作为精确 Wasserstein DRO 解或鲁棒性保证。邻域加权 ADMM 也是原有 P2P 近似形式，这里未证明它与中心化 ADMM 有相同极限。结果应解释为该具体 FDR-SVM 实现的实测表现。

## 复现

在 `code/` 对应源码中安装 NumPy、SciPy、scikit-learn、matplotlib，设置 `GADGETSVM_DATA_DIR` 指向清单中的处理后数据，并运行 `python run_fdr_all.py --data-dir <路径> --cycles 5000 --jobs 8`。本次 covtype 和 gisette 通过 `--covtype-path`、`--gisette-path` 指向与清单一致的版本。`results/<数据集>/run.log` 记录每项实际执行过程。
