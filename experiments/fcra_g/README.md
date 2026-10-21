# FCRA-G：延迟标签锚定的在线图校正

新增独立入口 `experiments.fcra_g`，保留现有 FCRA 与 `fcra_dual`。FCRA-G 在完整
记忆 FCRA 的参数学习路径之外，增加一条不更新网络参数的预测校正路径：用当前无标签
样本的邻域结构，把已返回真值作为类别锚点传播到当前预测。

两条路径相互分离：

1. 参数学习：已返回标签通过完整监督记忆重放，维护长期分类能力。
2. 预测校正：在当前 batch 上构建局部图，求解无标签节点概率，并用延迟反馈选择
   校正强度；图结果不回写主模型，也不作为伪标签参与训练。

本方法是待验证的研究方案，不保证优于现有 FCRA。

## 服务器运行

在仓库根目录、原实验环境中执行：

```bash
conda activate df-atta

python -u -m experiments.fcra_g.run \
  --dataset-root datasets \
  --gpus 0 1 \
  --seeds 110 123 456 \
  --output-dir storage/fcra_g/pacs_bs32_seeds110_123_456
```

双卡时每张卡同时运行一个完整任务，空闲卡自动领取下一任务。单卡：

```bash
python -u -m experiments.fcra_g.run \
  --dataset-root datasets \
  --gpu-idx 0 \
  --seeds 110 123 456 \
  --output-dir storage/fcra_g/pacs_bs32_seeds110_123_456
```

指定物理卡：

```bash
CUDA_VISIBLE_DEVICES=2,3 python -u -m experiments.fcra_g.run \
  --dataset-root datasets --gpus 0 1 \
  --seeds 110 123 456 \
  --output-dir storage/fcra_g/pacs_bs32_seeds110_123_456
```

`--gpu-idx` 与 `--gpus` 互斥。`--seeds` 必须是三个不同整数；不传时自动抽取三个
seed，四个 D 共用。默认输出 `storage/fcra_g/<UTC时间>/`，已有目录会报错。

默认检查点目录与现有 PACS 入口相同：

```text
storage/checkpoints/round1/PACS/[2, 0, 1, 3]/ResNet18_3l_meanpool_0dp_Truefzbn/0.0001lr_0.0wd/ERM_no_param/
```

需要已有 `datasets/PACS/{art_painting,cartoon,photo,sketch}`。不下载数据、不重训
源模型、不写入源检查点。

## 固定实验设置

继续固定：

- PACS Photo → Art → Cartoon → Sketch；
- online batch size = 32；
- D = 0、20、50、100；
- 每个 D 使用相同三个 seed；
- 旧 FCRA 的查询选择规则与查询时间；
- 旧 FCRA 的逐批早停监督更新结构和训练 minibatch = 16；
- ResNet18、冻结 BN、dropout=0、SGD lr=1e-3、momentum=0.9、`eh/el`、
  `cold_start`、`nc_increase`、`stop_tol` 等 SimATTA 配置；
- 评价指标：三个目标域准确率、stream accuracy、domain macro accuracy、
  queried/returned/cross-domain/pending。

FCRA-G 不向图模块分配监督预算，也不减少原 FCRA 的监督更新次数。图求解没有神经
网络反向传播；但额外前向、图构建和求解时间必须单独报告。

## 每批时序

D>0 时，每个全局 batch 依次执行：

1. 读取当前无标签 batch。
2. 继承 FCRA 先固定主模型预测与查询计划。查询选择只读取当前主模型预测，不读取真值。
3. 确定本批专家权重；对已有累计证据先做时间衰减。
4. 用当前 batch 与近期无标签记忆构图，并从完整监督记忆选择真实标签锚点。
5. 求解校正概率 `Q_U` 和锚点影响量 `h`。
6. 由三个固定专家生成最终融合概率，并在本批固定，用于在线准确率。
7. 为被查询样本保存查询时的三个专家概率、最终概率和当前权重。
8. 释放本批到期标签。
9. 在训练前用查询时保存的预测评分三个专家，更新后续 batch 的证据。
10. 将新返回样本加入完整监督记忆；执行原 FCRA 监督与伪监督训练。
11. 更新近期无标签记忆，进入下一批。

为严格复用旧 FCRA 的查询 ID 与查询时间，本实现先调用继承的查询规划。该规划只
使用主模型预测、特征和 pending 信息，与本批图计算相互独立；图校正仍在任何标签
释放和模型更新之前完成。评价真值只用于在线指标和 `graph.csv` 的离线诊断，不进入
图构建、专家权重或训练。

D=0 完全绕过新增机制：不构图、不维护无标签图记忆、不启用专家融合、不评分专家，
直接使用原 FCRA 路径。

## 两类记忆

### 完整监督记忆

沿用 FCRA 的 `received` 与完整重放记忆。保存所有已返回标签；监督重放始终使用
完整记忆，不受图锚点上限影响。低分样本不删除，未返回标签不参与监督。

### 近期无标签记忆

保存最近 256 个已经完成预测、但当前仍未返回真值的样本：

\[
M_U=\{(id_j,x_j,v_j)\}.
\]

- `v_j=normalize(E_source(x_j))`，源编码器冻结。
- 只保存已经观察到的图像，不访问未返回真值。
- 真值返回后从 `M_U` 移除。
- FIFO 淘汰只影响图校正；被淘汰样本返回真值后仍进入完整监督记忆。
- 查询台账、图记忆和完整监督记忆分开管理。

## 构图

### 无标签节点

`U` 包括当前 batch 全部样本和近期无标签记忆，按 `sample_id` 去重。当前 batch 节点
在前，`M_U` 节点在后。

对当前 batch 使用本批主模型概率；对 `M_U` 节点使用当前主模型重新计算
\(P_U[i]=p_\theta(x_i)\)，不复用旧模型缓存概率。

### 标签锚点

从完整监督记忆选择最多 128 个真实标签锚点。选择规则是确定性的：

1. 按类别分配基础名额 `128 // C`，余数按类别编号升序分配；
2. 每类内部按 `return_step` 降序、anchor index 降序，优先最近返回样本；
3. 某类不足时，剩余名额按类别编号升序轮转分配给仍有样本的类别。

该限制只控制图规模，不改变监督训练记忆。锚点使用真实 one-hot 标签，不使用伪标签。

### 特征与边

所有图节点使用冻结源编码器的归一化特征 \(v\)。每个节点在余弦相似度为正的邻居中
选择相似度最高的最多 `k=8` 个，不包含自身。初始有向边权：

\[
W_{ij}=\exp((\cos(v_i,v_j)-1)/\tau),
\qquad \tau=0.1.
\]

随后：

\[
W\leftarrow\max(W,W^\top),
\qquad
W\leftarrow \frac{W}{\max(1,\max_i\sum_j W_{ij})}.
\]

图不按真实域 ID 构边，也不按域边界重置。

## 概率校正

对无标签节点求 \(Q_U\)：

\[
\min_Q\;
\frac12\|Q_U-P_U\|_F^2
+\frac{\lambda}{2}\operatorname{tr}(Q_U^\top L_{UU}Q_U)
-\lambda\operatorname{tr}(Q_U^\top W_{UL}Y_L),
\]

默认 \(\lambda=1\)。等价线性系统：

\[
(I+\lambda L_{UU})Q_U
=P_U+\lambda W_{UL}Y_L.
\]

实现使用 `torch.linalg.solve`，不显式计算逆。求解在 CPU 双精度下完成，再转回
float32。只允许浮点修复：

- 极小负数截为 0；
- 每行重新归一化；
- 非有限值或求解异常时整批回退到 \(P_U\)。

锚点影响量：

\[
(I+\lambda L_{UU})h=\lambda W_{UL}\mathbf1.
\]

没有真实锚点或与锚点不连通时 \(h=0\)。\(h\) 是图结构量，不是分类正确率或校准
置信度。

## 三个固定专家

三个专家共用同一次图求解，不训练额外网络。

- 专家 0：主模型 \(p^{(0)}=P_U\)。
- 专家 1：保守图校正
  \[
  \beta^{(1)}=0.10+0.40h,\qquad
  p^{(1)}=(1-\beta^{(1)})P_U+\beta^{(1)}Q_U.
  \]
- 专家 2：较强图校正
  \[
  \beta^{(2)}=0.25+0.75h,\qquad
  p^{(2)}=(1-\beta^{(2)})P_U+\beta^{(2)}Q_U.
  \]

没有真实锚点时，两种图专家只做较弱平滑；有真实标签支持时允许更强校正。主模型
始终作为独立专家保留。最终预测：

\[
p_{out}=\sum_k w_k p^{(k)}.
\]

## 延迟反馈选择校正强度

### 查询时封存预测

样本在 step=q 被查询时保存：

- sample_id、query_step；
- 三个专家在该时刻的完整概率；
- 最终输出概率；
- 当前专家权重。

保存后不修改。这些预测生成时尚未使用该样本真值。

### 返回时评分

标签在 step=r 返回后，用查询时封存的概率计算：

\[
\ell_i^{(k)}=-\log\max(p_{q,i}^{(k)}[y_i],\epsilon),
\qquad
g_i^{(k)}=\operatorname{clip}(\ell_i^{(0)}-\ell_i^{(k)},-1,1).
\]

因此 \(g_i^{(0)}=0\)。每个返回标签只计一次。评分目的不是估计整个当前域的无偏
收益，而是记录该专家在当时可用信息下是否优于主模型。

### 证据与权重

每个 batch 先衰减：

\[
S_k\leftarrow\rho S_k,\qquad \rho=0.99.
\]

新返回事件加入：

\[
S_k\leftarrow S_k+\sum_{i\in R_t}g_i^{(k)}.
\]

默认先验 \(\pi=[0.6,0.3,0.1]\)，\(\eta=0.1\)：

\[
w_k=\frac{\pi_k\exp(\eta S_k)}{\sum_j\pi_j\exp(\eta S_j)}.
\]

实现使用 `softmax(log(pi) + eta * S)` 的稳定形式。本批新反馈更新后的权重只用于
后续 batch；当前 batch 的预测在释放标签前已经固定。

主动查询存在选择偏差，因此这些证据只反映被查询样本上的历史表现，不是当前全部
数据上的无偏风险估计。

## 诊断与计算开销

每个 batch 的 `graph.csv` 记录：

- 主模型、最终融合、两个图专家的准确率；
- 主模型与各专家的预测分歧率；
- 图校正改变的预测数；
- 修改后正确、修改后错误的数量；
- 当前 batch 节点的 `h` 均值和 `h=0` 比例；
- 三个专家权重；
- 图节点数、锚点数、无标签记忆节点数、边数（对称化后的无向边）；
- 图构建耗时、线性求解耗时、是否回退。

每个返回事件的 `expert_feedback.csv` 记录：

- 查询时三个专家的真实类别概率、最终输出概率；
- 三个专家的查询时损失；
- 相对主模型的截断增益；
- query_step、return_step、delay；
- query_domain、return_domain，仅用于评价。

`costs.json` 额外报告：

- 主编码器与固定教师编码器的前向调用次数；
- 图模块对 `M_U` 的主模型前向调用次数与图像数；
- 图构建、线性求解、训练调用耗时；
- 总运行耗时、平均图节点/边数；
- 流处理期间的 CUDA 峰值已分配/保留显存。

前向次数通过编码器 forward hook 统计，包含查询、伪标签、记忆概率、训练和特征刷新，
是编码器调用次数，不是严格 FLOPs。相同优化步数不代表相同 FLOPs 或相同耗时。

图模块不使用随机采样，不消耗原 FCRA 监督抽样的随机数。

## 输出

默认输出 `storage/fcra_g/<UTC时间>/`：

| 文件 | 内容 |
|---|---|
| `manifest.json` | FCRA-G、固定 D、三个 seed、batch size、Git 版本、数据/权重哈希、GPU 与状态 |
| `per_seed.csv` | 每次成功任务的准确率、反馈计数及成本字段 |
| `summary.csv`、`summary.md` | 四个 D 的三个 seed 均值 ± 样本标准差 |
| `D*/seed_*/settings.json` | 实际 FCRA/FCRA-G 设置、图与专家超参数、设备与版本 |
| `D*/seed_*/result.json` | 该次完整运行结果，method=FCRA-G |
| `D*/seed_*/batches.csv` | 更新前逐批准确率、查询/返回/pending |
| `D*/seed_*/domains.csv` | 各域准确率及反馈审计 |
| `D*/seed_*/feedback.csv` | 查询步、到期步、返回步、域 |
| `D*/seed_*/query_diagnostics.csv` | PCFR pending 惩罚前后候选数与新查询数 |
| `D*/seed_*/replay.csv` | 完整监督记忆的 need/relevance/score、重放概率与覆盖 |
| `D*/seed_*/calibration.csv` | 教师预测、真值、是否正确、缓存大小 |
| `D*/seed_*/updates.csv` | 每次 SGD 的监督/伪监督损失、alpha、门控统计 |
| `D*/seed_*/graph.csv` | 每批图节点/边/耗时、专家准确率、分歧、修改正确/错误、h 与权重 |
| `D*/seed_*/expert_feedback.csv` | 每个返回事件的查询时专家概率、损失与截断增益 |
| `D*/seed_*/costs.json` | 前向次数、图/求解/训练/总耗时、峰值显存 |

D=0 仍输出 `graph.csv` 与 `expert_feedback.csv` 表头，但图路径未启用。

## 假设与限制

- 源模型局部特征邻域具有一定类别一致性，这是假设，不是已验证事实。
- 首次反馈前没有真实锚点，图平滑只能依赖邻域结构与模型先验；系统性类别混淆
  无法由平滑保证纠正，甚至可能扩大错误。
- 历史锚点只有与当前节点存在图连接时才会影响当前预测；连接错误可能造成负迁移。
- 延迟反馈评分有主动查询选择偏差，不是无偏风险估计。
- 图输出更自信不等于更可靠；`h` 只表示锚点影响力。
- 本入口不修改长期监督损失，不把图结果作为伪标签，D=0 保持原 FCRA 路径。
- 没有无条件不退化保证，必须结合 `graph.csv`、`expert_feedback.csv` 和成本报告
  判断收益来自结构平滑还是真实锚点传播。

本次交付仅做静态代码阅读、时序/索引/求解核对和 Git 差异检查，**没有在本地运行
Python、测试、模型或实验**，不预填准确率，也不宣称 FCRA-G 已经有效。
