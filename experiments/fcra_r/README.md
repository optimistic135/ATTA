# FCRA-R：监督锚点检索与局部残差传播

新增独立入口 `experiments.fcra_r`，保留现有 FCRA、`fcra_dual` 和 `fcra_g`。
本方法在完整记忆 FCRA 的监督学习路径之外，增加一条不更新网络参数的预测校正路径：

1. 已返回标签继续用于完整记忆重放，维护长期分类能力。
2. 当前样本只从真实已返回标签中检索监督锚点，计算当前模型的类别残差，
   在局部邻域内传播，并按延迟反馈选择校正强度。

两条路径相互分离。残差校正不更新编码器或分类头，不把校正概率作为伪标签，
不写入真实标签记忆，也不消耗监督优化步数。

本实现对应 `experiments/fcra_g/ANALYSIS_DESIGN.md` 中的条件化候选 FCRA-R。
现有 FCRA-G 日志证据尚不足以证明图方向或强度是主要瓶颈，因此本入口是待验证实现，
不宣称已经解决该问题，也不保证优于现有 FCRA。

## 服务器运行

首次验证复用一份已完成的完整记忆 FCRA sweep 作为固定协议：查询 ID、查询时间、
anchor 顺序和逐批监督更新次数从参考结果重放。参考目录必须包含 `manifest.json`
和 12 个 `D*/seed_*` 任务，不是单个 D 或 seed。

```bash
conda activate df-atta

python -u -m experiments.fcra_r.run \
  --dataset-root datasets \
  --reference-dir /path/to/completed-fcra-results \
  --gpus 0 1 \
  --seeds 110 123 456 \
  --output-dir storage/fcra_r/pacs_bs32_seeds110_123_456
```

参考目录结构：

```text
completed-fcra-results/
├── manifest.json
├── D0/seed_.../
│   ├── settings.json
│   ├── result.json
│   ├── batches.csv
│   ├── feedback.csv
│   └── updates.csv
├── D20/seed_.../
├── D50/seed_.../
└── D100/seed_.../
```

参考必须是更新后的完整记忆 FCRA（`settings.json` 含 `replay_mix` 等字段），不能是
早期硬 Top-16 或梯度投影版本。默认复用参考 manifest 的三个 seed；传入 `--seeds`
时必须与参考完全一致。程序会校验参考 manifest、数据流批次数、查询 anchor 顺序、
到期/返回时间、监督更新序列、源权重 SHA256，并在 worker 启动后再次校验参考文件
哈希。参考 trace 没有图像内容指纹，因此仍必须使用同一份 PACS 数据和相同的域内
样本顺序；“批次数相同”不等于图像内容相同。

### 单卡

```bash
python -u -m experiments.fcra_r.run \
  --dataset-root datasets \
  --reference-dir /path/to/completed-fcra-results \
  --gpu-idx 0 \
  --seeds 110 123 456 \
  --output-dir storage/fcra_r/pacs_bs32_seeds110_123_456
```

### 指定物理 GPU

```bash
CUDA_VISIBLE_DEVICES=2,3 python -u -m experiments.fcra_r.run \
  --dataset-root datasets \
  --reference-dir /path/to/completed-fcra-results \
  --gpus 0 1 \
  --seeds 110 123 456 \
  --output-dir storage/fcra_r/pacs_bs32_seeds110_123_456
```

`--gpu-idx` 与 `--gpus` 互斥。默认输出 `storage/fcra_r/<UTC时间>/`，已有目录
会报错。默认检查点目录与现有 PACS 入口相同：

```text
storage/checkpoints/round1/PACS/[2, 0, 1, 3]/ResNet18_3l_meanpool_0dp_Truefzbn/0.0001lr_0.0wd/ERM_no_param/
```

需要已有 `datasets/PACS/{art_painting,cartoon,photo,sketch}`。不下载数据、不重训
源模型、不写入源检查点。

## 固定实验设置

- PACS Photo → Art → Cartoon → Sketch；
- online batch size = 32；
- D = 0、20、50、100；
- 参考 FCRA 的三个 seed；
- 参考 FCRA 的查询 ID、查询时间和逐批监督更新次数；
- 原 FCRA 的完整记忆重放、教师伪监督门控、SGD lr=1e-3、momentum=0.9、
  train minibatch=16 和 SimATTA 配置；
- 评价指标：三个目标域准确率、stream accuracy、domain macro accuracy、
  queried/returned/cross-domain/pending。

参考 trace 固定的是逐批更新次数，不保证每次 minibatch 的随机抽样与旧运行逐次
相同；监督损失、完整记忆重放、伪监督门控和训练 minibatch 大小保持不变。
残差校正不向监督模块分配或借用优化步数。

## 固定超参数

以下值在验证前固定，不通过测试准确率、已有 utility 表或结果文件调参。

| 名称 | 默认值 | 作用 |
|---|---:|---|
| `k` / `retrieval_neighbors` | 8 | 每个当前样本最多检索的已返回锚点数 |
| `tau_s` / `retrieval_similarity` | 0.8 | 监督锚点余弦相似度阈值 |
| `tau` / `residual_temperature` | 0.1 | 局部残差邻居权重温度 |
| `kappa` / `residual_prior_count` | 4.0 | 支持强度先验计数 |
| `beta_1,beta_2` | 0.25, 0.50 | 两个固定候选校正强度 |
| `K` / `feedback_cache_size` | 512 | 最近已评分反馈事件上限 |
| `tau_f` / `feedback_similarity` | 0.8 | 局部反馈邻居相似度阈值 |
| `tau_g` / `feedback_temperature` | 0.1 | 局部反馈权重温度 |
| `tau_age` / `feedback_age` | 50.0 | 反馈事件时间衰减尺度（step） |
| `kappa_f` / `feedback_prior_count` | 4.0 | 局部反馈证据先验计数 |
| `eta` / `feedback_eta` | 1.0 | 局部证据对候选权重的影响强度 |
| `pi` / `feedback_prior` | 0.6, 0.3, 0.1 | beta=0、beta1、beta2 的先验权重 |
| `g_max` / `gain_clip` | 1.0 | 反馈增益截断 |
| `epsilon` | 1e-12 | 数值稳定项 |

## 1. 监督锚点检索

维护完整已返回标签记忆：

\[
\mathcal M_t=\{(x_i,y_i)\mid t_i^{\mathrm{return}}<t\}.
\]

未返回标签不进入监督锚点集合，低相似度样本也不会从完整记忆中删除。当前样本只从
\(\mathcal M_t\) 检索监督锚点，避免真实标签与无标签节点竞争邻居名额。

使用冻结源编码器构建稳定检索特征：

\[
v(x)=\operatorname{normalize}(E_0(x)).
\]

对当前样本 \(x\)，按余弦相似度检索至多 \(k\) 个已返回样本：

\[
\mathcal N_t(x)=\operatorname{TopK}_{i\in\mathcal M_t}\cos(v(x),v(x_i)).
\]

仅保留：

\[
\mathcal N_t^+(x)=\{i\in\mathcal N_t(x)\mid
\cos(v(x),v(x_i))\geq\tau_s\}.
\]

## 2. 当前模型残差

在当前 batch 任何参数更新之前，用同一版本的监督主模型计算：

\[
p_t(x)=\operatorname{softmax}(f_{\theta_t}(x)).
\]

对检索到的已返回样本计算类别残差：

\[
r_{i,t}=e_{y_i}-p_t(x_i).
\]

残差同时包含真实类别概率不足的正分量和错误类别概率过高的负分量。残差随主模型
变化重新计算。每个 batch 对所有完整记忆锚点只计算一次当前模型概率并复用；不长期
缓存旧模型残差。

## 3. 局部残差方向

对可信监督邻居定义非负权重：

\[
a_i(x)=\exp\left(\frac{\cos(v(x),v(x_i))-1}{\tau}\right),
\qquad i\in\mathcal N_t^+(x).
\]

局部支持量：

\[
s_t(x)=\sum_{i\in\mathcal N_t^+(x)}a_i(x).
\]

局部残差方向：

\[
\bar r_t(x)=
\begin{cases}
\displaystyle \frac{\sum_i a_i(x)r_{i,t}}{s_t(x)}, & s_t(x)>0,\\
0, & s_t(x)=0.
\end{cases}
\]

## 4. 支持强度与方向一致性

监督支持强度：

\[
\rho_t(x)=\frac{s_t(x)}{\kappa+s_t(x)},\qquad \kappa>0.
\]

邻居残差加权平均幅度：

\[
m_t(x)=
\begin{cases}
\displaystyle \frac{\sum_i a_i(x)\|r_{i,t}\|_2}{s_t(x)}, & s_t(x)>0,\\
0, & s_t(x)=0.
\end{cases}
\]

方向一致性：

\[
\xi_t(x)=\frac{\|\bar r_t(x)\|_2}{\epsilon+m_t(x)},
\qquad 0\leq\xi_t(x)\leq1.
\]

最终局部修正量：

\[
c_t(x)=\rho_t(x)\xi_t(x)\bar r_t(x).
\]

方向一致性只描述邻居修正是否相容，不解释为预测正确率。

## 5. 概率残差校正

固定候选强度集合：

\[
\mathcal B=\{0,\beta_1,\beta_2\},
\qquad 0<\beta_1<\beta_2\leq1.
\]

对每个候选强度：

\[
p_t^{(\beta)}(x)=
\Pi_{\Delta^{C-1}}\left[p_t(x)+\beta c_t(x)\right],
\]

其中 \(\Pi_{\Delta^{C-1}}\) 为欧氏概率单纯形投影。\(\beta=0\) 对应原主模型。
不对校正结果再次执行 softmax。没有可信邻居、没有已返回标签或修正量为零时，
所有候选回退为 \(p_t(x)\)。

## 6. 查询时封存与返回时评分

样本被查询时保存各候选在查询时刻的完整概率：

\[
\{p_{q_i}^{(\beta)}(x_i):\beta\in\mathcal B\},
\]

同时保存固定源特征 \(v_i\) 和查询时间。封存后不再修改。

标签返回后，在加入监督记忆或参与参数更新之前，计算：

\[
\ell_i^{(\beta)}
=-\log\max\left(p_{q_i}^{(\beta)}(y_i\mid x_i),\epsilon\right),
\]

\[
g_i^{(\beta)}
=\operatorname{clip}\left(
\ell_i^{(0)}-\ell_i^{(\beta)},-g_{\max},g_{\max}
\right).
\]

每个返回事件只评分一次。评分使用查询时封存预测，不使用吸收该标签后重新生成的
预测。实现通过延迟流释放钩子在 `received.update` 之前完成评分。

## 7. 局部反馈驱动的强度选择

保存最近 \(K\) 条已评分返回事件：

\[
\mathcal F_t=\{(v_i,t_i^{\mathrm{return}},g_i^{(\beta)})\}.
\]

对当前样本检索局部反馈邻居：

\[
b_i(x)=
\mathbf1[\cos(v(x),v_i)\geq\tau_f]
\exp\left(\frac{\cos(v(x),v_i)-1}{\tau_g}\right)
\exp\left(-\frac{t-t_i^{\mathrm{return}}}{\tau_{\mathrm{age}}}\right).
\]

对每个候选强度：

\[
G_t^{(\beta)}(x)=
\frac{\sum_i b_i(x)g_i^{(\beta)}}{\kappa_f+\sum_i b_i(x)},
\qquad \kappa_f>0.
\]

由固定先验和局部证据得到权重：

\[
w_t^{(\beta)}(x)=
\frac{\pi_\beta\exp(\eta G_t^{(\beta)}(x))}
{\sum_{\beta'\in\mathcal B}\pi_{\beta'}\exp(\eta G_t^{(\beta')}(x))}.
\]

最终输出：

\[
p_t^{\mathrm{out}}(x)=\sum_{\beta\in\mathcal B}
w_t^{(\beta)}(x)p_t^{(\beta)}(x).
\]

原主模型候选保留非零先验。没有局部反馈时使用固定先验；反馈显示较强校正有害时，
降低其权重。本批新返回事件只能影响后续 batch。

## 8. 与监督学习路径分离

已返回标签继续进入完整监督记忆，沿用原有监督重放与教师伪监督更新。残差校正路径：

- 不更新编码器或分类头；
- 不将校正概率作为伪标签；
- 不将残差传播结果写入真实标签记忆；
- 不消耗监督优化步数；
- 不根据真实域边界重置或选择参数。

## 每批时序

1. 使用已有已返回标签检索监督锚点；参考模式同时重放本步查询 anchor。
2. 计算当前主模型预测和锚点当前概率/残差，每批只算一次锚点概率。
3. 聚合局部修正量 `c` 和候选概率 `p^(beta)`。
4. 使用已有反馈事件选择校正强度并生成本批最终预测。
5. 为本批查询样本封存各候选预测、固定源特征和当前权重。
6. 释放到期标签；在加入监督记忆前用封存预测评分反馈事件。
7. 将新返回样本加入完整监督记忆；按参考逐批次数执行原 FCRA 监督与伪监督训练。
8. 进入下一批。

D=0 仍重放参考查询与监督更新次数，但绕过残差校正、候选封存和反馈评分，直接使用
原 FCRA 预测与训练路径。

评价真值只用于在线指标和 `residual.csv` 的离线诊断，不进入锚点检索、残差计算、
候选权重或训练。

## 诊断与计算开销

`residual.csv` 每个 batch 记录：

- 主模型、最终输出、两个候选的准确率；
- 主模型与各候选/最终输出的分歧率；
- 校正改变的预测数、修改后正确/错误；
- 平均支持量、无支持比例、`rho`、`xi`、修正量范数；
- 平均检索到的监督邻居数、局部反馈邻居数、反馈缓存大小；
- 三个候选权重；
- 传播耗时。

`residual_feedback.csv` 每个返回事件记录：

- 查询时三个候选的真实类别概率、最终输出概率；
- 查询时三个候选与最终输出的预测类别及是否正确；
- 三个候选查询时损失；
- 相对主模型的截断增益；
- query_step、return_step、delay；
- query_domain、return_domain，仅用于评价；
- 查询时三个候选权重。

`costs.json` 记录：

- 主模型/教师完整前向与编码器前向调用次数；
- 锚点当前概率的模型前向调用次数与图像数；
- 残差传播、训练、总运行耗时；
- 平均检索邻居数、反馈邻居数、支持量、修正量范数；
- 流处理期间 CUDA 峰值已分配/保留显存。

前向次数通过 forward hook 统计，包含查询、伪标签、锚点概率、训练和特征刷新。
相同优化步数不代表相同 FLOPs 或相同耗时。残差模块不使用随机采样，不消耗监督
抽样的随机数。

## 输出

默认输出 `storage/fcra_r/<UTC时间>/`：

| 文件 | 内容 |
|---|---|
| `manifest.json` | FCRA-R、参考目录/哈希、固定 D、seed、batch size、Git 版本、数据/权重哈希、GPU 与状态 |
| `per_seed.csv` | 每次成功任务的准确率、反馈计数、参考/实际更新次数及成本字段 |
| `summary.csv`、`summary.md` | 四个 D 的三个 seed 均值 ± 样本标准差 |
| `D*/seed_*/settings.json` | 实际 FCRA/FCRA-R 设置、参考哈希、残差超参数、设备与版本 |
| `D*/seed_*/result.json` | 该次完整运行结果，method=FCRA-R |
| `D*/seed_*/reference_budget.csv` | 参考逐步查询数和监督更新额度 |
| `D*/seed_*/batches.csv` | 更新前逐批准确率、查询/返回/pending |
| `D*/seed_*/domains.csv` | 各域准确率及反馈审计 |
| `D*/seed_*/feedback.csv` | 查询步、到期步、返回步、域 |
| `D*/seed_*/query_diagnostics.csv` | 参考查询计划的新查询数；候选数/冗余字段在参考模式下为空 |
| `D*/seed_*/replay.csv` | 完整监督记忆的 need/relevance/score、重放概率与覆盖 |
| `D*/seed_*/calibration.csv` | 教师预测、真值、是否正确、缓存大小 |
| `D*/seed_*/updates.csv` | 每次 SGD 的监督/伪监督损失、alpha、门控统计 |
| `D*/seed_*/residual.csv` | 每批支持强度、方向一致性、修正量、候选准确率、分歧、权重与传播耗时 |
| `D*/seed_*/residual_feedback.csv` | 每个返回事件的查询时候选概率、损失、截断增益与权重 |
| `D*/seed_*/costs.json` | 前向次数、传播/训练/总耗时、平均检索/反馈邻居、峰值显存 |

D=0 仍输出 `residual.csv` 与 `residual_feedback.csv` 表头，但残差路径未启用。
运行结束会逐步核对实际监督更新次数与参考 `updates.csv` 一致，并核对总查询/返回
事件与参考反馈计划；不一致则任务失败。

## 待验证假设

- H1：监督锚点检索能提供比无标签图邻居更直接的类别修正信号。
- H2：局部残差同时修正真实类别不足和错误类别过高，比单向类别投票更稳健。
- H3：方向一致性 `xi` 能区分相容与冲突的邻居残差。
- H4：基于查询时封存预测的延迟反馈能降低持续有害的校正强度。

这些假设需要分别用 `residual.csv`、`residual_feedback.csv` 和成本记录验证，
不能用整体准确率提升替代全部验证。

## 假设与限制

- 源模型局部特征邻域具有一定类别一致性，这是假设，不是已验证事实。
- 首次反馈前没有已返回锚点，残差校正只能回退主模型或使用极弱候选；该阶段没有
  新的真实类别信息。
- 主动查询存在选择偏差，局部反馈证据不是当前全部数据上的无偏风险估计。
- 历史锚点只有与当前样本相似度达到阈值时才会参与校正；相似度连接错误可能造成
  负迁移。
- 方向一致性只描述残差方向是否相容，不是预测正确率。
- 没有无条件不退化保证；必须结合诊断和成本报告判断收益来自何处。
- 参考 trace 只校验批次数、时间、anchor 顺序、配置和文件哈希，没有图像内容指纹。
- 本入口不修改长期监督损失，不把残差结果作为伪标签，D=0 保持原 FCRA 路径。

本次交付仅做静态代码阅读、时序/索引/残差公式核对和 Git 差异检查，**没有在本地
运行 Python、测试、模型或实验**，不预填准确率，也不宣称 FCRA-R 已经有效。
