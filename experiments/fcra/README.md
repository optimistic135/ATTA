# FCRA：PACS 延迟反馈主实验

本入口实现 Feedback-Calibrated Retrieval and Adaptation（FCRA），包含 PCFR
查询覆盖与历史检索，以及 FACA 反馈加权与冲突梯度修正。只运行 FCRA，固定
`D ∈ {0,20,50,100}`，每个 D 使用相同的三个 seed，共 12 次；不运行其他方法、
消融、参数搜索或 USE/SKIP 反事实实验。

只新增本目录。复用现有 `experiments.delayed_feedback` 的 oracle、去重聚类、
数据校验、Photo 源权重加载、调度器与进度条。

## 服务器运行

在已安装原项目依赖的服务器上，从仓库根目录执行：

```bash
conda activate df-atta
python -u -m experiments.fcra.run \
  --dataset-root datasets \
  --gpus 0 1
```

每张 GPU 同时运行一个独立任务，空闲卡自动领取下一个任务。终端显示总任务进度，
以及每张卡当前 D、seed、域、batch 进度、累计在线准确率、returned 和 pending。
重定向输出时沿用现有逐行进度记录。

单卡使用 `--gpu-idx 0`；不传设备参数默认单卡 0。GPU 编号是
`CUDA_VISIBLE_DEVICES` 后的逻辑编号。单卡、双卡都要求可见 CUDA GPU。
例如物理卡 2、3 对应：

```bash
CUDA_VISIBLE_DEVICES=2,3 python -u -m experiments.fcra.run \
  --dataset-root datasets --gpus 0 1
```

显式复用指定的三个 seed 并指定新输出目录：

```bash
python -u -m experiments.fcra.run \
  --dataset-root datasets \
  --seeds 110 123 456 \
  --gpus 0 1 \
  --output-dir storage/fcra/pacs_bs32_seeds110_123_456
```

`--gpu-idx` 与 `--gpus` 互斥；`--seeds` 必须提供三个不同的整数。默认抽取三个随机
seed，写入 manifest 后供所有 D 共用。可选 `--num-workers`，默认 0，与现有延迟入口
一致；该参数只改变数据加载进程数。

需要已有 `datasets/PACS/{art_painting,cartoon,photo,sketch}`。默认检查点目录：

```text
storage/checkpoints/round1/PACS/[2, 0, 1, 3]/ResNet18_3l_meanpool_0dp_Truefzbn/0.0001lr_0.0wd/ERM_no_param/
```

也可以通过 `--checkpoint-dir` 指定同时含 `encoder_2.pth` 和 `fc_2.pth` 的目录。
不自动下载数据、不重新训练源模型、不写入源检查点。原模型构造使用 torchvision
ImageNet 初始化，离线服务器须已有相应预训练权重缓存。

## 实验设置与反馈时序

Photo 源模型，Art → Cartoon → Sketch 连续目标流。各域按 seed 随机无放回遍历，
保留域内不足 32 张的最后一批；跨 D 使用相同流顺序。模型状态、已返回监督、查询
记忆、教师伪标签记忆和 pending queue 跨域保留。

原 PACS SimATTA YAML 的在线 batch size 为 100，本实验明确覆盖为 **32**，与当前
PACS delayed-feedback 入口一致。标准 PACS 目标流在 batch=100 时不足 100 批，D=100
将没有流内返回，因此四个 D 统一使用 32。结果不应与 batch=100 的延迟结果混报。

其余继承 PACS SimATTA 配置：ResNet18、冻结 BN、dropout=0；SGD 学习率 `1e-3`、
momentum=0.9、训练 minibatch=16、`eh=1e-2`、`el=1e-4`、`cold_start=0`、
`nc_increase=1`、`stop_tol=5`、target clustering 和 LE 开启。
教师沿用原算法的固定源教师，聚类默认沿用原配置的 sklearn CPU 路径。

**PACS 使用原算法的损失早停结构，`steps=30` 不是 PACS 的固定迭代上限。**
每个在线 batch 重新创建 SGD，收集五个监督 CE 的窗口均值；窗口均值未刷新历史最低值
超过五个窗口时停止，保持原实现的窗口更新方式。FCRA 的停止量仍是检索监督样本的
未加权平均 CE，训练梯度使用下述加权与投影。未使用 utility_prediction 的固定 30 步循环。

每个全局步 t：

1. 用更新前模型预测当前完整 batch，并记录准确率。
2. PCFR 选择新查询；低熵教师分支按原算法维护伪标签记忆。
3. oracle 释放到期标签。查询步 q 的标签在 q+D 步预测结束后返回；D=0 也不能影响
   当前批已经记录的预测。
4. 已有返回标签时，检索历史监督并执行 FACA；否则不执行依赖真实标签的更新。
5. 刷新查询记忆特征，进入下一 batch。流结束不冲刷 pending queue。

查询选择只接触图像、特征和 stream ID；真值由 oracle 保管至释放。历史检索和监督
更新只读取 `received`，没有未来真值输入。不增加独立的、首次反馈前的无监督训练阶段。

## 固定规则与公式

下面是对方法描述未指定部分的一组明确工程实现，不代表已调优或已验证的最优公式。
超参数仅固定一组默认值，不通过测试准确率或已有 utility 表拟合。

### PCFR：Pending-aware Query

对当前图像 x，使用学生模型预测熵 H(x)，类别数 C，以及当前特征 z(x)。
pending 集合 P 只含已请求且尚未返回的样本。定义：

\[
R_P(x)=\begin{cases}
\max(0,\max_{j\in P}\cos(z(x),z_j)), & P\ne\varnothing,\\
0, & P=\varnothing,
\end{cases}
\qquad
S_q(x)=\frac{H(x)}{\log C}-0.25R_P(x).
\]

保留同时满足 `H(x) >= eh` 与 `S_q(x) >= eh/log(C)` 的图像，再交给原 SimATTA 的
高熵增量 KMeans 和簇数增长机制选代表。即把 pending 惩罚用于候选覆盖筛选，
由原聚类处理本批样本的覆盖。没有 pending 时退化到原高熵候选门限。

pending 的特征在前一轮模型更新后刷新；首次返回前模型未更新，缓存仍有效。
本步即将到期但尚未 release 的标签仍算 pending，与“先查询、后 release”顺序一致。
同一批新代表不额外按 pending 惩罚再筛选；重复代表由继承的 stream-ID 去重逻辑合并，
簇权重相加，oracle 继续拒绝重复查询。

初始目标簇数 10，每批按原配置增加 1；候选数不足时继承原适配层的截断保护。
筛选可以减少查询次数，最终查询成本由实际 oracle 记录报告。原配置 `budgets=100`
不是原适应循环的硬预算，本实现也不新增固定 100 标签截断。

### PCFR：Error-aware Historical Retrieval

每个有监督反馈的在线 batch，在更新前用当前编码器重新计算全部已返回样本的
特征与真实类别置信度。设当前完整 batch 的平均特征为 \(\bar z_t\)：

\[
N_i=1-p_{\theta_t}(y_i\mid x_i),\qquad
R_i=\operatorname{clip}\left(\frac{1+\cos(z_i,\bar z_t)}{2},0,1\right),
\qquad s_i=N_iR_i.
\]

按 s_i 降序检索 `K=min(train_bs, 已返回样本数)`，即最多 16 个；同分按 anchor ID
升序决定。不按域过滤，跨域样本只要当前得分高仍可检索。样本与真值保留在全局缓冲区，
未被本批选中不等于删除，以后仍可再次检索。

### FACA：Feedback Calibration

对本批检索集合 \(\mathcal R_t\)，计算：

\[
w_i=\frac{s_i}{|\mathcal R_t|^{-1}\sum_{j\in\mathcal R_t}s_j},
\qquad L_T=\frac1m\sum_{i\in B_T}w_i\,\operatorname{CE}(f_\theta(x_i),y_i).
\]

若检索分数均值不超过 `1e-12`，权重统一设为 1。权重在本次在线 batch 更新循环中固定，
下一在线 batch 重新计算。重放沿用原 `InfiniteDataLoader` 的均匀有放回抽样和最后小批
行为，所以抽样 minibatch 的权重均值不必恰为 1，其期望仍对应检索集合的加权目标。

### FACA：Conflict Adaptation

参考目标 L_ref 使用 SimATTA 的低熵固定教师伪标签记忆 minibatch 的交叉熵；它是可在线
获得的参考代理，可能包含历史域样本，并非当前 batch 的真实标签目标。
分别计算全部可训练参数上的梯度：

\[
g_{\mathrm{ref}}=\nabla_\theta L_{\mathrm{ref}},\quad
g_T=\nabla_\theta L_T,\quad
\widehat U=\eta\,g_{\mathrm{ref}}^\top g_T.
\]

为避免逐个样本对全网络反向传播，在检索监督 minibatch 的**加权聚合梯度**上做冲突修正：

\[
g_T^{\mathrm{safe}}=
\begin{cases}
g_T-\dfrac{g_{\mathrm{ref}}^\top g_T}{\|g_{\mathrm{ref}}\|^2}g_{\mathrm{ref}},
&g_{\mathrm{ref}}^\top g_T<0\;\land\;\|g_{\mathrm{ref}}\|^2>10^{-12},\\
g_T,&\text{otherwise}.
\end{cases}
\]

沿用 SimATTA 的目标混合比例，而不是用检索子集大小改变它：

\[
\alpha=\frac{n_{\mathrm{returned}}}{n_{\mathrm{returned}}+n_{\mathrm{pseudo}}},
\qquad g=(1-\alpha)g_{\mathrm{ref}}+\alpha g_T^{\mathrm{safe}}.
\]

保留原 cold_start 对 alpha 的限制。SGD 通过这个合并梯度更新，momentum 沿用 0.9。
若伪标签记忆为空，原适配层会回退到真实标签重放；FCRA 将它落实为校准后的监督梯度
使用一次，不把真实标签复制出一条未经校准的“参考分支”，并跳过冲突投影。
此时日志中 alpha 为原回退比率 0.5（默认 cold_start=0），有效梯度为 g_T。
若有伪标签但参考梯度范数为零，仍按原 alpha 混合，只跳过投影。

未使用参数的梯度按零向量处理。非有限梯度会明确报错，避免输出看似成功的 NaN 结果。
投影只约束当前原始损失梯度：数值误差、SGD 动量、有限学习率和后续模型变化都意味着
它不保证实际参数步或未来准确率必然改善。`first_order_improvement` 是参考损失的一阶代理，
不是已测量的 USE−SKIP utility；不使用旧 utilities.csv 或未来准确率计算权重。

## 输出与解释

默认输出 `storage/fcra/<UTC时间>/`，已有目录会报错。源文件与旧实验结果不覆盖。

| 文件 | 内容 |
|---|---|
| `manifest.json` | FCRA、固定 D 列表、三个 seed、batch size、Git 版本、数据路径、源权重 SHA256、GPU 与完成状态 |
| `per_seed.csv` | 每次成功任务的准确率、反馈计数；由父进程增量保存 |
| `summary.csv`、`summary.md` | 四个 D 的三个 seed 均值 ± 样本标准差；CSV 含各指标有效 n |
| `D*/seed_*/settings.json` | 实际 SimATTA/FCRA 设置、训练 minibatch、设备、PyTorch/CUDA 版本 |
| `D*/seed_*/result.json` | 该次完整运行结果，method=FCRA |
| `D*/seed_*/batches.csv` | 更新前逐批准确率、查询/返回/pending 数 |
| `D*/seed_*/domains.csv` | 各域在线准确率及反馈审计计数 |
| `D*/seed_*/feedback.csv` | 标签的查询步、到期步、返回步及域 |
| `D*/seed_*/query_diagnostics.csv` | pending 数、惩罚前后候选数、过滤数、新查询数、平均冗余 |
| `D*/seed_*/retrieval.csv` | 每批检索样本 ID、已返回标签、need、relevance、score、weight 和标签返回信息 |
| `D*/seed_*/updates.csv` | 每次 SGD 的梯度夹角、投影前后内积、范数、冲突标志、损失、alpha 与一阶代理 |
| `D*_seed_*.log` | 各任务独立日志，包含失败回溯 |

无检索或无训练时，对应诊断 CSV 仍输出表头。诊断文件和 result 在该次完整流成功结束时
写出；失败任务请先查看日志。某次失败后停止分派新任务，允许已运行任务结束并保存成功
结果，manifest 标记 failed；不会把不满三个 seed 的均值作为完整汇总。

准确率与比例以百分数输出。Stream accuracy 按各批实际样本数加权；Domain macro
accuracy 是 Art、Cartoon、Sketch 三域准确率的等权平均。跨域返回比例分母为实际已返回数，
不是查询总数；无返回时为 N/A。比例先按 seed 计算，再求均值和样本标准差。
不同 D 的主动查询与训练轨迹可不同，三个 seed 仅反映流顺序与适应随机性，不包含源模型
重训练方差。FCRA 会改变查询数，后续比较旧方法时还须同时检查标注成本。

本次只实现并交付代码，按要求**没有在本地执行 Python、测试、模型推理或实验**。
检查仅限代码阅读、时序/梯度/索引追踪和 Git 差异检查。没有提供实测准确率，也没有宣称
FCRA 已经有效；本入口只产出 FCRA 的结果，不额外验证基线、消融或负收益事件比例。
