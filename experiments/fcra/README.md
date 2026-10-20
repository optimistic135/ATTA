# FCRA：PACS 延迟反馈主实验

本入口实现 Feedback-Calibrated Replay and Adaptation（FCRA）。延迟返回的真值
同时用于纠正当前学生模型，并检验固定源教师在查询时做出的预测，从而局部校准
后续伪监督。实验只运行 FCRA，固定 `D ∈ {0,20,50,100}`，每个 D 使用相同的三个
seed，共 12 次；不运行其他方法、消融、参数搜索或 USE/SKIP 反事实实验。

只新增本目录。复用现有 `experiments.delayed_feedback` 的 oracle、去重聚类、
数据校验、Photo 源权重加载、调度器与进度条。PACS 域顺序、batch size、延迟集合、
seed 数、优化器、早停结构、评价指标和输出汇总方式均与既有 FCRA 入口一致。

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
`nc_increase=1`、`stop_tol=5`、target clustering 和 LE 开启。教师沿用原算法的
固定源教师，聚类默认沿用原配置的 sklearn CPU 路径。

**PACS 使用原算法的损失早停结构，`steps=30` 不是 PACS 的固定迭代上限。**
每个在线 batch 重新创建 SGD，收集五个监督 CE 的窗口均值；窗口均值未刷新历史最低值
超过五个窗口时停止，保持原实现的窗口更新方式。FCRA 的停止量仍是当前监督
minibatch 的未加权平均 CE。监督 minibatch 现在按重放概率从完整记忆重新抽取，
但训练步数仍由这个早停窗口决定，不新增固定步数上限。

每个全局步 t：

1. 用更新前模型预测当前完整 batch，并记录准确率。
2. PCFR 选择新查询；低熵教师分支按原算法维护伪标签记忆。新查询样本在此刻保存
   固定教师的预测类别和归一化特征。
3. oracle 释放到期标签。查询步 q 的标签在 q+D 步预测结束后返回；D=0 也不能影响
   当前批已经记录的预测。
4. 新返回标签加入完整监督记忆；用返回真值生成教师校准事件。事件在学生监督更新
   之前生成。
5. 用更新前模型计算完整监督记忆的纠错需求与当前 batch 局部相关性，得到本在线
   batch 的重放概率。
6. 执行联合更新。每次 SGD 都从完整记忆重新抽取监督样本，并从教师伪标签记忆
   抽样计算局部反馈门控。
7. 刷新查询记忆特征，进入下一 batch。流结束不冲刷 pending queue。

查询选择只接触图像、特征和 stream ID；真值由 oracle 保管至释放。历史重放和
监督更新只读取 `received`，没有未来真值输入。不增加独立的、首次反馈前的无监督
训练阶段。延迟为零时，标签也只能在当前 batch 预测完成后用于校准和训练。

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
高熵增量 KMeans 和簇数增长机制选代表。没有 pending 时退化到原高熵候选门限。
本步即将到期但尚未 release 的标签仍算 pending，与“先查询、后 release”顺序一致。

初始目标簇数 10，每批按原配置增加 1；候选数不足时继承原适配层的截断保护。
原配置 `budgets=100` 不是原适应循环的硬预算，本实现也不新增固定 100 标签截断。

### 完整监督记忆

维护所有标签已返回的查询样本：

\[
\mathcal M_t=\{(x_i,y_i)\mid \text{标签已在 }t\text{ 时刻返回}\}.
\]

低分样本不会被删除。未返回标签不能进入监督记忆，也不能参与监督重放。每次
在线 batch 的监督更新都从整个 \(\mathcal M_t\) 抽样，而不是只围绕固定的少量
高 need 样本训练。

### 纠错需求与当前局部相关性

在更新前，用当前学生模型重新计算记忆样本的真实类别置信度：

\[
N_i=1-p_{\theta_t}(y_i\mid x_i).
\]

设当前完整 batch 为 \(\mathcal B_t\)，学生编码器特征为 \(z\)。对每个历史样本
\(i\)，计算它与当前 batch 每个样本的相似度：

\[
a_{ij}=
\operatorname{clip}
\left(
\frac{1+\cos(z_i,z_j)}{2},0,1
\right).
\]

取最相似的 \(k\) 个当前样本：

\[
R_i=\frac1k\sum_{j\in\operatorname{TopK}(a_{i,:})}a_{ij},
\qquad k=\min(4,|\mathcal B_t|).
\]

采用局部邻域而不是单个 batch 中心，避免不同类别的当前特征被压缩成同一方向。
高 need 只表示当前模型尚未识别好该样本，不直接等同于高监督收益。

### 覆盖与纠错混合抽样

定义纠错偏好：

\[
s_i=N_iR_i.
\]

本在线 batch 的重放概率为：

\[
P_t(i)=
(1-\rho)\frac1{|\mathcal M_t|}
+
\rho
\frac{s_i+\epsilon}
{\sum_{j\in\mathcal M_t}(s_j+\epsilon)},
\]

默认 \(\rho=0.25\)、\(\epsilon=10^{-12}\)。全部分数接近零时使用均匀抽样。
概率在本次在线 batch 内固定，但每次梯度更新都重新从完整记忆抽样，监督
minibatch 大小沿用 `train_bs=16`，有放回抽取。

监督损失采用普通交叉熵，不再额外乘以 need、relevance 或归一化权重：

\[
L_Y=\frac1{|B_Y|}
\sum_{i\in B_Y}
\operatorname{CE}(f_\theta(x_i),y_i).
\]

### 保存待验证的教师预测

使用固定源教师 \(f_T\)。对被查询样本保存：

\[
(x_i,\hat y_i^T,v_i),
\]

其中 \(\hat y_i^T\) 为教师预测，\(v_i\) 为归一化教师特征。标签返回后，在学生
监督更新之前生成校准事件：

\[
c_i=\mathbf1[\hat y_i^T=y_i].
\]

校准缓存只包含已返回真值对应的事件，最多保留最近 512 条。教师预测在查询时
保存，因此即使 D=0，也不会使用当前 batch 的真值重新计算教师预测。

### 局部反馈门控

对一个教师伪标签样本 \(x\)，从校准缓存寻找：

- 教师预测类别相同；
- 教师特征余弦相似度不低于 0.8；

的最多 8 个最近邻，记为 \(\mathcal N_t(x)\)。邻居权重为：

\[
a_i(x)=
\exp\left(
\frac{\cos(v(x),v_i)-1}{\tau}
\right).
\]

定义反馈门控：

\[
r_t(x)=
\frac{\kappa+\sum_{i\in\mathcal N_t(x)}a_i(x)c_i}
{\kappa+\sum_{i\in\mathcal N_t(x)}a_i(x)}.
\]

默认 \(\tau=0.1\)、\(\kappa=4\)。没有合格邻居时，令 \(r_t(x)=1\)，保持原伪监督
行为。邻近区域出现已验证教师错误时，降低该区域的伪监督权重。

该分数是局部反馈门控，不代表整个当前域的正确概率。由于主动查询存在选择偏差，
不能把返回样本的平均正确率直接推广到整个域，也不能把 \(r_t(x)\) 当作域级精度。

### 校准伪监督损失

对低熵教师伪标签集合抽取的 minibatch：

\[
L_P=
\frac1{|B_P|}
\sum_{j\in B_P}
r_t(x_j)
\operatorname{CE}
\left(f_\theta(x_j),\hat y_j^T\right).
\]

分母固定为样本数，不能除以权重之和。这样，错误反馈能够真正降低不可靠伪监督的
绝对影响，而不是把被削弱的样本重新归一化回原权重。每次 SGD 重新抽取伪标签
minibatch 并重新计算门控。

### 联合更新

沿用 SimATTA 的监督与伪监督混合比例：

\[
\alpha_t=
\frac{n_{\mathrm{returned}}}
{n_{\mathrm{returned}}+n_{\mathrm{pseudo}}},
\]

\[
L_t=\alpha_tL_Y+(1-\alpha_t)L_P.
\]

保留原 cold-start 规则。伪标签记忆为空时，沿用真实标签重放回退，此时只使用
\(L_Y\)，日志中 `alpha=1.0`、`has_pseudo=False`。有伪标签但校准缓存为空时，
所有 \(r_t(x)=1\)，退化为原未校准伪监督。

不强制将真实监督梯度投影到伪监督梯度的方向，也不使用冲突投影的诊断字段。
伪监督是否应被削弱，由已经返回的真实反馈提供证据，而不是由两路梯度的几何
关系直接决定。

## 输出与解释

默认输出 `storage/fcra/<UTC时间>/`，已有目录会报错。源文件与旧实验结果不覆盖。

| 文件 | 内容 |
|---|---|
| `manifest.json` | FCRA、固定 D 列表、三个 seed、batch size、Git 版本、数据路径、源权重 SHA256、GPU 与完成状态 |
| `per_seed.csv` | 每次成功任务的准确率、反馈计数；由父进程增量保存 |
| `summary.csv`、`summary.md` | 四个 D 的三个 seed 均值 ± 样本标准差；CSV 含各指标有效 n |
| `D*/seed_*/settings.json` | 实际 SimATTA/FCRA 设置、重放与门控超参数、训练 minibatch、设备、PyTorch/CUDA 版本 |
| `D*/seed_*/result.json` | 该次完整运行结果，method=FCRA |
| `D*/seed_*/batches.csv` | 更新前逐批准确率、查询/返回/pending 数 |
| `D*/seed_*/domains.csv` | 各域在线准确率及反馈审计计数 |
| `D*/seed_*/feedback.csv` | 标签的查询步、到期步、返回步及域 |
| `D*/seed_*/query_diagnostics.csv` | pending 数、惩罚前后候选数、过滤数、新查询数、平均冗余 |
| `D*/seed_*/replay.csv` | 每个有监督更新 batch 的记忆大小、need/relevance/score 统计、重放概率范围、有效样本数、实际抽到的不同样本数与覆盖率 |
| `D*/seed_*/calibration.csv` | 每个返回标签对应的教师预测、真值、是否正确、缓存大小及查询/返回信息 |
| `D*/seed_*/updates.csv` | 每次 SGD 的监督/伪监督损失、alpha、门控均值/最小/最大、被抽样本平均概率与校准缓存大小 |
| `D*_seed_*.log` | 各任务独立日志，包含失败回溯 |

`retrieval.csv` 已由 `replay.csv` 取代：新方法不再只记录固定 top-K 检索集合，
而是记录完整记忆的重放分布和每次更新的实际覆盖。无重放、无校准事件或无训练时，
对应诊断 CSV 仍输出表头。诊断文件和 result 在该次完整流成功结束时写出；失败任务
请先查看日志。某次失败后停止分派新任务，允许已运行任务结束并保存成功结果，
manifest 标记 failed；不会把不满三个 seed 的均值作为完整汇总。

准确率与比例以百分数输出。Stream accuracy 按各批实际样本数加权；Domain macro
accuracy 是 Art、Cartoon、Sketch 三域准确率的等权平均。跨域返回比例分母为实际
已返回数，不是查询总数；无返回时为 N/A。比例先按 seed 计算，再求均值和样本
标准差。不同 D 的主动查询与训练轨迹可不同，三个 seed 仅反映流顺序与适应随机性，
不包含源模型重训练方差。FCRA 会改变查询数，后续比较旧方法时还须同时检查标注成本。

本次只实现并交付代码，按要求**没有在本地执行 Python、测试、模型推理或实验**。
检查仅限代码阅读、时序/梯度/索引追踪和 Git 差异检查。没有提供实测准确率，也没有
宣称 FCRA 已经有效；本入口只产出 FCRA 的结果，不额外验证基线、消融或负收益事件
比例。
