# 实验一：PACS 历史标签监督价值预测

仅增加本目录文件。复用 `experiments.delayed_feedback` 的 oracle、去重聚类选样、
数据验证及源权重加载，不改变任何现有入口。只使用 PACS 的 Photo 源模型与
Art → Cartoon → Sketch domain-wise 数据流；不运行 random stream。

## 服务器运行

激活原项目环境，从仓库根目录执行：

```bash
conda activate df-atta
python -u -m experiments.utility_prediction.run \
  --dataset-root /home/user3/chenc/workspace/git-rep/ATTA/datasets \
  --gpu-idx 0 \
  --seed 110 \
  --output-dir storage/utility_prediction/pacs_seed110
```

一条命令顺序运行 d=0、20、50、100，并生成汇总和预测分析。
必须已有 `datasets/PACS/{art_painting,cartoon,photo,sketch}` 与原 PACS round1
`encoder_2.pth`、`fc_2.pth`。默认权重目录由现有 delayed-feedback 的 PACS 配置给出；
不同权重可加 `--checkpoint-dir /path/to/checkpoints`。
不会下载数据、重新训练源模型或修改源权重。

默认使用 scikit-learn CPU 聚类，模型在指定 GPU 上运行，num_workers=0。
这是单进程顺序实验，不增加双卡调度或恢复机制；不依赖 Triton。
输出目录已存在时报错，避免覆盖。换 seed 时使用新的输出目录。

可选参数：`--delays 0 20 50 100`、`--horizon 20`、`--steps 30`、`--batch-size 32`。
默认 seed=110；每个 d 都从同一源权重、同一 seed 重新开始。

**为什么 batch size=32？** 原 PACS batch size=100 时目标流不足100批，d=100
的标签在流结束前不能返回，更无法观察未来20批。四个延迟统一改为32，以保证有窗口，
保留各域末批，不重复数据、不循环数据流。若自定义数据仍不足 d+horizon+1 批，直接报错。
不同 batch size 下的延迟代表不同数量的样本，不能与原 batch size=100 的实验混同。

## 实验协议

1. 全局步从0开始，跨域连续计时。先预测当前 batch，再查询与返回标签，最后更新。
   在 t 查询的标签于 t+d 批预测结束后返回；d=0 也不影响当前批的已记录预测。
2. 运行一条真实的 delayed-SimATTA 轨迹，记录选样、到期返回与固定 teacher 的低熵伪标签。
   未到期真值不可进入训练；不冲刷流末等待队列。查询数量机制沿用现有适配层，
   不把 `atta.budgets=100` 误当作查询硬上限。
3. 每个返回批次在更新前保存一份当前模型。每个返回的独立图片/标签是一个事件；
   同批多个标签逐一消融，其他同批返回标签作为共同背景。
4. 对同一事件复制该模型，分别运行 USE/SKIP。两者使用完全相同的未来输入、
   已记录查询/标签返回计划、固定 teacher 伪标签、重放索引、随机数和优化器设置。
   USE 使用该标签；SKIP 在当前和后续重放中将该标签的真值损失置零，
   不重抽其他样本，不改变损失分母或混合系数。若无低熵样本，源分支回退到真值重放时，
   也会同时屏蔽该标签，防止绕过 SKIP。该图像仍可作为无标签聚类记忆或伪标签样本。
5. 返回时第一次更新的目标 minibatch 强制包含被测标签一次，USE/SKIP 两者索引相同，
   避免某个标签恰好未被抽到而没有实际参与 USE。之后正常随机重放。
6. 使用 SimATTA 的 SGD（momentum=0.9）、学习率、低熵分支、cold_start 混合规则，
   每批固定30次更新。**这是有意替代原 PACS 的损失早停**：USE/SKIP 的损失不同，
   依损失停止会混入更新次数差异。优化器沿用原实现每批重新建立。
7. 未来 t+1,…,t+20 批先预测再更新。逐批计算准确率，然后按 batch 等权平均：
   U = mean(A_USE,batch) − mean(A_SKIP,batch)。准确率及 U 存为0–1；汇总U转换为百分点。
   最后不足20批的事件保留原始观测，但标记 `complete_window=False`，不进入主要统计或预测器。

**解释边界：** U 是在真实轨迹固定的未来选样/反馈计划下，该标签参与监督的受控收益。
两个分支不各自重新主动选样；否则“其余训练一致”不成立。
它不是重新运行两套主动策略得到的完整策略收益。
未来真值仅用于离线评分和已到期 oracle 反馈，不参与指标计算或提前训练。
同批事件不是独立试验，窗口也有重叠；本实验不据此给出独立样本显著性结论。

## 可观察指标与预测分析

`return_features.csv`、`utilities.csv` 每标签记录：

| 字段 | 含义 |
|---|---|
| delay/query_step/return_step | 查询延迟及全局查询、返回批次 |
| query_domain/return_domain | 查询、返回所在域 |
| query_entropy/return_entropy | 同一图像查询与返回时的预测熵 |
| return_true_probability | 返回时、更新前，模型赋予已返回真实类别的概率 |
| need | 连续纠错需求：1 - return_true_probability，范围为[0,1] |
| need_binary | 旧二值纠错指标：返回时预测类别是否与真值不同，保留供对照 |
| prediction_changed/prediction_js | 预测类别变化、预测分布的JS散度 |
| feature_temporal_cosine | 同一图像查询与返回时特征的余弦相似度 |
| feature_batch_cosine | 历史图像当前特征与返回批特征均值的余弦相似度 |
| gradient_temporal_cosine | 查询/返回时使用各自预测伪标签的分类头梯度余弦相似度 |
| gradient_reference_cosine | 返回真值的分类头梯度与当前批 teacher 伪标签梯度的余弦相似度 |
| accuracy_use/accuracy_skip/utility | 未来窗口的两分支平均准确率及差值 |

梯度使用线性分类头交叉熵的解析梯度，包括权重与偏置；不是全网络梯度。
梯度参照和其他特征不使用未来窗口标签。`gradient_reference_cosine` 使用已经返回的标签，
因此这里预测的是**标签返回时、采用监督更新前的价值**，不声称能在标签返回前使用该特征。

分析自动输出：各d的正/负/零收益比例、均值/中位数、指标与U的Spearman相关性；
正/负阈值分别是 U>1e-6、U<−1e-6，其余视为零。
用 StandardScaler + Ridge(alpha=1) 预测U，无超参数搜索。
所有d共享时间边界，最后约30%的返回步留作测试，同批事件不拆分；
训练事件要求 return_step+horizon < 测试起点，防止未来评分窗口重叠泄漏。
报告MAE、均值预测基线MAE、R²、正收益AUC和符号准确率；样本不足或单类时相应指标为N/A。
不要仅凭相关性或训练集拟合宣称预测成功；应检查留出集优于基线并用新seed重复。

不同d的模型会选择不同图像，返回位置也不同，因此不同d均值变化是描述性分析，
不能单凭它断言“对同一标签延迟越长一定越差”或相反。

## 结果位置

以上命令输出到 `storage/utility_prediction/pacs_seed110/`：

- `manifest.json`：运行状态、seed、延迟、Git提交、checkpoint SHA256、设备和环境。
- `summary.md` / `summary.csv`：各延迟的收益分布与完整窗口计数。
- `feature_correlations.csv`：整体及分延迟的指标相关性。
- `prediction_metrics.json`：按时间留出的预测效果及均值基线。
- `heldout_predictions.csv` / `predictor_coefficients.csv`：有足够样本时的留出预测和标准化系数。
- `D0/`、`D20/`、`D50/`、`D100/`：每个延迟独立结果。
- 各D目录的 `return_features.csv`、`utilities.csv`：逐标签指标与真实收益。
- `future_scores.csv`：逐标签、逐未来batch的两分支准确率，可审计20批均值。
- `batches.csv` / `domains.csv` / `feedback.csv`：真实轨迹与oracle查询/返回记录。
- `settings.json` / `factual_result.json` / `result.json`：参数、真实流表现和完成状态。
- `snapshots/`：临时模型状态，逐批完成收益评估后删除；失败时可能保留。

独立重做分析（不重新运行模型）：

```bash
python -m experiments.utility_prediction.analyze \
  --output-dir storage/utility_prediction/pacs_seed110
```

## 连续纠错需求 Need 与六项特征导出

现在采用 `need = 1 - p(y_i | x_i; theta_return)`：概率来自标签返回后、
本步任何监督更新之前的模型，类别 `y_i` 是已经返回的真实标签。
它不是 `1 - max(p)`，也不是预测熵；预测正确但置信度低的样本仍可能有较大的 need。
该特征依赖已经返回的标签，用于返回后是否采用监督的研究，不声称标签返回前可用。

新实验会在 `return_features.csv` 和 `utilities.csv` 保存该概率及连续 need，
完成后自动导出 `utility_features_continuous.csv`。其六项核心字段为：

| 字段 | 定义 |
|---|---|
| utility | 完整未来窗口的 USE - SKIP 准确率差，保持0--1尺度；乘100为百分点 |
| delay | 延迟的batch数 |
| need | 1 - 返回时真实类别概率 |
| pred_drift | prediction_js |
| grad_cos | gradient_reference_cosine，分类头真值梯度与当前批teacher伪标签梯度的余弦相似度 |
| feature_similarity | feature_batch_cosine |

导出文件还保留 `sample_id`、`return_step`、`return_true_probability` 和 `need_binary`，
只统计完整窗口且 utility 非空的事件。不会改变原有预测分析器的特征列表或覆盖原来的二值导出文件。

### 已完成实验如何补算

旧 CSV 没有保存真实类别概率，不能仅由 argmax、熵或最终模型准确恢复。
正常完成实验的返回时快照也已按原流程删除，因此需要重放事实轨迹。
以下命令只重放每个 D 的真实适应过程，**不重新运行 USE/SKIP 分支**，不改变现有 utility：

```bash
python -m experiments.utility_prediction.backfill_need \
  --output-dir storage/utility_prediction/pacs_seed110 \
  --gpu-idx 0
```

默认从原 `manifest.json` 读取 seed、数据与checkpoint路径，从各D的 `settings.json`
读取 batch size、训练步数、评估窗口和 SimATTA 参数。若目录位置变了，可额外传
`--dataset-root /new/datasets` 和 `--checkpoint-dir /new/weights`。
源权重 SHA256 必须与原实验一致。

补算不会保存模型快照，也不会调用反事实收益评估。完成所有D前，必须逐事件对齐
sample ID、查询/返回步、标签、预测类别、原熵、预测漂移、特征和梯度相似度；
数值使用 `rtol=1e-5, atol=1e-7`。若不匹配就报错，不能把新轨迹的 need 与旧 utility 混用。
请尽量使用原数据和原依赖环境。检查通过后另存 `need_backfill.csv`，原结果文件保持不变。

补算默认读取原 `feedback.csv`，按原查询步、anchor索引、样本ID固定重放，
**跳过 K-means 选样与整个历史特征缓存的重复编码**。Utility训练损失从原图重新计算，
不使用聚类权重；返回时的特征指标也重新计算，不依赖被跳过的聚类特征缓存。
这只用于补算原事实轨迹，不是对在线 SimATTA 算法的改动。
确需诊断原选样过程时可加 `--reselect`，使用较慢的重新聚类路径。

每个返回批在更新前立即核对，偏离即停止并报告 D、返回步及不匹配字段/样本键，
不用等整个D运行完才发现不一致。已完整对齐的D保存到 `need_backfill_parts/`，
中断后重试会通过输入与缓存SHA256核验后复用这些结果。
缓存过期或不完整时可指定新的 `--cache-dir /path/to/new-cache`。

这些优化仍需要重放事实轨迹的SGD训练，不能恢复从未保存的信息，也不保证数值轨迹
在不同代码/数据/依赖环境下完全一致。若原概率未记录、返回时快照也已删除，
就没有只靠旧摘要CSV精确恢复连续need的方法；不能改用最终模型或放松对齐来伪造原时刻的need。

随后导出：

```bash
python -m experiments.utility_prediction.export_features \
  --output-dir storage/utility_prediction/pacs_seed110
```

导出器会读取新实验已记录的概率，或自动按 `(delay, sample_id, return_step)`
一对一关联 `need_backfill.csv`。输出为 `utility_features_continuous.csv`。
已有同名文件会报错；可用 `--output-file /path/to/new.csv` 指定其他新文件。
补算侧文件也支持 `--output-file`；自定义侧文件通过导出器的 `--need-features /path/to/sidecar.csv` 传入。

可在服务器验证导出与轨迹对齐逻辑（不运行模型实验）：

```bash
python -m unittest experiments.utility_prediction.test_need
```

本次改进按要求未在本地运行实验或测试；补算是否与原轨迹对齐以服务器实际检查为准。

每个标签都需要两条20批更新分支，耗时显著大于普通SimATTA，不能承诺更快。
按返回批保存模型而非每标签保存；CPU内存复用选样缓存，未来输入只缓存20批，
GPU同时只驻留真实模型和一个分支模型。临时快照可能占用数GB磁盘，请留足空间。
该实现按交付要求只经过语法解析、差异和控制流静态检查，未运行模型、实验或测试，
所有输出均由服务器实际运行生成，没有预填实验结果。
