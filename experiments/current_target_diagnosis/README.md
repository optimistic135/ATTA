# 当前目标的同快照诊断：PACS D=100

## 要回答的问题

反馈校正分类头改善了历史查询预测，但完整方法弱于 SP。这里只区分三种可能：当前目标没有改善、目标改善却产生不利更新、历史返回标签上的门控判断不代表未来收益。不提出或运行新的正式适应算法。

仅运行 Photo(source) → Art → Cartoon → Sketch、batch32、D=100、seed=110/123/456。共三个主任务，每个任务的正式预测/训练继续使用原 SP-ATTA。其他延迟、额外数据集和参数搜索均不加入。

## 最小设计

### 一条事实轨迹

复用原始 `simatta_rootcause` 的查询 ID、时钟、源检查点哈希、逐批正式更新额度。SP 的完整真实记忆、固定源伪标签池、抽样和原门控保持原路径。

另维护一个诊断用反馈校正分类头：复制固定源分类头，保持源编码器不变。每个有新反馈且正式额度>0的批次，在正式训练结束后执行一次分类头 SGD(lr=0.001, momentum=0)：

L_head = CE(hψ(E0(x)), y) + 0.01 Σ_parameters mean((ψ−ψ0)²).

真实图像从完整已返回记忆抽样，使用旧反馈研究中该批最后一次正式抽样的相同索引。这个头不影响 SP 的预测、门控、查询或正式优化，更新只算诊断开销。没有任何新标签的提前访问。

### 五个小探针

预设起点 100、130、170、210、240（0-based）。在各起点之后10批内，找到首次同时存在新返回标签、旧已返回标签、正正式训练额度的批次。没有合格触发则记录跳过；不使用域边界决定触发。

在新标签并入正式记忆前，保存同一在线模型快照。构造最近4批图像窗口，每批原图为弱视图、水平翻转为强视图。固定源分支和诊断校正分支在相同弱视图上给出软概率。

每个分支自己的筛选条件为：最大概率>=0.9，且弱/强视图类别一致。排除本次新返回图像，保证候选验证集不进入候选损失。分支可能选择不同样本，这是目标策略的一部分，不能把它解释为纯概率变化。

三个候选使用相同模型快照、相同旧真值 minibatch、相同当前图像 minibatch、相同源学习率、momentum=0.9、空优化器状态，各更新一步：

L_TRUE = CE_truth

L_SOURCE = CE_truth + 0.25 mean_batch[m_source KL(stopgrad(q_source) || p(x_flip))]

L_REPAIR = CE_truth + 0.25 mean_batch[m_repair KL(stopgrad(q_repair) || p(x_flip))].

真实监督系数均为1。辅助损失按整个采样 batch 归一化，不按通过筛选的样本数归一化。某个分支没有可信样本时，该分支辅助损失为零，不更换抽样或偷偷补充候选。

REPAIR 在探针中始终使用校正分支目标，不套用旧研究的“历史证据为正才使用校正头”的分支选择规则；这样直接检验这些目标有没有潜力。诊断不等于重新部署 Current-Feedback。

### 两类独立评分

1. **当前目标质量（仅离线评估）**：在 batch 结束的诊断钩子中，用评估真值评分此前保存的两种目标；报告全部允许样本、源筛选样本、校正筛选样本、双方共同筛选、类别分歧、实际抽样样本。保存每个图像的预测、置信度、正确性、CE及筛选标记。评估真值存入独立只供评分的短窗口，不进入任何目标、梯度、门控或分类头训练。
2. **未来更新效果（仅离线评估）**：三个候选训练后冻结，评分触发批之后10批，不包含触发批。相对 TRUE，分别统计改对数 fixed、改错数 broken，净收益为100×(fixed−broken)/samples；另比较 REPAIR−SOURCE。只有完整10批窗口进入汇总。

未来批按照其实际 Cartoon/Sketch 域归类；跨边界探针分开归入两域，不能简单按触发域归类。目标质量按照图像实际所属域归类；窗口重复包含图像时按“窗口—图像观测”计数，不当作独立样本。

### 接受事件不混淆

在同一批新返回真值上分别评分三个候选：

gain_source = CE_fresh(TRUE) − CE_fresh(SOURCE)

gain_repair = CE_fresh(TRUE) − CE_fresh(REPAIR).

各自 gain>0.001 时记录为该候选自己的假设接受事件；不会把决定写回主模型。另记录原 SP 历史池门控的决定。

未来汇总分成 all、source_accepted、repair_accepted、both_accepted、sp_accepted。比较 SOURCE 与 REPAIR 的目标优劣，优先看 all 或双方共同接受事件；两者各自接受事件可能不同，不能拿两个条件均值直接相减推断目标优势。

## 服务器命令

```bash
git pull --ff-only origin codex/pacs-delayed-feedback
conda activate df-atta
python -u -m experiments.current_target_diagnosis.run \
  --dataset-root datasets \
  --reference-dir storage/simatta_rootcause/pacs_d100_bs32_seeds110_123_456 \
  --gpus 0 1 \
  --seeds 110 123 456 \
  --output-dir storage/current_target_diagnosis/pacs_d100_seeds110_123_456
```

参考目录是**已完成的原始 SimATTA D100 根因诊断**，含 D100/seed_110 等，不是 feedback_teacher 或 sp_atta 的目录。

每 GPU 同时一个任务，保留逐域和总任务进度。单卡用 `--gpu-idx 0` 替代 `--gpus 0 1`。支持自定义 `--checkpoint-dir`、`--num-workers`。

同目录续跑：旧任务停止后，在同一命令末尾加 `--resume`。完整种子结果校验后跳过；未完成种子从头重跑，原失败目录及日志移动到 `_resume_archive/UTC时间/`。没有逐 batch 检查点。不能用此入口续跑先前的 feedback_teacher 实验；那是不同的研究。代码/参数若改变算法含义，请用新目录。

## 输出

根目录：

- `summary.md`：SP 主轨迹、逐域/接受事件的候选收益、当前目标准确率。
- `summary.csv`：SP 主轨迹准确率、跨域比例、查询返回数及开销。
- `future_per_seed.csv`、`future_summary.csv`：Cartoon/Sketch、事件子集、三候选准确率、fixed/broken、净收益。
- `quality_per_seed.csv`、`quality_summary.csv`：当前目标的分域/筛选子集正确性与 CE 差。
- `per_seed.csv`、`manifest.json`、任务日志。

每个 seed：

- `event_effects.csv`：建议先看；每个同快照事件的门控证据、目标质量与未来净纠错直接放在一行。
- `target_probes.csv`：触发域、候选 CE、接受决定、可信数量、快照及抽样哈希。
- `target_quality.csv`：当前图像两种目标的逐样本离线评分。
- `target_future.csv`：未来逐批三候选准确率及改对/改错数。
- `target_draws.csv`：共同真值及图像抽样身份。
- `repair_updates.csv`、`target_skipped.csv`、`diagnostics.json`。
- 原 SP 的 batches/domains/feedback/gates/updates/budget/compute 等记录。

`candidate_updates` 是原 SP 影子开销；`diagnostic_candidate_updates` 是新增三候选开销；`diagnostic_head_updates` 是诊断分类头开销。`main_updates` 始终只有原正式预算，不扣除或混入诊断步骤。compute/budget 另外记录诊断骨干前向/反向计数。耗时和峰值内存包含诊断，不能与原 SP 直接作为部署效率比较。

## 结果到改进策略

| 观察 | 下一步值得单独验证的策略 |
| --- | --- |
| 共同筛选样本上校正目标反而更差 | 停止扩大分类头校正，不增加修复使用频率 |
| 目标改善，但 REPAIR 相对 TRUE 净收益为负 | 调低辅助强度或限制参数更新范围；先保留真实监督 |
| REPAIR 有正收益，但历史反馈门控接受与未来收益不一致 | 优先研究验证准则，而不是继续改目标 |
| SOURCE、REPAIR 都不如 TRUE | 暂停当前自训练路线，回到真实反馈利用 |
| 目标改善，REPAIR 同时优于 SOURCE 和 TRUE | 再接回正式 SP 并验证整流收益 |

以上是下一步候选方向，不是诊断能直接证明的最优策略。一步效应不保证多步训练收益，五个探针不代表整个流。统计先在 seed 内按样本加权，再跨三个 seed 报均值±样本标准差，CSV保留有效n；不能把探针数量当成独立seed数量。

**本地未运行 Python、测试、训练或实验，只做静态审查；不预填准确率、不宣称方案已有效。**
