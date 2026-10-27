# 理想目标诊断：PACS D=100

## 目的与设置

上一轮中，当前高置信度目标仍大量错误，而 SOURCE/REPAIR 的一步辅助收益没有稳定超过 TRUE。本实验只区分：**主要是类别目标错误，还是在这套图像/掩码/强度下，更新本身也不能带来收益？**

固定 Photo(source) → Art → Cartoon → Sketch，batch32，D=100，seed=110/123/456。共三个 SP-ATTA 主任务，不增加正式算法、延迟或参数搜索。原 current_target_diagnosis 和已有结果不修改。

SP 主轨迹使用相同原始查询计划、返回时钟、完整真值记忆、正式更新预算、伪标签池及门控。反馈校正分类头只供诊断，按上一版同样的已返回真值抽样训练，不影响 SP。它和全部影子候选都是额外诊断开销，不能解释为相同 FLOPs 下的新方法。

## 五个同快照候选

保留起点100/130/170/210/240、每个起点后10批内首次合格反馈触发、最近4批图像窗口、一步 SGD 和未来10批冻结评分。

| 候选 | 真实监督 | 当前辅助目标 | 辅助掩码 |
| --- | --- | --- | --- |
| TRUE | 完整旧已返回记忆抽样，系数1 | 无 | 无 |
| SOURCE | 同一抽样 | 固定源分支概率 qS | mS |
| REPAIR | 同一抽样 | 反馈校正分类头概率 qR | mR |
| ORACLE_SOURCE | 同一抽样 | qS 的真实类别校正版 | **原 mS 不变** |
| ORACLE_REPAIR | 同一抽样 | qR 的真实类别校正版 | **原 mR 不变** |

五个候选来自同一份更新前模型快照，使用同一旧真值 minibatch、同一当前图像 minibatch、同一学习率、momentum=0.9、空优化器状态及重置后的相同 RNG。真实监督只使用触发前已经返回的标签，排除当批新返回验证标签。

当前辅助损失为 0.25 × mean_batch[m KL(q || p(x_flip))]，按整个抽样 batch 大小归一化。原图/水平翻转一致且置信度>=0.9才被原掩码选中；不因真实类别改掩码、重选图像或补充样本。

### 理想目标如何构造

对原概率向量 q，原预测 c=argmax(q)，离线评估真实类别为 y：

- 若 c=y，q 完全不变。
- 若 c≠y，只交换 q[c] 与 q[y]。

这样每张图像的概率值集合、最大置信度和熵全部保持不变；对被选中样本，最大概率对应真实类别。不同于直接使用 one-hot 标签，这避免额外改变目标尖锐程度。代码逐样本检查概率集合不变，以及选中样本的理想目标类别正确，并输出概率向量供审查。

这纠正的是主预测类别，不代表知道完美的类别条件概率；也不保证梯度范数不变。

### 真值何时使用与隔离边界

触发时先构造普通三个候选，保存更新前快照、当前目标、掩码和所有抽样索引。随后 SP 继续其正常正式训练。

在该批结束的评估钩子中，当前窗口评估真值才交给两个 ORACLE 副本。它们从**之前保存的快照**开始一步更新，不从已经训练后的 SP 模型开始。分类头本批修复仍在之后执行。

评估真值只用于离线评分和两个明确标注的、不可部署的 ORACLE 副本，不能进入 SP、查询策略、普通 SOURCE/REPAIR、门控或反馈校正分类头。代码检查 ORACLE 更新前后 SP 和分类头指纹未变；RNG 隔离后恢复。

这是一项故意允许影子副本访问当前真值的非因果诊断，**不是合法延迟协议下可直接部署的方法**。下一批及未来10批真值仍只用于评分。

## 主要比较与解释

| 比较 | 用途 |
| --- | --- |
| ORACLE_SOURCE − SOURCE | 在原源分支掩码上，纠正类别内容能带来多少改善 |
| ORACLE_REPAIR − REPAIR | 在原校正分支掩码上，纠正类别内容能带来多少改善 |
| ORACLE_SOURCE − TRUE | 原源掩码下的正确类别辅助，能否超过纯真实监督 |
| ORACLE_REPAIR − TRUE | 原校正掩码下的正确类别辅助，能否超过纯真实监督 |

必须同时看前后两类比较：理想目标只比错误目标好、但仍不超过 TRUE，不足以支持增加当前辅助训练。

- 理想目标明显超过对应实际目标，且超过 TRUE：支持优先研究目标可靠性，而非增加更新次数。
- 理想目标改善，但仍未超过 TRUE：目标错误只解释一部分，下一步再单独检查强度、参数范围或图像抽样。
- 理想目标也没有正收益：只能说明这套一步、权重、掩码和图像组合未显示潜力；不能证明当前数据学习在任何设置下都无效。
- 分域结果不同：先定位在哪个域、哪些事件产生收益，不用全流均值掩盖差异。
- 选中样本数或实际被纠正数为零：该探针没有识别该问题的能力，不把零收益解释成否定证据。

ORACLE 不是性能上界，也不保证每次优于实际目标。

## 服务器运行

```bash
git pull --ff-only origin codex/pacs-delayed-feedback
conda activate df-atta
python -u -m experiments.oracle_target_diagnosis.run \
  --dataset-root datasets \
  --reference-dir storage/simatta_rootcause/pacs_d100_bs32_seeds110_123_456 \
  --gpus 0 1 \
  --seeds 110 123 456 \
  --output-dir storage/oracle_target_diagnosis/pacs_d100_seeds110_123_456
```

参考目录仍是已完成的**原始 SimATTA D100 根因诊断**，不是上一轮 current_target_diagnosis 结果。每 GPU 同时一个任务，逐域和总进度条保留。单卡替换为 `--gpu-idx 0`。支持 `--checkpoint-dir`、`--num-workers`。

同目录续跑在本命令最后加 `--resume`：完整 seed 校验后跳过，失败/未完成 seed 从头执行，旧残留归档到 `_resume_archive/`。先停止旧进程。不能在上一轮三候选实验目录中续跑五候选实验，也不能把旧 SOURCE/REPAIR 结果与新 ORACLE 结果直接拼接代替同快照对照。

## 输出与统计

先看根目录 `summary.md` 的 Oracle contrasts 表：按 ALL/Cartoon/Sketch，以及 all、source_accepted、repair_accepted、both_accepted、sp_accepted 分组。

接受事件继承普通候选在历史新返回标签上的判断；没有 ORACLE 门控，也绝不根据 ORACLE 收益决定正式更新。两个不同接受集合的条件均值不能直接相减；优先看 all 或同一个事件子集中的配对差。

每个 seed 的 `event_effects.csv` 包含每个快照的门控、目标准确率、所有候选净收益，以及 `oracle_source_changed_selected` / `oracle_repair_changed_selected`。先确认被纠正数量，再解释收益。

`oracle_targets.csv` 新增逐样本审计：原/理想预测、真实类别、原/理想置信度、完整概率向量、原掩码与是否改动。`target_draws.csv`、快照/抽样哈希、`target_future.csv` 保留。

`future_per_seed.csv` / `future_summary.csv` 包含五候选准确率及每一配对的 fixed、broken、净收益：100×(fixed−broken)/样本数。`quality_*` 仍报告普通源目标与校正目标质量；ORACLE 的已选中类别正确性由断言及审计表检查。

只有完整未来10批窗口进入未来汇总；按评分数据的实际域分开。seed内按样本加权，跨三个seed报均值±样本标准差及有效n。不把探针、图像或重复窗口视为独立seed。

`main_updates` 只有原 SP 正式更新；`candidate_updates` 是原 SP 门控影子开销；`diagnostic_candidate_updates` 包含本实验全部五候选；`oracle_candidate_updates` 是其中两个 ORACLE 的子集，不能再次相加；`diagnostic_head_updates` 是独立分类头开销。耗时与峰值内存包含诊断，不用于宣称部署效率优势。

**本地仅静态审查，未运行 Python、测试、诊断或训练；不预填准确率，不宣称任何改进已经有效。**
