# 筛选可用性诊断：PACS D=100

## 目的与最小设置

上一轮SP-Select/SP-Correct的目标训练draws为0。本轮不增加正式算法，只回答四个问题：

1. warmup后分数是否有变化、能否达到0.8？
2. 哪一道条件开始把样本全部筛掉？
3. 分数较高的样本是否确实更正确，优于原置信度排序吗？
4. 总体Brier≈0.25是否主要来自学习开始前查询时的0.5预测？

固定PACS：Photo(source) → Art → Cartoon → Sketch，batch32，D=100，seed=110/123/456。每seed只跑一条完整SP主轨迹，共3个任务，不重跑原三组×三seed。

## 不改变正式学习

SP继续原查询ID/时间、源检查点、完整标签记忆、历史伪监督门控和逐批正式额度。可靠性模型与源分类头副本采用原选择性方法的相同初始化、输入、学习率、更新频率和训练抽样规则，但全部作为影子训练，不向SP提供当前目标，也不占用SP原有正式额度。

- 复制源分类头，只用合法返回真值训练；固定源编码器不更新。
- 风险模型仍为22维输入、两个sigmoid输出的线性模型。
- 特征仍为两组概率、置信度、类别间隔、熵、预测一致性与JS散度。
- 返回标签只监督查询时保存的源/校正预测正确性，不使用训练后的预测替代原记录。
- 有新返回且quota>0时，批末执行一次风险训练；quota>=2时再执行一次分类头训练。规则对应原方法的最后两次额度，诊断中额外记账。

这是可用性诊断，不是相同步数/FLOPs下的新部署方法比较。SP正式轨迹应保持原结果，实际数值一致性仍需服务器验证。

## 1. 分数分布

每批在新反馈参与学习前，对最近4批所有图像记录分数，**包括被拒绝样本**。两种分支分别报告：

- `warm_all`：满足warmup、未与本次返回验证图像重叠的窗口图像；
- `warm_base`：上述图像进一步满足该分支置信度>=0.9且原图/翻转类别一致，但尚未使用风险阈值。

报告样本数、均值、分布标准差、min、P05、P25、中位数、P75、P95、P99、max及阈值通过率。另报告repair−source分数差的分布，检查0.1 margin是否可达。

分位数在每seed内计算，再跨seed汇总，不把所有seed混在一起计算一个分位数。窗口图像会重复，计数是观测数，不是独立样本数。

## 2. 逐级筛选人数

源路径依次记录：

```text
window → allowed → warmup → confidence≥0.9
       → flip consistency → source risk≥0.8 → SP gate
```

校正路径进一步检查：

```text
window → allowed → warmup → confidence≥0.9
       → flip consistency → repair risk≥0.8
       → repair−source≥0.1 → class disagreement → SP gate
```

每阶段报告进入、通过、丢弃及条件通过率。分为全流和仅warmup后两种统计，避免等待期的全拒绝掩盖后续阈值瓶颈。

源筛选数对应SP-Select；SP-Correct采用校正优先、否则源回退的同一规则，最终假设动作数量单独记录。SP gate为原SP在实际轨迹上的决定；这里不执行任何当前目标训练，也不模拟每次训练draw，不能把假设选择人数称作实际训练数量。

warmup定义仍为：已经返回至少32条查询记录，且完成至少8次可靠性更新。最先确认是否进入warmup，再解释阈值。

## 3. 离线固定覆盖率排序

在每个分支自己的`warm_base`集合中，固定比较前10%和后10%：

- 风险分数前10%目标准确率；
- 风险分数后10%目标准确率；
- 原置信度前10%/后10%目标准确率；
- 风险高分−风险低分，以及风险高分−置信度高分。

两种排序使用同一候选集合、同一个目标分支、相同数量，取`ceil(0.1N)`个观测，至少1个。边界同分采用分数权重：若边界还需选k个、共有m个同分观测，则每个权重k/m。完全同分时，高低分准确率都等于候选总体准确率，不利用样本顺序虚构排序能力。

这些只是事后诊断，不会变成在线top10%策略，也不使用评估真值选阈值。ALL与Cartoon/Sketch分开报告；跨域ALL的排序优势可能来自难度构成变化，必须检查分域结果。两种分支的候选集合不同，不能将其准确率直接相减解释为纯校正效果。

另输出固定0.1宽度分数箱的样本数、平均分数、准确率与Brier，用于观察分数与正确性是否对应。

## 4. 按查询时学习状态拆分Brier

每次查询记录分数、原置信度、风险更新次数和warmup状态；标签到期后评分。分组依据是**查询时**，不是返回时：

| 分组 | 定义 |
| --- | --- |
| before_learning | 查询时尚未完成任何风险更新 |
| after_learning | 查询时已经完成至少一次风险更新 |
| after_warmup | 查询时已满足warmup，是after_learning的子集 |
| all | 所有流内已返回查询 |

原始风险模型零初始化，学习前输出0.5，对任意二值正确性标签Brier恒为0.25。该事实用于解释总平均，不能据此推断学习后当前分数仍恒为0.5。

比较三个预测器：学习到的可靠性分数、原最大置信度、合法返回历史的在线正确率基线。后者只使用查询前已返回的记录，Beta(1,1)平滑为`(correct_count+1)/(returned_count+2)`，不使用测试集总体比例。

只评价最终流内返回的查询，存在主动选样和流尾截断偏差。D100下后期很多查询尚未返回，因此学习后Brier可能缺样本；结合当前窗口的离线分箱与排序，不只看查询Brier。

## 服务器命令

```bash
git pull --ff-only origin codex/pacs-delayed-feedback
conda activate df-atta
python -u -m experiments.selection_diagnosis.run \
  --dataset-root datasets \
  --reference-dir storage/simatta_rootcause/pacs_d100_bs32_seeds110_123_456 \
  --gpus 0 1 \
  --seeds 110 123 456 \
  --output-dir storage/selection_diagnosis/pacs_d100_seeds110_123_456
```

参考目录仍是已完成的**原始SimATTA D100根因诊断**，含D100/seed_110等，不是selective_feedback结果。每GPU同时一个任务，逐域和总进度条保留；单卡用`--gpu-idx 0`。支持`--checkpoint-dir`、`--num-workers`。

同目录续跑在本命令最后加`--resume`：先停止旧进程，完整seed校验后跳过；未完成seed重新执行，残留与旧报告归档到`_resume_archive/`。不能用上一轮方法的目录续跑此诊断。

## 输出与阅读顺序

先看根目录`summary.md`：warmup → 分数上限 → 筛选流失 → 固定覆盖率排序 → 查询阶段Brier。

| 根目录文件 | 内容 |
| --- | --- |
| distribution_* | 分数及margin分布 |
| funnel_* | 逐级通过数与丢弃数 |
| ranking_* | 高低分与等覆盖置信度对照 |
| query_brier_* | 学习前后及warmup后Brier |
| score_bins_* | 分数箱与实际正确率 |
| warmup_* | warmup起点、批次数和假设策略人数 |

每类均有`per_seed.csv`和`summary.csv`，并给出有效n。均值±样本标准差按三个seed计算，不将窗口观测当作独立seed。

每seed保存：

- `score_observations.csv`：所有窗口图像的分数、置信度、条件与事后正确性；包含拒绝样本。
- `filter_funnel.csv`：逐批逐级人数。
- `query_scores.csv`：查询时状态与返回后正确性。
- `policy_counts.csv`：warmup与源筛选/校正优先的假设人数。
- `shadow_updates.csv`：影子头/风险损失和真实训练抽样身份。
- `diagnostics.json`：本seed报告数据；`result.json`最后写入作为完成标记。
- 原SP的batches/domains/feedback/gates/updates/budget/compute等。

SP正式更新计入main_updates，影子模块计入shadow_head_updates / shadow_reliability_updates。原SP门控候选仍单独计数；没有新增当前目标探针训练。耗时包含日志、评分及影子开销，不用于宣称部署效率优势。

## 预先约定的解释

- 未进入warmup：先检查反馈数量/训练时序，不调整风险阈值。
- warmup后max仍小于0.8，但风险排序优于置信度：可能有排序信号，值得另开合法反馈校准实验；不能直接用测试标签确定新阈值。
- 高低分正确率接近，且不优于置信度排序：当前可靠性模型缺少选择能力，不继续靠降低阈值增加覆盖。
- 风险过滤有样本但margin无通过：定位校正判别或margin限制，不整体放宽所有规则。
- Brier优于原置信度却不优于在线基率：可能只是减轻过度自信，不能说明获得个体可靠性判断。
- 学习前样本占多数：总Brier≈0.25不能作为学习后无效的充分证据。

**本地仅静态审查，未执行Python、测试、诊断或训练，不预填结果。**
