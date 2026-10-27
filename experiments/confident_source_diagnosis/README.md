# 高置信源目标：等待期质量与训练收益诊断

## 两个实验，分别回答不同问题

上一轮同批分析发现，warmup后源置信度前10%目标整体约90.70%正确，Sketch约96.56%；可靠性排序跨域不稳定，分支差异监督很晚才进入训练。这些是动机，不能直接当作新方法的训练收益。

本轮不训练完整新算法，只做两个最小实验：

| 实验 | 问题 | 成本 |
| --- | --- | --- |
| A：等待期离线质量 | Art和首次反馈前是否也有少量可靠源目标？ | 读取已有CSV，无GPU、不训练 |
| B：高精度目标训练收益 | 这些目标加入受限更新后，是否真的改善未来预测？ | 三个seed的SP主轨迹＋少量影子候选 |

固定PACS Photo(source) → Art → Cartoon → Sketch，batch32，D=100，seed=110/123/456。不做参数搜索，不加其他延迟或数据集。旧代码及结果不覆盖。

## 实验A：补查等待期与Art

复用完整selection_diagnosis日志，不要求warmup。等待期从流开始，到第一次标签返回的批次为止（包含该批，因为预测与选样都在更新前）。

主分析只看当批新图像：源置信度>=0.9、原图/水平翻转类别一致的候选中，每批取ceil(10%×N)，至少1个；边界同分用均匀分数权重计算期望正确率。分别报告Art、等待期Cartoon及其早/中/晚时段，给出数量、空批次比例、实际覆盖率及随机选择期望。

最近4批窗口仅作补充。时段按每域等待期内的批次划分，与此前warmup后三段不同，不能直接当成相同时间段比较。修复分支在首次反馈前原则上尚未学习，主表只显示源分支。

```bash
python -u -m experiments.selection_diagnosis.analyze_waiting \
  --input-dir storage/selection_diagnosis/pacs_d100_seeds110_123_456
```

默认输出到输入目录的`waiting_analysis/`。可用`--output-dir`指定新目录，相同输入再运行加`--resume`可校验并跳过；失败报告归档，不重训。

**能得到：** 等待期目标的精度和可用数量。**不能得到：** 等待期适应已经有效；高精度目标可能与当前模型预测一致，未必包含新的类别纠错信息。

## 实验B：四个同快照一步候选

### 正式轨迹与选样

只运行每seed一条原SP轨迹，保留原查询计划、返回时钟、完整真值记忆、历史池、门控和全部正式更新额度。没有可训练风险模型或校正分类头，不依赖它们的二次反馈。

每批由固定源模型选当前批的高置信目标：

1. 原图最大概率>=0.9，且水平翻转类别一致；
2. 按原图源置信度选ceil(0.1N)，非空至少1个；
3. 同分由独立、固定seed的随机顺序均匀破平，不使用真值；
4. 直接用全部选中的K个图像，不再随机抽16张后稀释它们。

训练使用源模型原图软概率，在线模型学习翻转视图。当前损失为KL均值，分母为选中K。它不同于此前“随机batch中乘稀疏掩码再除以B”的损失，因此不声称跨实验差异仅来自选样。

### 四个候选的目标

沿用起点100/130/170/210/240。每个起点后10批内，找到首个有新返回、旧真值、正式额度和可用历史池的时刻。从同一个正式更新前快照生成：

| 候选 | 一步训练目标 |
| --- | --- |
| TRUE | L_T |
| HISTORY | L_T + 0.25 L_H |
| REDUCED | L_T + 0.1875 L_H |
| CURRENT | L_T + 0.1875 L_H + 0.0625 L_C |

L_T为完整旧已返回真值池的CE，L_H为原历史伪标签池的CE，L_C为当前精确小覆盖目标的KL。真实系数始终1。CURRENT保留原名义辅助总系数0.25，只用当前目标替代四分之一历史项。

REDUCED是必要控制：若CURRENT优于HISTORY，可能只是历史辅助减弱，不能立刻归因于新目标。CURRENT与REDUCED的历史系数相同，二者差异检验当前项的额外贡献。

四个候选用同一快照、旧真值抽样、历史抽样、学习率、momentum=0.9和空优化器状态，各训练1步。新返回标签不进入候选真值抽样，其图像也从历史候选抽样排除。CURRENT没有候选时回退HISTORY，并记录inactive；只有active事件能隔离当前项与REDUCED的差异。

随后冻结各候选，评价后续10批，不包含触发批。所有候选丢弃，不回写SP。当前/未来评估真值只用于评分，绝不生成训练目标或门控。

影子辅助系数固定0.25，不根据原SP接受与否改变；另记录SP接受标记用于描述性分层。不能把此局部干预当成部署策略。

### 额外记录：目标精度不等于新纠错信息

选样时保存更新前在线预测、源预测和need=1−p_online(source_class)。评估后统计：

- 源目标正确、在线模型错误：直接纠错机会；
- 源目标错误、在线模型正确：改错风险；
- 两者都正确/都错误可由记录推导。

这些是弱视图上的机会指标，不等于实际更新改对/改错，也不能排除对强视图或未选样本的间接泛化。

### 服务器命令

```bash
python -u -m experiments.confident_source_diagnosis.run \
  --dataset-root datasets \
  --reference-dir storage/simatta_rootcause/pacs_d100_bs32_seeds110_123_456 \
  --gpus 0 1 \
  --seeds 110 123 456 \
  --output-dir storage/confident_source_diagnosis/pacs_d100_seeds110_123_456
```

参考是原始SimATTA D100根因诊断，不是selection_diagnosis或其他方法结果。每GPU同时一个任务，有逐域和总进度条；单卡替换为`--gpu-idx 0`。同目录续跑加`--resume`，完整seed跳过，未完成seed从头执行，旧残留先归档。支持`--checkpoint-dir`、`--num-workers`。

## 输出

根目录`summary.md`先显示SP事实轨迹，再显示候选未来净收益和选中目标质量。SP主准确率用于核查主轨迹，不作为新算法收益。

- `future_per_seed/summary.csv`：CURRENT−HISTORY、CURRENT−REDUCED、CURRENT−TRUE、REDUCED−HISTORY的fixed/broken及净收益，按ALL/Cartoon/Sketch和all/active/sp_accepted分组。
- `targets_per_seed/summary.csv`：全流、首次反馈前后、探针时刻的精度、数量、机会和风险。
- 各seed的`event_effects.csv`：每个快照的目标质量与未来收益放在一行。
- `elite_targets.csv`：逐个实际选中目标的源/在线预测与事后正确性。
- `elite_selection.csv`、`elite_draws.csv`、`elite_probes.csv`、`elite_future.csv`、`elite_skipped.csv`：数量、身份、快照哈希与评分。
- 原SP的batches/domains/feedback/gates/updates/budget/compute等保留。

未来只有完整10批窗口进入汇总，按实际评分域分组；跨域窗口可同时出现在两域事件计数中，不能相加当作独立探针。seed内按评分样本数加权，跨三个seed报均值±样本标准差及有效n。

离线实验A报告同分期望；实验B实现可执行的私有随机破平，因此实际精度可能不同。CURRENT多做K张图像的前向/反向，相同一步不代表相同FLOPs。main_updates只有原SP正式步骤，candidate_updates为原SP门控开销，diagnostic_candidate_updates为四候选开销。

## 预期得到的结论与失败判据

| 观察结果 | 可支持的判断／下一步 |
| --- | --- |
| 等待期源目标精度高且每批有样本 | 有等待期学习信号，值得另做等待期更新对照；尚未证明Art准确率提高 |
| 等待期精度低或数量不足 | 不把warmup后的结论外推到反馈前 |
| CURRENT同时优于HISTORY与REDUCED，多个seed方向一致 | 支持小比例当前项具有额外价值，再实现完整在线方法 |
| CURRENT优于HISTORY，但接近REDUCED | 改善主要可能来自减弱历史辅助，不能归功于当前目标 |
| CURRENT优于REDUCED，但不优于HISTORY | 当前项有边际作用，但本配比没有补偿减少的历史项，不宣称完整替换有效 |
| CURRENT−REDUCED约为零或负 | 本设置尚未展示当前项价值；结合机会/风险/need定位原因，不继续堆模块 |
| 没有选中样本或完整窗口 | 数据不足，不作有效性判断 |

预期是区分这些解释，不是预先保证哪种结果。只有三个seed、五个探针，单个正均值不自动成立为显著性、创新性或整流收益。

**本地未执行Python、测试、分析或训练，仅静态审查。**
