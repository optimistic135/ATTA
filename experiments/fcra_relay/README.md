# FCRA-Relay：独立四延迟实验

实现说明见 [方法文档](../../docs/fcra_relay_method.md)。新增代码仅位于 experiments/fcra_relay；现有方法、实验入口和结果不修改。本地仅静态阅读和差异检查，未执行 Python、测试、CLI、推理或训练。

## 服务器命令

在项目根目录运行：

```bash
git pull --ff-only origin codex/pacs-delayed-feedback
conda activate df-atta
python -u -m experiments.fcra_relay.run \
  --dataset-root datasets \
  --gpus 0 1 \
  --seeds 110 123 456 \
  --output-dir storage/fcra_relay/pacs_acs_bs32_seeds110_123_456
```

单卡改用 `--gpu-idx 0`，不能同时提供 `--gpus`。可选 `--checkpoint-dir /path/to/photo_checkpoint`，目录需包含 encoder_2.pth 和 fc_2.pth，默认沿用 round1 Photo 检查点；不需要 reference-dir。

输出目录必须不存在；重跑时换新目录。每张 GPU 同时运行一个任务，复用现有逐阶段及总任务进度条。固定只运行 FCRA-Relay 的 D=0/20/50/100 × 三 seed，共 12 次，不自动运行其他方法或消融。

## 数据协议

- Photo 源检查点独立初始化每个任务。
- 在线流仅为 Art → Cartoon → Sketch，batch=32，训练 minibatch 沿用配置（16）。
- 三个目标域各完整访问一次，同图像全流最多查询一次。
- Photo 仅用于源检查点初始化，不进入在线测试流。
- 先预测，再释放到期标签，再训练；D=0 的当前标签也不能影响当批已记录预测。
- 不用真实域边界重置状态，oracle 时钟跨阶段连续，流末不冲刷队列。
- 学习与决策只使用图像、模型预测及已返回真标签，域名称只用于离线统计。
- 查询沿用 FCRA 的熵/聚类及 pending 抑制规则，学习会改变实际查询计划，未强制匹配旧方法的查询数。

## 与旧方法的关键区别

1. 首次反馈前每批也有四次当前图像学习。
2. 返回真标签传播到近期图像的训练目标，不再对历史图像换风格。
3. 不要求传播类别与当前伪标签一致，允许覆盖高置信错误。
4. 查询时保存候选传播，标签返回后直接校准；没有候选模型审计，也不等待审计通过才启动传播。
5. 使用独立损失权重，真实监督不再因伪标签池扩大被稀释。
6. 只更新 layer4 和分类头，源教师、早期骨干冻结，EMA 连续更新。

默认阈值、损失权重及预算全部写在 spec.DEFAULTS，并随每次任务输出到 settings.json。默认值未在本地调参，不代表已验证最优配置。

## 输出

根目录：manifest.json（Git、checkpoint SHA256、协议、seed、GPU）、per_seed.csv、summary.csv、summary.md、stage_summary.csv。

summary.md 分为准确率、标签反馈、学习诊断三张表；所有均值和样本标准差由三个 seed 实际计算。前三域平均也先逐 seed 计算后汇总，不能用三个域标准差简单平均。

| 文件 | 内容 |
| --- | --- |
| result.json | 三目标域、全流/宏平均、反馈比例与学习计数 |
| stages.csv | 每阶段准确率、查询归属的返回/跨域/跨阶段统计 |
| domains.csv | Art、Cartoon、Sketch 三个语义域指标 |
| batches.csv | 更新前计分、原始模型计分、融合纠对/纠错数、首次反馈前后标记 |
| feedback.csv | 物理 ID、查询/到期/返回 batch、域/阶段、跨域/跨阶段标记 |
| queries.csv | 选择和 pending 抑制诊断 |
| query_predictions.csv | 查询时原始/传播/融合软预测、候选及采用标记，不含未返回真标签 |
| relay.csv | 候选/采用传播数量、可靠性、相似度、有效支持度、预测修改次数 |
| calibration.csv | 查询时传播在真标签返回后的正确性与真实类别对数概率变化 |
| updates.csv | 当前/真实监督损失、源约束、传播训练次数、覆盖 EMA 目标次数、梯度范数 |
| budget.csv | 每批主训练额度、实际步数、未用额度和编码器调用计数 |
| compute.json | 主模型/源教师/EMA 的编码器前向与反向调用/图像数、耗时 |

只记录编码器调用与图像数，不将其等同于 FLOPs；初始 checkpoint 加载不计入流运行时间。

## “延迟返回比例”的三个分母

- **returned_pct = 100 × 返回数 / 查询数**：用户请求的标签在流内完成返回的比例，主要关注此项。
- **cross_domain_pct = 100 × 跨域返回数 / 返回数**：已返回标签里有多少在不同语义域返回。
- **positive_delay_returned_pct = 100 × 实际等待 batch 大于零的返回数 / 返回数**：固定 D>0 时理论上为 100%，D=0 时为 0%，用于检查协议。

pending_pct = 100 × 未返回数 / 查询数；同一 seed 有查询时 returned_pct + pending_pct = 100%。没有查询或没有返回时相应比例为 N/A，不写成 0。比例先逐 seed 计算再平均。

本流每个域仅访问一次，因此 cross_stage_pct 与 cross_domain_pct 相同。真实等待长度记录为 realized_delay_mean（单位 batch）。

## 指标解释与边界

三域平均看 first_three_macro_accuracy_pct（等权），全流平均看 first_three_stream_accuracy_pct（按样本数加权）。因测试流只有这三个域，两者分别与 domain_macro_accuracy_pct、stream_accuracy_pct 相同；保留别名便于读取已有分析流程。

relay_net_corrections 是同批固定主模型预测与融合预测的净纠错，不是 USE−SKIP 训练干预收益。raw_stream_accuracy_pct 的模型本身已经受 Relay 训练影响，也不是独立基线。

relay_used_samples 与 relay_overrides 是训练抽样次数，允许重复，不是独立图像数；current_updates 与 supervised_updates 可以重叠，不能相加当作 main_updates。

首次返回 batch 的预测仍记为 before feedback。每批四次当前更新，有真标签后另外 26 次纯监督重放，总上限 30；无真标签时仅四次，未用额度不回填。不声称各延迟实际总步数相同或与旧 FCRA 完全预算匹配。

长延迟的精度恢复、固定源语义空间的可靠性和新机制贡献均需服务器结果验证；本次没有对照方法，不能预先声称有效或新颖性已经成立。
