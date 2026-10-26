# 原SimATTA D=100 简单根因诊断

[实验设计](DESIGN.md)说明比较逻辑。[静态检查](STATIC_REVIEW.md)说明未运行验证的边界。只新增此包，不修改现有算法或旧实验。

## 服务器运行

在项目根目录执行：

```bash
git pull --ff-only origin codex/pacs-delayed-feedback
conda activate df-atta
python -u -m experiments.simatta_rootcause.run \
  --dataset-root datasets \
  --gpus 0 1 \
  --seeds 110 123 456 \
  --output-dir storage/simatta_rootcause/pacs_d100_bs32_seeds110_123_456
```

单卡改用 `--gpu-idx 0`，不能与 `--gpus` 同时使用。可选 `--checkpoint-dir /path/to/photo_checkpoint`，需包含encoder_2.pth和fc_2.pth；默认沿用round1 Photo checkpoint。不需要reference-dir。输出目录必须不存在，避免覆盖；重跑请换目录。

仅运行原SimATTA D=100的三个seed主轨迹；每条轨迹最多五次TRUE/PSEUDO/MIX/SKIP诊断。无需启动四个不同算法，也不跑其他delay或参数搜索。双卡一张卡一个主任务，显示原逐域和总任务进度条。

默认batch32，Photo → Art → Cartoon → Sketch；每次主轨迹沿用原SimATTA的PACS早停次数。诊断分支各十步短更新后冻结，观察后十批；额外诊断计算单独理解，不占用或替换主模型的训练步数。

本地未运行Python、CLI、测试、推理或实验，仅做静态审查。服务器运行之前不能声称原轨迹数值已复现或根因已确定。

## 输出与先看什么

根目录：

- `summary.md`：主SimATTA各域/全流准确率和返回统计；四分支后续准确率及相对SKIP收益；TRUE−MIX、梯度指标。
- `per_seed.csv`：每个seed的完整主实验和诊断指标。
- `paired_per_seed.csv`：每个seed的配对收益，便于检查收益方向是否一致。
- `summary.csv`：均值、样本标准差及每个指标有效n。
- `manifest.json`：Git、源checkpoint SHA256、seed、GPU和固定协议。

每次运行的D100/seed目录：

| 文件 | 内容 |
| --- | --- |
| result.json / settings.json | 结果、原SimATTA配置、诊断默认值和环境信息 |
| batches.csv / domains.csv / feedback.csv | 未改变的主SimATTA流结果与标签查询/返回/跨域信息 |
| phases.csv | 更新前预测在首次反馈前后所属阶段，及每批原模型实际更新数 |
| probes.csv | 每个时刻×分支的历史拟合、未来收益、改对/改错、alpha、同域比例、梯度和初始哈希 |
| probe_batches.csv | 分支后续每批结果、评价域，可检查是否跨域边界 |
| probe_training.csv | 诊断分支每步真实/伪监督损失和实际目标；未用的损失为空 |
| probe_draws.csv | 配对的真标签anchor索引和伪标签池索引 |
| skipped_probes.csv | 未满足触发条件的预设时刻 |

第一步看 `TRUE_minus_SKIP_pp` 和 `TRUE_minus_MIX_pp` 的逐seed结果，再结合历史拟合变化和梯度余弦判断解释。不能仅凭均值正负直接声称统计显著。

四分支准确率对应各自固定快照后的局部未来窗口，不能与整流accuracy直接比较。只有窗口完整的事件进入汇总；若没有完整窗口，准确率和收益为N/A，不写成0。

原主模型的Main updates只计官方one_step_train调用，不包括候选短更新和梯度诊断。不同分支的目标计算量不同：SKIP零步，其他各十步，MIX每步含两种数据。这里控制的是目标干预的更新步数，不声称FLOPs相同。

snapshot包含完整模型参数和BN缓冲区。候选优化器与主模型隔离；Python/NumPy/CPU及当前GPU随机状态恢复；不会操作其他GPU的RNG。候选更新后检查主模型state_dict哈希未改变。分支仅临时存于CPU，未来评分时逐个搬到GPU以控制显存，不导出大规模snapshot文件。
