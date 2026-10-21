# PACS D=0：解释无快速适配器 dual 的收益

本实验回答：在相同信息和更新预算下，针对性重放与教师反馈校准各自是否有效？
无快速适配器的 dual 使用 FCRA 的学习路径，因此无需再次比较快分支。
仅运行 D=0，四组、三个参考 seed，共 12 次，不做参数搜索。

| 组别 | 针对性重放 | 教师校准 | 用途 |
| --- | --- | --- | --- |
| simatta_control | 关闭（均匀抽样） | 关闭（权重恒为 1） | 受控 SimATTA 目标函数对照 |
| replay_only | 开启 | 关闭 | 重放单独效果 |
| calibration_only | 关闭 | 开启 | 校准单独效果 |
| full_fcra | 开启 | 开启 | 无快速适配器 dual/FCRA 的完整学习路径 |

## 一条运行命令

```bash
python -u -m experiments.fcra_ablation.run \
  --dataset-root datasets \
  --reference-dir /path/to/completed-fcra-results \
  --gpus 0 1 \
  --output-dir storage/fcra_ablation/pacs_d0_modules
```

单卡用 `--gpu-idx 0` 替代 `--gpus 0 1`。`--num-workers` 默认 0。
参考使用之前已完成的基础 FCRA 根目录（不是 dual、D0 子目录或单个 seed）。
沿用现有校验：manifest 必须是完整 PACS 四延迟、三 seed 的 FCRA 实验；
只读取 D0 三份 trace，自动使用其中的三个 seed。无需重新跑基础 FCRA。
只生成新目录，不覆盖参考，不修改其他算法入口。

## 如何保证比较公平

所有组共享 Photo 源权重、Art → Cartoon → Sketch 流及三个 seed，online batch=32、
训练 minibatch、冻结 BN、SGD 学习率/momentum 和损失混合比例。
同 seed 的查询 ID、查询时间、真实标签数、每批更新次数全部来自同一个 D0 trace。
先预测并计分，再获取标签和训练；不把评价真值用于查询、重放评分或选参数。
所有组都不创建适配器。完整记忆不裁剪，每次更新都重新有放回抽样。
两个模块只改变抽样分布和伪监督权重，不改变更新额度、真实监督 CE 或混合比例。

`simatta_control` 保留 SimATTA 的固定教师低熵伪标签、普通真实/伪标签 CE 联合目标，
使用统一 runner 的均匀有放回抽样。它是预算匹配的算法目标对照，**不是原始 SimATTA
整套程序的逐位复现**：不使用原默认 batch=100、不自行选样、不自行早停，
也不使用原 InfiniteDataLoader 的随机数消费方式。伪标签为空时，统一 runner 只计算
一次真实监督损失；原实现会从真实记忆抽取两路 batch 后混合。因此不能把该受控结果
直接称为原论文精度，也不能用它声称已经比较原始独立选样策略。

固定查询是为了隔离学习模块贡献，不证明新的主动策略更好。D=0 的 pending 惩罚
不起作用，所以本轮不为它增加消融。full_fcra 应重新跑，不直接复制已有均值。
诊断计算在四组中均保留，不通过跳过计算改变随机抽样；相同 seed 不保证不同
算法轨迹的 minibatch 内容相同。相同优化次数不等于相同 FLOPs。
旧 trace 没有图像内容指纹，必须使用相同数据文件和类别目录。

## 如何解释结果

主要看 `summary.md` 的 Stream，同时检查三个域是否存在收益/损失抵消。
`paired_deltas.csv` 按相同 seed 计算百分点差，而不是相减两个标准差。

- replay_only − simatta_control：重放单独贡献。
- calibration_only − simatta_control：校准单独贡献。
- full_fcra − calibration_only：已有校准时，加入重放的贡献。
- full_fcra − replay_only：已有重放时，加入校准的贡献。
- full_fcra − simatta_control：完整学习路径的净收益。
- interaction = full − replay_only − calibration_only + control：正值说明组合收益
  超过两个独立收益之和；负值说明可能存在重叠或相互干扰。

三个 seed 的配对均值、样本标准差和正收益 seed 数只作描述，不自动宣称显著性。
若 full 无法超过控制组，不能继续把旧实验差异归因为这两个模块；旧差异可能包含
batch、标注量、训练次数、选样与评估协议的影响。本实验只检验固定查询下的机制。

## 输出与静态检查范围

根目录输出 manifest.json、per_seed.csv、summary.csv/md、paired_deltas.csv。
每组/seed 输出 settings.json、result.json、batches.csv、domains.csv、feedback.csv、
updates.csv、replay.csv、calibration.csv，以及继承的预算/分阶段诊断。
summary 在完整 12 次成功后才生成，并检查各组标注数和主模型更新数一致。
所有运行还检查实际总更新数等于参考、反馈事件与参考逐条一致。

辅助机制证据：replay.csv 的覆盖率/有效样本数解释抽样变化；updates.csv 的 gate_mean
解释校准是否真的启用（关闭组恒为 1，有伪标签时）；calibration.csv 记录教师错误证据。
这些诊断是辅助证据，不能替代模块开关带来的配对准确率比较。

本次只编写代码并静态追踪参数、标签时序、模块开关、预算和汇总字段，进行 Git
差异检查；没有在本地执行 Python、测试、模型推理或实验，没有预填性能结果。
