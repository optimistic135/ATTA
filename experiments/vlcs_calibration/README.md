# VLCS：calibration_only 的跨数据集验证

只运行 D=0，不做 Oracle、随机保护、快速适配器、针对性重放或超参数搜索。
三组、三个 seed（默认 110/123/456），共 9 次，检验 PACS 上的局部教师反馈门控
在另一数据集上是否带来正收益。代码不保证提升，不预填实验数值。

## 服务器命令

```bash
python -u -m experiments.vlcs_calibration.run \
  --dataset-root datasets \
  --gpus 0 1 \
  --output-dir storage/vlcs_calibration/d0_seeds110_123_456
```

单卡改为 `--gpu-idx 0`，可用 `--seeds` 指定三个不同 seed，`--num-workers` 默认 0。
不需要 `--reference-dir`，程序自动先运行每个 seed 的原始 SimATTA，再复用其记录。
每个 GPU 一次运行一个 seed 的流水线；每组在全新的 Python 子进程中运行，重新加载
相同源权重、初始化随机数和数据流，避免组间模型/随机数污染。
输出目录必须是新目录，不覆盖 PACS、门控诊断或旧 VLCS 结果；本入口暂不支持续跑。
失败时停止派发新 seed，已运行的结果留在各组目录和日志，不生成不完整的均值表。

## 数据和源权重

需要 `datasets/VLCS/{C,L,S,V}/<类别>/<图像>`，或
`datasets/VLCS/{Caltech101,LabelMe,SUN09,VOC2007}/<类别>/<图像>`。
源域 Caltech101，目标流 LabelMe → SUN09 → VOC2007，batch=32，保留末批。
默认检查点目录：

```text
storage/checkpoints/round1/VLCS/[0, 1, 2, 3]/ResNet18_3l_meanpool_0dp_Truefzbn/0.0001lr_0.0wd/ERM_no_param/
```

须含 encoder_0.pth、fc_0.pth，可用 `--checkpoint-dir` 指定同架构 Caltech101 权重。
不下载数据、不训练源模型；原模型构造仍需 torchvision ImageNet 权重缓存。
所有 worker 验证检查点 SHA256 与父进程 manifest 一致。

读取官方 VLCS SimATTA.yaml：lr=1e-3、eh=1e-2、el=1e-3、cold_start=100、stop_tol=5，
保持官方训练 minibatch、模型、冻结 BN 等配置；只覆盖 online batch=32 和数据加载
worker 数。VLCS 的 steps=100 不会变成额外固定更新上限：原代码对 VLCS 沿用监督
损失窗口早停。`budgets=100` 不是原算法硬标注上限，结果会报告实际标注数。
门控沿用 PACS 参数，不根据 VLCS 准确率调参。

## 三组比较

| 组别 | 查询与更新额度 | 学习机制 |
| --- | --- | --- |
| simatta_original | 自行高熵聚类查询、原始损失早停 | 原 SimATTA 的 InfiniteDataLoader、联合 CE、空伪标签两路真实监督回退 |
| simatta_control | 复用该 seed 的原始组查询/逐批更新记录 | 统一训练循环的均匀真实记忆重放、未校准伪监督 |
| calibration_only | 同 control | 只增加固定教师局部反馈门控 |

原始方法保留现有流适配层的 query ID 去重与极小候选簇数保护，使用更新前首次在线
预测计分，不执行原 __call__ 中额外的随机流第二轮与适应后全域测试。
控制组和方法组不重跑选样与早停；共享完整真实记忆、伪标签记忆大小、混合比例、
训练 minibatch 与优化参数，不使用未查询真值参与训练或门控。D=0 也是预测后取标签。
父类的诊断 hook 默认为空操作；本实验没有启用隐藏真值 Oracle。

原始组完成后，独立的 NativeTrace 校验时间、域顺序、每批样本数、查询 ID、插入顺序、
D=0 返回时序、每步 SGD iteration 与总计数，并计算文件哈希。控制组/方法组校验参考
设置、运行时数据结构，完成后逐条核对反馈和总更新额度；汇总再次检查三组成本一致。
参考不包含图像内容指纹，整个流水线须使用同一份未改动的数据。

## 看什么结果

完成后查看 `summary.md`、`summary.csv` 和 `paired_deltas.csv`：

- calibration_vs_control：最重要的门控增量效果，逐 seed 配对差及正收益 seed 数。
- calibration_vs_original：完整方法相对原始流程的效果。
- control_vs_original：统一训练实现相对原始采样/训练循环的差异。

不能把相对原始组的全部收益归因于门控；控制组还改变了训练抽样方式与空伪标签回退。
同时查看 LabelMe、SUN09、VOC2007、Stream（样本加权）、Macro（三域等权），避免域间
收益和损失互相抵消。成本表报告实际 queried/returned/updates，全部三 seed 成功后
才生成均值 ± 样本标准差；三 seed 不自动证明显著性，同更新次数也不等于同 FLOPs。

各组/seed 包含 settings/result、batches/domains/feedback、updates 和独立日志。
控制与校准组额外记录 replay/calibration、预算和分阶段统计（继承原入口）。
settings 保留实际 SimATTA/FCRA 参数与环境版本。manifest 记录数据路径、权重哈希、
代码提交和运行完成状态。

这是跨数据集迁移检查，不是此前 PACS 调参所用 seed 的独立确认；若据此修改方法，
仍需另留未参与设计的验证运行。当前只检验单个 Caltech101 源方向，不代表全部 VLCS
leave-one-domain-out 结果。

本次只做静态代码阅读、调用链/标签时序/参数/索引/预算/汇总字段检查与 Git 差异检查；
没有在本地运行 Python、测试、离线分析、模型推理或实验。
