# PACS SimATTA 标签延迟反馈实验

新增入口独立于原有 `ATTA`、配置和 README；不改变原始 SimATTA 的调用行为。

## 运行

在安装好原项目依赖的实验机器上，从仓库根目录执行：

```bash
python -m experiments.delayed_feedback.run \
  --dataset-root /path/to/datasets \
  --gpus 0 1
```

双卡时每张卡同时运行一个独立实验，空闲卡自动领取下一个任务。
不是把一个模型拆到两张卡上；总任务数仍为 12，共用同一组三个 seed。
单卡用法保持兼容，将 `--gpus 0 1` 改为 `--gpu-idx 0`；不传设备参数也默认单卡 0。
两个参数互斥，GPU 编号不得重复或为负数。

GPU 编号是当前进程可见设备的逻辑编号。例如设置 `CUDA_VISIBLE_DEVICES=2,3` 后，
仍传 `--gpus 0 1`，对应物理卡 2 和 3。显式使用 `--gpus` 时若 CUDA 不可用，
或某个编号超出可见范围，任务会报错，不会悄悄改用 CPU。

服务器后台运行：

```bash
nohup python -u -m experiments.delayed_feedback.run \
  --dataset-root /path/to/datasets \
  --gpus 0 1 \
  > delayed_feedback.log 2>&1 &
```

### 进度展示

在交互终端前台执行时，默认显示 tqdm 进度条：一条展示全部 12 个实验的完成进度，
每张 GPU 各有一条，展示当前 D、seed、域、已完成/总 batch 数、耗时、预计剩余时间和处理速度。
域内每完成一批，更新累计在线准确率（`acc`）、已返回标签数（`returned`）和等待标签数（`pending`）。
域切换时该卡进度条重新计数；等待模型加载时显示 `loading`。

两个子进程只上报进度，由主进程统一绘制，避免双卡进度互相覆盖。
各实验的原始日志继续保存到 `D*_seed_*.log`。
`nohup` 或重定向到文件时改为逐行进度记录，可用 `tail -f delayed_feedback.log` 查看，
不会在日志中写入用于移动终端光标的多行进度条。

`--dataset-root` 下必须已有 `PACS/art_painting`、`PACS/cartoon`、
`PACS/photo`、`PACS/sketch`，每个目录内是按类别组织的图片。
不自动下载数据，不自动训练源模型。

默认加载仓库现有 round1 的 Photo checkpoint：

```text
storage/checkpoints/round1/PACS/[2, 0, 1, 3]/ResNet18_3l_meanpool_0dp_Truefzbn/0.0001lr_0.0wd/ERM_no_param/
```

也可以用 `--checkpoint-dir /path/to/checkpoints` 指向同时包含
`encoder_2.pth` 和 `fc_2.pth` 的目录。所有 seed、D 都加载同一组权重；
三个 seed 衡量数据流与适应随机性的影响，不包含源模型重训练方差。
原模型构造仍使用 torchvision 的 ImageNet 预训练初始化；离线运行时需预先准备其权重缓存。

首次运行随机抽取三个不重复的 seed，保存到 `manifest.json`，对 D=0、10、20、30
逐一复用，共 12 次运行。每次使用独立子进程；某次失败后停止分派新任务，
允许其他已启动任务结束并保存成功结果，不把未完成的结果当成完整均值。
默认 `--num-workers 0`，该值是每个实验进程的数据加载子进程数，可按实验机器资源调整。

复现实验可显式指定记录下来的三个 seed，并使用新的输出目录：

```bash
python -m experiments.delayed_feedback.run \
  --dataset-root /path/to/datasets \
  --seeds 123 456 789 \
  --output-dir /path/to/new-results \
  --gpus 0 1
```

默认输出目录为 `storage/delayed_feedback/<UTC时间>/`。
已有输出目录会报错，防止覆盖先前结果。运行日志分别保存，终端显示任务进度；
完整成功后打印最终均值表。

## 实验口径

- Photo 源模型，Art → Cartoon → Sketch 连续适应；batch size=100，保留不足 100 张的最后一批。
- 沿用原 `configs/TTA_configs/PACS/SimATTA.yaml` 参数、聚类选样、低熵伪标签分支和训练目标。
- 每域内部按 seed 随机无放回遍历。跨 D 使用相同 seed，因而输入顺序相同；模型演化不同，主动选择的样本和查询数可能不同。
- 本实验保留原 SimATTA 的查询数量机制，没有新增固定查询比例或预算截断。
  原实现的 `atta.budgets: 100` 不在适应循环中执行硬预算限制，不能据此声称恰好查询 100 个标签。
- 每步先预测并记录成绩，再查询和释放反馈，最后训练。时间步从 0 开始，查询步为 t 的标签在 t+D 步预测结束后返回。
  D=0 的标签也不能影响当前批已记录的预测。
- 模型、已返回标签、聚类记忆与等待队列均跨域保留，流结束后不冲刷队列。
- 已查询未返回的样本仅以图像、特征和样本 ID 参与聚类，不以真值或伪真值参与目标监督。
  真实标签由独立 oracle 在到期时返回。新查询的 ID 随原始熵筛选索引传递，避免把过滤后的索引误用于原始 batch。
- 标签未返回时跳过依赖真值的 `cluster_train`；已有真值后每批继续原来的记忆重放。
  不额外发明无监督训练损失。待返回样本也会刷新特征，刷新不需要标签。
- 聚类候选数不足时，将簇数截断至候选数；空候选池不查询。这是对极小/空候选池的防错保护。

## 输出表

`summary.md` 为可直接查看的表格，每一行对应 D，列为：

| D | Art准确率 | Cartoon准确率 | Sketch准确率 | 全流准确率 | 查询数 | 返回数 | 跨域返回数 | 跨域比例 | 未返回数 |
|---|---|---|---|---|---|---|---|---|---|
| 0 / 10 / 20 / 30 | 三个seed均值±标准差 | 同左 | 同左 | 同左 | 同左 | 同左 | 同左 | 同左 | 同左 |

所有准确率与比例使用百分数。准确率基于更新前首次预测；全流准确率按样本数加权，
不是三个域准确率的算术平均。`summary.csv` 另有域宏平均。

跨域比例 = **返回时域不同于查询时域的标签数 / 流结束前已返回标签数 × 100%**。
未返回标签不计入分母；没有任何返回时为 N/A。
先计算每个 seed 的比例再取均值，不用均值计数重新计算比例；
`summary.csv` 提供各指标有效 seed 数 `_n` 和样本标准差 `_std`（ddof=1）。

输出包括：

- `manifest.json`：三个 seed、D 列表、GPU 列表及可见设备配置、Git 提交、checkpoint SHA256 与路径。
- `per_seed.csv`：12 次运行的独立结果，随每次成功运行增量保存。
- `summary.csv`、`summary.md`：四个 D 的均值与标准差，仅完整运行后生成。
- `D*/seed_*/settings.json`：该次参数、实际 GPU 编号和运行环境。
- `D*/seed_*/result.json`：该次汇总结果。
- `D*/seed_*/domains.csv`：各域实际样本数、batch 数、准确率，及按**查询来源域**归属的全流结束时反馈计数。
- `D*/seed_*/batches.csv`：逐批预测与反馈状态，反馈计数为截至当前步的全局累计。
- `D*/seed_*/feedback.csv`：逐标签查询步、到期步、返回步及域信息。
- `D*_seed_*.log`：该次运行的完整日志，失败时先查看对应日志。

## 可选验证

以下检查覆盖时钟、跨域计数、均值汇总与多卡调度，不加载 PACS 或模型；由使用者在实验机器运行：

```bash
python -m unittest experiments.delayed_feedback.test_protocol experiments.delayed_feedback.test_summary experiments.delayed_feedback.test_scheduler
```

本次代码交付按要求没有在本地运行实验或测试；没有预填任何实测准确率。
