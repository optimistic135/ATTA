# VLCS SimATTA 标签延迟反馈实验

与 PACS 使用同一套延迟协议、调度、进度条和统计代码，不修改原始 ATTA 源码或官方配置。

## 服务器运行

从仓库根目录，在安装好原项目依赖的环境中执行：

```bash
python -m experiments.delayed_feedback.run_vlcs \
  --dataset-root datasets \
  --gpus 0 1
```

也可以使用统一入口：

```bash
python -m experiments.delayed_feedback.run \
  --dataset VLCS \
  --dataset-root datasets \
  --gpus 0 1
```

单卡使用 `--gpu-idx 0` 替换 `--gpus 0 1`。前台显示总实验进度条及每张卡的域内 batch 进度条，
包括 D、seed、准确率、已返回和等待标签数。后台运行时输出逐行进度：

```bash
nohup python -u -m experiments.delayed_feedback.run_vlcs \
  --dataset-root datasets \
  --gpus 0 1 \
  > delayed_feedback_vlcs.log 2>&1 &
```

## 数据和初始模型

`--dataset-root` 指向 VLCS 目录的父目录；接受以下两种域目录命名：

```text
datasets/VLCS/{C,L,S,V}/<类别>/<图片>
datasets/VLCS/{Caltech101,LabelMe,SUN09,VOC2007}/<类别>/<图片>
```

原始数据加载器按域目录名排序分配编号。入口会检查完整域目录列表，
防止多余目录或不支持的命名导致源域 checkpoint 与实际数据错配。
不自动下载数据，不自动训练源模型。

默认使用现有 round1 Caltech101 源模型：

```text
storage/checkpoints/round1/VLCS/[0, 1, 2, 3]/ResNet18_3l_meanpool_0dp_Truefzbn/0.0001lr_0.0wd/ERM_no_param/
```

目录需同时包含 `encoder_0.pth` 和 `fc_0.pth`。
可用 `--checkpoint-dir /path/to/weights` 指定其他同架构 Caltech101 源模型。
全部 12 次运行加载相同权重，不对三个 seed 分别重训练源模型。
原模型构造需要 torchvision 的 ImageNet 权重缓存，离线服务器需提前准备。

## 配置与协议

| 项目 | 设置 |
|---|---|
| 源域 | Caltech101（C，编号 0） |
| 目标流 | LabelMe（L）→ SUN09（S）→ VOC2007（V） |
| batch size | 100，保留最后不足 100 张的一批 |
| 延迟 D | 0、10、20、30 个 batch |
| 随机 seed | 随机生成 3 个，不同 D 复用，共 12 次运行 |
| 反馈规则 | t 步查询，t+D 步预测结束后返回 |
| 跨域比例 | 跨域返回标签数 / 已返回标签数 |
| 流结束 | 不补发未到期标签，单独统计等待数量 |

“相同配置”指上述实验协议相同。方法参数读取官方
`configs/TTA_configs/VLCS/SimATTA.yaml`，因此保留 VLCS 的 `cold_start=100`、
`el=1e-3`、`steps=100` 等配置，不将 PACS 的方法参数覆盖到 VLCS。
其中 `steps` 是否限制训练迭代仍按原 SimATTA 的实现执行。

与 PACS 一样，保留官方聚类查询数量机制；`budgets: 100` 不构成原循环中的硬预算限制。
模型和等待队列跨域连续保留；只有已返回真值参与监督训练；准确率来自每批更新前的首次预测。
复用 PACS 的查询 ID 去重保护：多个聚类中心指向同一图片时只查询一次，并合并聚类权重。

若要与某次 PACS 运行使用相同的 seed，可读取其 `manifest.json`，再显式指定：

```bash
python -m experiments.delayed_feedback.run_vlcs \
  --dataset-root datasets \
  --gpus 0 1 \
  --seeds 123 456 789
```

## 输出

默认保存到 `storage/delayed_feedback/VLCS/<UTC时间>/`，支持 `--output-dir` 指定新的目录。
已有目录会报错，不覆盖 PACS 或历史结果。

`summary.md` 与 `summary.csv` 各有四个 D 的结果，显示三个 seed 的均值和样本标准差。
分域列为 LabelMe、SUN09、VOC2007 准确率，另有全流准确率、查询数、返回数、
跨域返回数、跨域比例及未返回数。全流准确率按样本数加权。
没有已返回标签的跨域比例记为 N/A。

同时保存 `per_seed.csv`、`manifest.json`，以及各实验的 `settings.json`、`result.json`、
`domains.csv`、`batches.csv`、`feedback.csv` 和日志。`domains.csv` 包含实际样本数与 batch 数。

可由使用者在服务器上运行以下检查，不加载模型或数据：

```bash
python -m unittest experiments.delayed_feedback.test_vlcs experiments.delayed_feedback.test_summary experiments.delayed_feedback.test_protocol experiments.delayed_feedback.test_scheduler
```

本次交付按要求没有在本地运行实验或测试，未填入任何实测结果。
