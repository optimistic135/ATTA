# 即时反馈：局部门控与整体降权对照

研究问题：有限主动反馈是否提供了有价值的逐样本伪监督权重，还是整体降低固定源模型
稳定参考的损失强度即可解释收益？本入口只增加两个对照，不改变原查询策略、
门控公式、学习率、混合系数、记忆容量或训练次数，不设计延迟反馈。
内部沿用旧代码的 `teacher` 字段名，实际指固定源模型提供稳定参考。

## 两个新组

每次 SGD 更新先在**该组自身轨迹的当前伪标签 minibatch**上计算原门控 g：

| variant | 实际训练权重 | 保持什么 |
| --- | --- | --- |
| uniform_gate | 每个位置均为 mean(g) | 当前批平均权重 |
| shuffled_gate | 随机排列当前批 g | 当前批完整权重分布及平均权重 |

使用固定源模型原低熵伪标签、完整真实记忆均匀有放回重放、原联合损失及固定查询/更新
记录；无快速适配器、针对性重放、置信匹配、梯度门控或超参数搜索。
打乱使用独立 CPU Generator（seed+93471），不消耗主训练 RNG；每次更新重新打乱，
允许随机排列恰好保留原位置，不人为强制错配。按抽样位置打乱，重复图像可以获得不同权重。
没有伪标签时不执行变换；原始门控全为1或仅有一个位置时，对照自然退化。

每次变换断言平均权重保持（浮点容差1e-6）。汇总还核对两个新组全程真实/伪标签
抽样序列指纹，发现不一致拒绝生成成功汇总。该指纹没有与未记录这些指纹的旧基线比较。
所有预测先计分，再查询取标签、更新证据及学生。

## 服务器准备

在仓库根目录、原实验所用 Python/CUDA 环境运行。先同步当前分支：

```bash
git switch codex/pacs-delayed-feedback
git pull --ff-only origin codex/pacs-delayed-feedback
```

可先在**服务器**执行无需数据/CUDA的变换公式检查（仍需原环境的 torch/torchvision 等导入依赖）：

```bash
python -m unittest experiments.gate_controls.test_math
```

以下数据根目录必须与旧实验 manifest 记录的绝对路径相同；示例 `datasets` 仅适用于
旧实验确实使用它的情况，若旧实验使用 `storage/datasets`，应相应替换。
`--num-workers` 默认0，必须与旧实验一致。默认自动选用各数据集既有 round1 源权重；
可用 `--checkpoint-dir` 指定相同权重，SHA256必须一致。不下载数据或训练源模型。

## PACS：新增6次，不重跑旧消融

需要两个已有目录：

1. `--reference-dir`：原完整基础 FCRA sweep，含 manifest 与 D0/seed_*；继承旧格式校验，
   要求原四延迟 sweep 已完成，**本入口只读取 D0，不运行其他延迟**。
2. `--baseline-dir`：由该参考生成的已完成 FCRA-ablation 实验根目录，含
   simatta_control/seed_* 和 calibration_only/seed_*，不能只提供 summary.csv。

把下面两个 `/path/to/...` 替换为真实目录：

```bash
python -u -m experiments.gate_controls.run \
  --dataset PACS \
  --dataset-root datasets \
  --reference-dir /path/to/completed-fcra-sweep \
  --baseline-dir /path/to/completed-pacs-ablation \
  --gpus 0 1 \
  --output-dir storage/gate_controls/pacs_d0
```

继承参考中三个 seed，Photo → Art → Cartoon → Sketch，online batch=32。
仅调度 uniform_gate/shuffled_gate 各3次。不会启动旧基线或 simatta_original。
旧原始 SimATTA 的独立查询/更新预算可能不同，因此不将它纳入四组门控归因表。

## VLCS：新增6次，不重跑旧三组

`--reference-dir` 是已有 VLCS-calibration 三组完整结果根目录；同时用作 baseline-dir，
无需另传。必须包含 simatta_original 的原生 trace 与两个受控组逐 seed 文件。

```bash
python -u -m experiments.gate_controls.run \
  --dataset VLCS \
  --dataset-root datasets \
  --reference-dir storage/vlcs_calibration/d0_seeds110_123_456 \
  --gpus 0 1 \
  --output-dir storage/gate_controls/vlcs_d0
```

Caltech101 → LabelMe → SUN09 → VOC2007，online batch=32；使用旧三个 seed 和原生查询/
更新记录。数据目录、源权重、SimATTA 参数及训练 minibatch 与旧运行相同。

两个数据集命令依次执行以免占用同一组 GPU。单卡将 `--gpus 0 1` 改成 `--gpu-idx 0`。

## 失败与续跑

原命令末尾增加 `--resume`，只复用通过身份、配置、参考、预算、反馈时序和必需文件
校验的 result.json。无完成标志的任务从流起点重跑，部分目录/日志先移入
resume_history/<UTC时间>/；不是从中间模型恢复。原汇总先备份，不覆盖旧输入。
输出目录独占文件锁阻止两个协调器同时写入。

首次运行输出必须是新目录，并且不能与数据、权重或参考目录相互包含。
旧输入文件哈希和 ATTA/experiments Python 源码、两份 SimATTA 配置的哈希写入协议；
worker 开始、结束及最终汇总核验。代码/输入改变后不能续跑混合结果，需新目录。
更换 GPU 分配允许；保持相同软硬件环境有利于可重复性。失败后协调器停止派发新任务，
等待已经运行的任务结束，保留成功结果，下一次续跑不会重跑有效完成项。

## 输出与分析顺序

根目录：

- summary.md/csv：四组逐域、Stream（样本加权）、Macro（域等权）及成本；均值±样本标准差。
- paired_deltas.csv：按同seed配对的百分点差，重点看 calibration_only − uniform_gate、
  calibration_only − shuffled_gate，并检查逐域结果。
- all_results.csv：两个只读基线和两个新组的完整结果字段；per_seed.csv仅包含新组。
- diagnostics_summary.csv：新组按 seed/当前域/伪标签正确性分组，分别给出图片等权与抽样加权结果。
- manifest.json：输入/代码哈希、参考协议、提交、运行状态；每组独立日志。

每个新组/seed沿用 batches/domains/feedback/updates/calibration/replay/budget 等原始诊断，
并增加：

- gate_updates.csv：每次有伪监督的更新记录原/实际平均门控、平均权重匹配误差、
  原门控反馈覆盖率、正确/错误伪标签抽样数及相应权重总和、实际移除CE总量。
- gate_samples.csv：按当前训练域、伪标签来源域、sample_id汇总的抽样次数、
  原/实际平均门控、平均移除CE、反馈覆盖率。同一图像可以在多个当前域出现。
- result.json：增加完整训练抽样指纹和伪标签抽样位置总数。

覆盖定义为有至少一个“同源模型预测类别且cos>=0.8”的反馈近邻，不能用 g<1 代替：
存在全部正确的邻居时 g仍为1。coverage表示被抽到的伪标签覆盖，不是整个未查询流的覆盖。
正确性通过独立评价字典记录，实际权重在读取该字典前已经计算完成。
隐藏真值不用于匹配、变换、抽样、模型更新、预算或选择超参数。

如果没有伪监督更新，诊断表只包含 draws=0 占位，不能当作零错误或零风险。
缺少正确或错误样本时，相应分组不输出，不补0。

## 解释边界

1. 两个新对照保持的是**各自轨迹、各自当前批次**原始门控的权重质量；适应后轨迹不同，
   不保证与旧 calibration_only 的门控均值相同，也不匹配CE值或梯度范数。
   因而这些是机制对照，不能声称严格排除了所有全局强度效应。
2. 旧基线只读复用；没有为它重建缺失的样本正确性诊断。新组记录的 raw_gate
   也不能冒充旧 calibration_only 轨迹的权重或正确性。
3. 固定查询隔离学习机制，不能证明新的主动查询策略更优。
4. 正确/错误权重差异只是选择性描述；局部门控超过统一和打乱对照、且有配对预测收益，
   才能更有力支持选择性使用稳定参考。三开发seed不自动证明统计显著或泛化。
5. 同更新次数不等于同计算量；诊断包含额外特征相似度统计和写盘开销。
6. 原trace没有图像内容指纹，须确保数据文件及类别目录未改动；仅校验路径不能证明内容一致。

## 本地检查范围

本次仅进行静态代码阅读、调用路径/源模型角色/标签隔离/采样随机数/索引/配置/
预算/结果字段检查与 Git 差异检查。本地未运行 Python、公式检查、测试、训练、
推理、数据分析或服务器任务。服务器实际正确性与性能待运行验证，不预填结果。
