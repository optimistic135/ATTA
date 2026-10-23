# VLCS：门控的更新方向检查

仅新增 direction_check、uniform_relaxation 两组，各三个 seed（从已有实验继承）。
复用 simatta_original、simatta_control、calibration_only、confidence_match 的完整结果，
不重跑这些旧组。D=0、batch=32、源权重、数据顺序、原始 SimATTA 查询和逐批更新次数不变。
熵差阈值继承已完成 confidence_match 实验（当前为 0.05），没有新的参数搜索。

## 运行

在服务器仓库根目录、已有 df-atta 环境中执行。以下命令仅供用户运行；
此次实现没有执行任何本地 Python、测试、训练或服务器任务。

先运行无需数据的分类头公式检查：

```bash
python -m unittest experiments.gradient_compatibility.test_math
```

检查包含：解析梯度与 autograd 的夹角一致性（含/不含偏置）、零梯度回退、
非有限值拒绝、门控公式、抑制强度的可匹配和饱和情况。

然后启动唯一这一项实验，共六次新运行：

```bash
python -u -m experiments.gradient_compatibility.run \
  --dataset-root datasets \
  --reference-dir storage/vlcs_calibration/d0_seeds110_123_456 \
  --confidence-dir storage/vlcs_confidence_match/d0_entropy_gap005 \
  --gpus 0 1 \
  --output-dir storage/vlcs_gradient_compatibility/d0_seeds110_123_456
```

reference-dir 必须指向旧三组实验；confidence-dir 必须指向已完成的置信匹配实验。
两处目录只读，并校验配置、权重、文件哈希及查询/更新预算。缺少完整旧结果时直接报错，
不会自动补跑。输出必须是独立新目录，不允许与参考目录相互包含。

中断后先确保上一轮协调进程及其 worker 均已结束，再使用相同指令加 `--resume`：

```bash
python -u -m experiments.gradient_compatibility.run \
  --dataset-root datasets \
  --reference-dir storage/vlcs_calibration/d0_seeds110_123_456 \
  --confidence-dir storage/vlcs_confidence_match/d0_entropy_gap005 \
  --gpus 0 1 \
  --output-dir storage/vlcs_gradient_compatibility/d0_seeds110_123_456 \
  --resume
```

只跳过存在且通过协议验证的 result.json；未完成任务的目录和日志移入 resume_history，
然后从该 seed 的流起点重新运行，不是从中间模型继续。完整 seed 不重复运行。
代码或参考输入变化时拒绝混用已有结果，需要另建输出目录。
协调进程使用文件锁防止同一输出并发运行；锁由操作系统随进程结束释放。

## 两组实现

### direction_check

保留原来的置信匹配权重 g。使用每次更新原有的已标注训练批次和伪标签训练批次，
从学生实际前向中取得分类头输入 h、logits z；不新增模型前向、不重新采样、不额外反向传播。

对线性分类头，单样本交叉熵的权重梯度为 `(softmax(z)-one_hot(y)) outer h`，
偏置梯度为 `softmax(z)-one_hot(y)`。比较每个伪样本梯度与监督批次平均梯度的余弦 c。
只用已返回真实标签，不读取未查询标签或评价正确性。

```text
conflict = max(0, -c)
final_gate = 1 - conflict * (1 - g)
```

方向一致或无法定义方向（梯度范数过小）时，final_gate=1。
冲突越强，越充分地执行原门控；不增强原有抑制、不反转梯度。
方向和门控均停止梯度。真实标签监督损失、联合损失的 alpha、优化器和训练次数不变。
记录的是当前原始 CE 分类头方向，不包含编码器梯度、SGD 动量或权重衰减。

### uniform_relaxation

先完成全部 direction_check，再运行此组。
读取同一 seed、同一步/迭代的 direction_check 日志，仅取平均移除权重 m：

```text
available = mean(1 - g_control)
scale = min(1, m / available)  # available=0 时 scale=0
final_gate = 1 - scale * (1 - g_control)
```

每次更新使用统一系数，不能利用逐样本冲突信息。若 available>=m，
其 mean(1-final_gate) 与方向检查组相同；无法匹配时保留可行边界并记录差额，
不会为了匹配而加强原抑制。两组逐伪监督更新核验监督/伪标签采样索引哈希，发现差异直接报错。

这是复用另一条训练轨迹统计的诊断对照，不是可独立部署的 ATTA 方法。
系数来自相同时刻已发生的训练，不依据评价准确率或隐藏伪标签正确性选取。
匹配的是权重质量，不是交叉熵数值、梯度范数或优化方向。

## 输出和判断

- summary.md：复用四组、新增两组的准确率及配对差值；门控强度匹配差额和预算。
- paired_deltas.csv、per_seed.csv：配对差值及六次新增运行结果。
- 每个 seed 的 direction.csv：采样哈希、原匹配门控、最终门控、移除权重及目标/实际差额。
  direction_check 另有夹角均值、冲突比例和无效方向比例；control 另有统一缩放系数。
- state_matching.csv 保留置信匹配阶段权重；updates.csv 中 gate_mean 是真正用于训练的最终权重。
- budget.csv 保留编码器调用和时间；分类头解析计算存在额外开销，相同更新次数不代表相同 FLOPs。

优先比较 direction_check 与 confidence_match，再比较 direction_check 与 uniform_relaxation。
若只优于置信匹配、未优于整体放松，不足以支持选择性保护的作用。
若 control 存在较大强度匹配差额，不能声称已排除总体抑制强度的影响。
还需看相对 simatta_control 是否获得稳定收益，而非仅恢复原门控损失。
梯度冲突不等于伪标签错误，方向一致也不保证更新有益。
三个开发 seed 不构成独立确认，不据此声称统计显著或创新性已成立。
