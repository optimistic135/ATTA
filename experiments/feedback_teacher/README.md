# Feedback-repaired teacher: PACS D=100

最小三组实验，Photo(source) → Art → Cartoon → Sketch，batch=32，seed=110/123/456；共 9 次。仅 D=100，不增加参数搜索、其他数据集或等待期训练。旧代码和结果不修改。

## 三组及可回答的问题

| 组别 | 辅助信号 | 教师分类头 | 正式额度 |
| --- | --- | --- | --- |
| SP-ATTA | 原 SP 的源教师历史伪标签池 | 不更新 | 原始全部 student 步数 |
| Current-Source | 源教师实时标注最近 4 批图像 | 同样训练诊断头，但绝不使用其目标 | student + head = 原额度 |
| Current-Feedback | 可信时使用反馈修复头标注最近 4 批图像，否则回退源教师 | 已返回真值更新 | student + head = 原额度 |

Current-Feedback 对 Current-Source 是主要比较：相同图像窗口、目标形式、头训练、额度分配与随机抽样，区别是是否允许反馈教师生成目标。两者均保留 SP 的 fresh-feedback 辅助门控。Current-Source 对 SP 联合改变了辅助图像、目标和少量预算分配，不能解释成纯粹的“缓冲区消融”。

## 算法

### 教师修复

保持源编码器 E0 和源分类头 h0 固定，复制一个可训练线性头 hψ。

q0(x) = softmax(h0(E0(x)))

qψ(x) = softmax(hψ(E0(x)))

在有新反馈且额度大于零的批次，把该批最后一个正式更新分配给分类头；其余更新用于学生。分类头从完整已返回真值池随机抽取一个 train_bs minibatch，使用 SGD(lr=0.001, momentum=0)：

L_head = CE(hψ(E0(x)), y) + 0.01 Σ_parameters mean((ψ−ψ0)²).

编码器特征停止梯度。两个 Current 组都执行这一步，Current-Source 的头仅供诊断，避免头训练成本成为两组的混杂因素。头训练不保证每批新标签都被抽中，不增加独立 fresh 重放机制。

### 当前图像学习

最近 4 批无标签图像及其固定源特征构成 FIFO 窗口，不使用域边界清空。弱视图为输入，强视图仅为水平翻转，避免加入额外增强参数。每批用当前头重新计算整个窗口的概率，不积累旧伪标签。

教师弱视图最大概率 >=0.9，且弱/强视图类别一致时，m=1；否则为零。

L_current = mean_batch[m(x) KL(stopgrad(q(x_weak)) || p_student(x_flip))].

使用原始软概率，不锐化；按采样 batch 大小归一化。每次正式 student 更新从窗口以独立私有随机种子重新抽取至多 train_bs 张图像。两个候选使用相同的第 0 次抽样；两组 Current 使用相同的窗口索引计划，掩码及概率允许不同。

保持完整真实标签记忆：L_student = CE_truth + λ_batch L_current。真实监督系数为 1。门控接受时 λ_batch=0.25，否则为 0。缺少新反馈、旧真值、可信当前样本或正式额度时回退纯真实监督。

### 返回反馈的使用时序

1. 当前批在线预测、记录查询时源教师和修复教师概率、构建无标签窗口；此时没有本批新返回真值。
2. Oracle 只释放已经到期的标签。以查询时保存的预测评分，得到严格先预测后见标签的教师准确率诊断。
3. 在任何本批训练之前，计算返回图像上源教师与修复教师的 CE 差。该差只作为下一批的教师选择证据。
4. 使用上一批已有证据选择当前辅助教师；从窗口排除本批新返回图像，用旧真值训练 TRUE 与 TRUE+0.25 CURRENT 两个一步影子候选，再在新返回真值上比较 CE。TRUE 的 CE 减 AUX 的 CE >0.001 时接受辅助学习。候选丢弃，未来评分不参与门控。
5. 将新标签并入完整记忆，按固定额度训练学生，最后执行预留的分类头更新。新头参数和本批新证据从下一批生效。

教师选择证据为返回事件 CE 差的批均值 EMA，衰减0.9；有证据且 EMA>0 时 Current-Feedback 使用修复教师，否则使用源教师。Current-Source 始终使用源教师。它是历史主动查询样本上的保守证据，不是当前域准确率的无偏估计。

## 预算与公平性

使用原 `simatta_rootcause` 的查询 ID、请求时间、返回时间、逐批更新额度和源检查点哈希。跨组校验实际反馈、正式额度、真值/旧伪池抽样索引哈希及固定源伪池哈希。Current 仍构建旧源伪池用于协议校验，但不将其目标用于正式辅助损失。

`main_updates` 表示**正式总步数**，包括 student 和 head；`student_updates + head_updates == reference quota`。SP 保持旧路径，head_updates=0。两组 Current 的 head 更新次数一致。

影子候选另记 `candidate_updates`，有资格的门控批次数可能因可信目标不同而变化；不声称影子成本或 FLOPs 相同。`compute.json` 和 `budget.csv` 记录 main/teacher/candidate 骨干前向与反向调用、图像数及额外 head 调用；耗时和 GPU 峰值内存也输出。分类头梯度不会进入固定源编码器或学生。

无首次返回前不训练，因此 D=100 的 Art 不预期改善。该实验检验返回后的知识利用，不解决缺少及时真值的全部损失。

## 服务器运行

```bash
git pull --ff-only origin codex/pacs-delayed-feedback
conda activate df-atta
python -u -m experiments.feedback_teacher.run \
  --dataset-root datasets \
  --reference-dir storage/simatta_rootcause/pacs_d100_bs32_seeds110_123_456 \
  --gpus 0 1 \
  --seeds 110 123 456 \
  --output-dir storage/feedback_teacher/pacs_d100_seeds110_123_456
```

每 GPU 同时一个任务，保留逐域及总任务进度条。单卡将 `--gpus 0 1` 替换为 `--gpu-idx 0`。可使用 `--checkpoint-dir PATH`、`--num-workers N`。新实验输出目录必须不存在；使用下面的 `--resume` 才能复用旧目录。

### 原目录断点续跑

先确认旧启动器及其子任务已停止，再执行：

```bash
python -u -m experiments.feedback_teacher.run \
  --dataset-root datasets \
  --reference-dir storage/simatta_rootcause/pacs_d100_bs32_seeds110_123_456 \
  --gpus 0 1 \
  --seeds 110 123 456 \
  --output-dir storage/feedback_teacher/pacs_d100_seeds110_123_456 \
  --resume
```

- 按“方法 × seed”判断完成状态，校验 `result.json`、相关输出文件、配置、反馈轨迹和更新预算；完整结果直接跳过，原文件及日志保持不变。
- 只有 batches.csv 或部分诊断 CSV、没有完整最终结果的任务仍属于未完成，从该 seed 的初始源检查点重新执行。当前实现没有逐 batch 的模型/优化器检查点，不能从中间 batch 恢复。
- 先把待重跑任务的旧目录、日志和旧汇总移入同一输出目录的 `_resume_archive/UTC时间/`，再运行；不删除残留数据。
- 所有任务完成后，合并复用结果与新结果，重建同一目录中的汇总表。若全部完成，则只校验并重新汇总，不启动 GPU 子任务。
- seeds、参考文件哈希、源检查点哈希、方法默认参数和数据路径必须一致；GPU 分配及 num-workers 可以调整。
- 记录 resume_history 及旧 manifest，允许本次 CSV 输出修复后的代码复用修复前已经成功完成的任务；不要求 Git commit 相同。若以后改动算法，请使用新输出目录，避免混合不同算法的结果。
- 更新后的启动器使用目录锁防止并行续跑；旧版启动器不认识此锁，必须先手动停止旧任务。

`--reference-dir` 必须是已完成的**原始 SimATTA D100 根因诊断**根目录，含 `D100/seed_110` 等；不是 sp_atta、scaled、FCRA 的结果目录。实验不会调用诊断代码重新生成参考轨迹。

## 输出与判断

根目录：manifest.json、per_seed.csv、summary.csv/md、paired.csv、paired_summary.csv、每任务日志。每组/seed：settings/result、batches/domains/feedback、gates、updates、budget、compute、local_future，以及：

- `teacher_feedback.csv`：查询时两个教师的正确性（返回后评分）、训练前返回 CE 差。
- `current_targets.csv`：可信覆盖率、实际抽样可信数、是否使用修复头、先前反馈证据。

报告各域/全流准确率、返回数和比例、跨域返回数和比例、未返回数，三个 seed 均值±样本标准差。跨域比例只以已返回标签为分母，先逐 seed 算再汇总。

机制判断顺序：教师查询时预测是否变好 → Current-Feedback 是否优于 Current-Source → 是否超过 SP。若修复头未变好，先停止扩展；若修复头变好但学生未提高，再检查覆盖率、门控接受率及局部 AUX−TRUE 收益。

`local_future.csv` 复用五个触发窗口：候选只训练一步后冻结，记录之后10批的 fixed/broken。它比较 CURRENT AUX 与 TRUE，不是 SKIP，也不代表正式整流增益。查询时教师准确率只覆盖已返回的主动查询样本，存在选择与流尾截断偏差；不能称为全流教师准确率。只有三个 seed，不自动作显著性或创新性声明。

**本地未执行 Python、测试、训练或实验；仅完成静态审查。实现有效性和运行兼容性待服务器验证。**
