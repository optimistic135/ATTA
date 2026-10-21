# FCRA-F：固定预算下的新反馈触发监督

独立入口 `experiments.fcra_feedback.run`。只新增本目录，不改原 FCRA/FCRA-G/FCRA-R。
本轮只检验新返回真值的训练份额及真实/伪监督权重，不增加模型、图、残差或参数搜索。

## 服务器运行

在原仓库根目录和已有环境中执行：

```bash
conda activate df-atta

python -u -m experiments.fcra_feedback.run \
  --dataset-root datasets \
  --reference-dir storage/fcra/pacs_bs32_seeds110_123_456 \
  --gpus 0 1 \
  --seeds 110 123 456 \
  --output-dir storage/fcra_feedback/pacs_bs32_seeds110_123_456
```

单卡将 `--gpus 0 1` 替换为 `--gpu-idx 0`，不要同时传两个选项。
若指定 `CUDA_VISIBLE_DEVICES=2,3`，入口仍使用逻辑编号 `--gpus 0 1`。
支持 `--checkpoint-dir`、`--num-workers`；未传 seeds 时自动复用参考的三个 seed。
不改变参考 seed；不传输出目录时使用 `storage/fcra_feedback/<UTC时间>/`。
已有输出目录报错，不覆盖旧实验。双卡每卡同时一个任务，保留总进度及逐域进度条。

参考必须是完整记忆版本的原 FCRA 根目录，包含 manifest.json 和
D0/D20/D50/D100 各三个 seed 的 settings/result/batches/feedback/updates 文件。
不能以 FCRA-G、FCRA-R、fcra_dual 的结果代替。参考文件及源权重均校验 SHA256。

## 固定设置与预算

- PACS，Photo 源模型，Art → Cartoon → Sketch；online batch=32。
- D=0/20/50/100 × 同三个 seed，共12次，全部从同一源检查点独立开始。
- 模型、固定源教师、冻结 BN、完整监督记忆、原混合重放和教师门控保持原配置。
- 保留训练 minibatch=16，SGD lr=0.001、momentum=0.9，每批重建一次优化器。
- 查询 ID、查询步、返回步和逐批优化次数都复用原 FCRA，不重新选样或早停。
- 每批先预测，再释放反馈，再训练；跨域连续计时，结束不冲刷 pending。

记参考第 t 批实际更新数为 u_t。新方法也恰好执行 u_t 次 optimizer.step。
前置新反馈阶段替换现有额度，不增加步骤；早停不再根据新损失重新决定预算。

固定的是优化步骤及监督 minibatch 大小，不保证同样的随机样本、FLOPs 或耗时。
诊断会在 D>0 时额外前向当前 batch，但没有额外反向。
数据流结构会校验；旧结果没有图像内容哈希，仍需使用同一份 PACS 与类别目录。

## 新反馈份额

### 待处理新反馈池

真实标签到期后才加入完整监督记忆，并登记到新反馈等待池。
等待池只含已返回真值，不含 pending 请求。

若本批无更新额度，保留等待状态，直至下一次有额度的批次。
获得新反馈抽样的标签在本批训练完成后移出等待池，仍永久保留于完整监督记忆。
若额度太小，尚未获得新反馈抽样的标签继续等待，不补发额外更新。
流末仍等待的标签据实记录，不执行额外结算。

### 批内更新分配

当 D>0、存在待处理新反馈且 u_t>0 时：

\[
u_t^{new}=\lceil0.25u_t\rceil,\qquad
u_t^{memory}=u_t-u_t^{new}.
\]

前 u_t^new 次更新使用：

- 8个待处理新反馈样本；
- 8个不在本次等待池中的历史真值样本。

历史样本按原 FCRA 的混合重放概率在历史子集上重新归一化抽取。
新反馈按独立随机生成器打乱后循环抽取，在可用槽位允许时先覆盖全部新反馈再重复。
没有历史样本时，16个监督槽位全部使用新反馈。

后续 u_t^memory 次更新继续从全部已返回记忆抽取16个样本，包括本批刚返回的标签。
没有待处理新反馈时，全部额度用于完整记忆重放。

两阶段共用本批同一个 SGD 优化器，阶段切换不清空 momentum。
新反馈随机排列使用私有 CPU Generator(seed+31817 mod 2^32)，不额外消耗主重放 RNG。
但 D>0 的采样策略已改变，其后续抽样序列不要求与旧算法逐次一致。

## 真实监督与伪监督分离

D>0 时，有有效伪标签记忆采用固定归一化损失：

\[
L=\frac{L_{true}+0.25L_{pseudo}}{1.25}
 =0.8L_{true}+0.2L_{pseudo}.
\]

无有效伪标签记忆时：

\[
L=L_{true}.
\]

L_true 为16个真实监督样本的平均 CE。有历史样本的新反馈更新中：

\[
L_{true}=0.5L_{new}+0.5L_{history}.
\]

没有历史样本时 L_true=L_new；完整记忆更新中 L_true=L_memory。
伪标签仍从原低熵教师记忆抽取16个样本，沿用原反馈门控。
门控后仍使用固定 minibatch 分母，不将弱门控重新归一化。

真实监督权重因此不依赖两种记忆池大小。默认有伪标签时新反馈/历史/伪监督
有效系数为0.4/0.4/0.2；无历史时为0.8/0/0.2。
日志保留旧 reference_alpha 供诊断，但 D>0 不用它训练。
默认值是一组固定实现选择，没有经过本地或目标域调优，不保证有效。

## D=0 旁路

D=0 不建立新反馈等待池、不执行私有反馈排列、不更改原损失权重。
保持原监督抽样及 alpha=N_returned/(N_returned+N_pseudo)，包括原 cold_start 规则。
仍执行参考逐批更新数，并生成反馈抽样审计；不进行新增更新后模型前向。

这保持的是原方法路径与目标函数，不承诺跨环境逐位复现。
budget_result.json 中 d0_batch_correct_match 审计是否逐批正确数一致，结果不参与训练。

## 输出与评价含义

根目录输出 manifest.json、增量 per_seed.csv、完整12次结束后的 summary.csv/md、日志。
均值与样本标准差使用相同三个 seed，空指标记 N/A，summary.csv 保留每项有效 n。
summary.md 包含原准确率/成本、两类更新次数、反馈覆盖、真实监督权重、反馈前后准确率。
paired_stream_gain_pp 在每个 seed 内计算本次与原参考的流准确率差，再汇总均值和样本标准差。

每个 D/seed 输出：

| 文件 | 内容 |
| --- | --- |
| settings.json/result.json | 配置、源与参考信息、正式在线结果 |
| reference_budget.csv | 运行前固定的原逐批查询数与优化步数 |
| budget.csv/budget_result.json | 实际逐批额度、阶段计数与最终严格审计 |
| updates.csv | 每一步 fresh/memory batch、阶段、真实/伪监督系数、分支损失 |
| feedback_exposure.csv | 每个返回标签的新反馈抽样、历史抽样、返回当批历史抽样、首次使用位置 |
| replay.csv | 完整记忆大小、阶段额度、覆盖率与原 need/relevance 统计 |
| online_comparison.csv | 当前更新前预测与同批旧参考正确数、反馈阶段、上一批返回数 |
| post_update_probe.csv | 同批更新前后修改、修复、破坏、错到错及净纠错 |
| batches.csv/domains.csv/feedback.csv | 原在线记录、各域统计、查询/返回审计 |
| calibration.csv/query_diagnostics.csv | 原教师校准和固定查询记录 |
| costs.json | 编码器前向/反向调用及图像数、时间、显存 |

### 同批净纠错：回看诊断

D>0 每批完成全部更新后，用 eval/no_grad 再预测当前 batch：

\[
net\_correct=repairs-harms=correct_{post}-correct_{pre}.
\]

评价真值只在 on_batch_end 中统计，绝不传入训练、抽样、额度或权重决策。
当前批正式预测在任何反馈/更新之前已固定，更新后结果不替换在线得分。
当批变化包含新反馈、历史重放和伪监督的共同作用，不是新标签单独的因果收益，
也不是 USE−SKIP utility。new_returns=0 的批次仍记录，便于区分情形。
D=0 的 post_update_probe.csv 只有表头。

### 下一批在线比较：实际更新前预测

after_return_online_gain_pp 只汇总“上一批有新标签返回”的当前批次：

\[
100\frac{\sum_t(correct_t-correct_t^{reference})}{\sum_t samples_t}.
\]

它比较本方法与参考的真实在线正确数，是配对差值。
双方模型历史已经不同，不能把差值归因于上一批某个标签的单独作用。
首次返回那批本身仍归入 before_first_feedback；后续批次归入 after_first_feedback。

## 静态检查与限制

- 只读取已释放标签；重复释放、查询计划或预算偏离均报错。
- 每个返回标签都有 exposure 记录；抽样总数严格等于 train_bs × 参考总更新数。
- 空伪记忆回退真值训练；空历史池使用全部新反馈；零额度不追加更新。
- 不用真实域 ID 重置模型、分配训练或选择权重。
- 训练前预测的标签只用于 oracle 和评价；同批更新后真值仅进入隔离诊断。
- 本版本同时改变新反馈份额和池大小权重，若有效不能单独归因于其中一项。
- 历史反馈可能与当前环境错位，提高真值权重仍可能产生负迁移，不能预先保证提升。
- 首次标签返回前主模型不更新；该版本不声称改善 D100 的无反馈 Art 阶段。

本次只做静态阅读与差异检查，未在本地运行 Python、测试、推理或训练。
