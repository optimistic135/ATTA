# 双时间尺度 FCRA：等待期适应与返回反馈校准

新增独立入口 `experiments.fcra_dual`，保留现有 FCRA。慢速主模型继承当前 FCRA 的
完整监督记忆、混合重放、固定源教师和教师反馈门控；快速残差适配器只在当前无标签
batch 上训练。返回真值用于更新慢速模型，并在训练前比较两个分支，形成局部融合证据。

只运行 PACS Photo → Art → Cartoon → Sketch，在线 batch=32，D=0/20/50/100。
三个 seed、查询 ID、查询时间和总优化次数取自同一份已经完成的 FCRA 结果。
本次是固定查询计划下的适应机制验证，不验证新的主动查询策略。

## 服务器运行

在仓库根目录、原实验环境中执行：

```bash
conda activate df-atta
python -u -m experiments.fcra_dual.run \
  --dataset-root datasets \
  --reference-dir /path/to/completed-fcra-results \
  --gpus 0 1
```

`--reference-dir` 必须指向此前完整 FCRA sweep 的根目录，不是某个单独 D 或 seed：

```text
completed-fcra-results/
  manifest.json
  D0/seed_.../{settings.json,result.json,batches.csv,feedback.csv,updates.csv}
  D20/seed_.../...
  D50/seed_.../...
  D100/seed_.../...
```

每个 D 都必须有 manifest 中的三个 seed。参考必须是更新后的完整记忆 FCRA，
settings 中含 replay_mix 等设置；不能使用最早硬 Top-16/梯度投影版本。
`updates.csv` 是实际 SGD 记录，不从配置 steps 或旧 replay.csv 的迭代汇总推测次数。

默认复用参考中的三个 seed，不重新抽取。可传 `--seeds`，但必须是相同的三个整数。
可指定 `--checkpoint-dir`、`--output-dir`、`--num-workers`；单卡用 `--gpu-idx 0`。
默认源权重为现有 PACS round1 的 encoder_2/fc_2，SHA256 必须与参考 manifest 一致。
GPU 编号受 CUDA_VISIBLE_DEVICES 影响，单卡和双卡都要求 CUDA。双卡每张卡同时运行
一个完整任务，复用已有总进度条和逐卡逐域进度。

输出默认 `storage/fcra_dual/<UTC时间>/`，已有目录报错，不覆盖参考结果。
不会下载数据、重训源模型或新增其他方法、消融和参数搜索。

## 固定预算与参考校验

父进程在启动 GPU 任务前读取全部 12 份 trace，校验 batch 时序、查询 ID 唯一性、
anchor 顺序、到期/返回时间、反馈计数和 SGD iteration 序列。记录参考文件的 SHA256；
worker 再次核验，避免预检查后参考被改动。当前模型初始化后还会核对实际各域批次数
和每批样本数。所有查询严格复用参考，不重新运行学生选样或 pending 冗余惩罚。

这些检查能核对时钟、结构、配置及权重，但旧结果没有图像内容指纹，因此仍须使用
同一份 PACS 数据及类别目录，不能把“样本数量相同”当作图像内容相同的证明。

令旧每步 SGD 次数为 u_t，T 为流总 batch 数，B=sum(u_t)。D>0 时：

\[
B_A=\min(T,\lfloor B/10\rfloor),\quad B_S=B-B_A.
\]

适配器每步名额：

\[
a_t=\left\lceil\frac{(t+1)B_A}{T}\right\rceil
     -\left\lceil\frac{tB_A}{T}\right\rceil.
\]

因此每步至多一次，若预算非零，第一批预测结束后即有可用名额。
慢速模型按旧累计更新量 U_t=sum_{k<=t}(u_k) 分配：

\[
s_t=\left\lfloor\frac{U_tB_S}{B}\right\rfloor
    -\left\lfloor\frac{U_{t-1}B_S}{B}\right\rfloor.
\]

B=0 时所有名额均为零。D=0 时 B_A=0，逐步慢速额度恰好为 u_t，不构造适配器。

没有可靠无标签样本时跳过适配器，将名额转到下一次有真实标签的慢速更新；若最后
一批仍有退款，则在该批预测结束后用现有标签结清。末批结清不改变任何已记录预测。
结清可能单独重建一次慢速 SGD；其他慢速训练沿用每批重建优化器。末尾严格检查：

\[
\text{actual slow updates}+\text{actual fast updates}=B.
\]

新的损失不决定早停次数。保留主模型原 SGD 学习率 1e-3、momentum=0.9、训练
minibatch=16 和损失，但以参考更新额度代替重新触发早停。D=0 保留旧版结构与额度，
不是承诺跨 GPU、依赖版本和数值环境逐位复现。result.json 的
`reference_d0_batch_correct_match` 只审计每批正确数是否一致，不用于改变训练。

同优化次数不等于同 FLOPs：一次快更新只对小适配器反向传播，但生成视图和评估
反馈需要额外前向。逐批报告慢编码器/固定教师的前向调用与图像数、慢编码器反向调用
与图像数、快更新数、同步后的耗时。时间包含流处理、指标和进度开销，不含模型加载。
这是优化步预算匹配实验，不能据此宣称计算量严格相同。

## 快速适配器

学生编码器 E_theta、分类头 W_theta 和适配器共享主模型：

\[
A_\phi(z)=z+W_{up}\operatorname{GELU}(W_{down}\operatorname{LN}(z)),
\quad p_S=\operatorname{softmax}(W_\theta E_\theta(x)),
\quad p_F=\operatorname{softmax}(W_\theta A_{\bar\phi}(E_\theta(x))).
\]

瓶颈维度16，LN无可训练仿射参数，上投影权重和偏置均为零初始化。
逐样本把残差径向缩放到不超过 0.1||z||_2；零特征对应零残差。
适配器使用 SGD(lr=1e-3,momentum=0)，EMA 动量0.99。

慢速更新后，用当前 batch 的原输入作弱视图；强视图在 ImageNet 均值/标准差反归一化
后的 [0,1] RGB 上做每图 p=0.5 水平翻转和温和颜色扰动，再归一化。
颜色因子为亮度/对比度/饱和度 [0.8,1.2]、色相 [-0.05,0.05]，本批共享颜色因子，
随机操作顺序。特征在 no_grad、编码器 eval 模式下计算并缓存。

EMA 弱视图未经锐化置信度>=0.9 且弱/强视图预测类别相同时 m_i=1。
目标 q 是 EMA 弱视图 logits 经温度0.5锐化后的停止梯度分布：

\[
L_A=\frac1B\sum_i m_i\operatorname{KL}(q_i\|p_\phi(x_i^s))
 +0.1\frac1B\sum_i\operatorname{KL}(\operatorname{sg}(p_S(x_i^w))\|p_\phi(x_i^w)).
\]

分母是完整当前 batch 大小。所有 m_i=0 时跳过，不通过仅做锚定损失消耗名额。
分类头通过 detached 权重作线性映射，梯度只进入适配器；运行时也检查主模型没有梯度。
初始化与视图增强使用独立 CPU Generator，不消耗慢速抽样的随机数。
历史返回图像仅用于专家评估和慢速监督，不用于快适配器训练。

## 返回反馈与分支融合

标签返回后，在当前批任何慢速或快速训练前，对返回图像重新计算当前两个分支：

\[
\Delta_i=\operatorname{clip}(\log p_F(y_i|x_i)-\log p_S(y_i|x_i),-2,2).
\]

同时记录查询时两个分支的真类概率，但它们只供离线审计，不替代返回时的当前比较。
当前预测使用的 gamma 在 release 之前已确定，本批新反馈只能影响以后的预测。

保存最近512条事件，key 为固定源教师归一化特征 v_i；当前样本在余弦>=0.8 的事件
中取最相似8条，计算：

\[
a_i=\exp((\cos(v,v_i)-1)/0.1)\exp(-(t-t_i^{return})/50),
\quad G=\frac{\sum_i a_i\Delta_i}{4+\sum_i a_i},
\quad \gamma=0.5\sigma(G/0.2),\quad p=(1-\gamma)p_S+\gamma p_F.
\]

无证据时 gamma=0.25；有害证据将权重压低；快速分支权重不超过0.5。D=0 固定
gamma=0。校准是局部、有偏且会过时的专家比较证据，不是当前域的无偏精度或
USE−SKIP utility 估计，也不保证未来准确率改善。

## 时序与监督主模型

每步先完整预测并记分，再固定查询、维护原低熵伪标签记忆、释放到期真值。
在更新前比较新返回事件并登记原 FCRA 教师校准；随后按额度训练慢速主模型，再尝试
当前无标签适配器更新。更新查询记忆特征，进入下一批。

慢速模型继续完整记忆混合抽样：75%均匀覆盖+25% need×局部相关性偏好；每次 SGD
重新抽样，监督 CE 不重复加权。伪监督沿用原 FCRA 的教师反馈门控、固定分母、alpha
及空伪标签时的监督损失回退。未返回真值不参与训练或证据，流结束不冲刷队列。
域 ID 仅用于读取既定流、反馈审计和评价，不用于适配器重置、路由或超参数选择。

## 输出与解释

根目录：manifest.json、增量 per_seed.csv、完整12次结束后的 summary.csv/md，以及各
任务日志。manifest 包含参考路径、文件哈希、源权重哈希、新旧 Git 版本和设备信息。
失败停止新任务派发，已启动成功结果仍保存，不生成不满三个 seed 的完整均值表。

每个 D/seed 目录：

- settings.json、result.json：实际设置和完整结果；method=FCRA-dual。
- budget_plan.csv：预先固定的逐步慢/快额度及查询数。
- budget.csv、budget_result.json：实际更新、退款、编码器计算计数、耗时和最终预算审计。
- batches.csv：融合准确率、两个分支各自正确数、gamma、反馈状态和阶段。
- domains.csv：各域在线准确率和域开始后前10批的样本加权准确率。
- phases.csv：首次反馈前、首次反馈后的样本数和准确率。
- feedback.csv：固定计划对应的真实 oracle 审计，完成时严格比对参考。
- expert_events.csv：查询时与返回前的真类概率、返回时增益。
- calibration.csv：继承的固定教师反馈校准。
- updates.csv、replay.csv：慢速梯度范数、损失、覆盖与抽样情况。
- adapter_updates.csv：可靠样本数、是否更新、两项损失、适配器梯度范数。

首次返回那一批的预测也在首次反馈之前，因此计入 before_first_feedback。
准确率都按实际样本数加权；Domain macro 对三域等权。跨域比例仅以已返回标签为
分母，先按 seed 计算再平均，无返回记 N/A。无样本阶段准确率也为 N/A。
首10批评价只作域切换诊断，不作为训练信号。

固定查询计划的作用是排除选样数量/内容变化的混杂；参考计划由旧算法产生，不能据此
宣称新主动策略更好。无标签适应可能仍会错误自训练，EMA、锚定和残差界只提供限制，
不是性能保证。仍需根据 D=50/100 的等待期和反馈后表现判断是否有效。

本次交付仅静态阅读、时序/梯度/预算核对和 Git 差异检查，**没有在本地运行 Python、
测试、模型或实验**。不预填准确率，不宣称已验证提升。
