# 静态审查记录

审查方式：阅读现有加载器、ResNet、FCRA、oracle、调度与进度代码，逐项检查新源码、导入和差异。未执行 Python、CLI、编译、测试、推理、训练或服务器实验。以下是源码检查结论，不是运行通过记录。

| 检查项 | 源码依据与边界 |
| --- | --- |
| 只跑所需任务 | run.py 固定 DELAYS=(0,50,100)，三个公共 seed，共九个任务；没有基线或消融调度 |
| Source 初始化 | 复用严格 Photo checkpoint 加载，保持配置 test_envs=(2,0,1,3) 以正确初始化；独立 STAGES 决定五阶段在线流 |
| 重复域不覆盖 | stages 使用 Art_1 / Art_2；domains 额外合并 Art；不存在以 Art 单键覆盖两次结果 |
| 重复图像身份 | 读取 FastDataLoader 的实际 sampler 顺序，域内索引加域偏移；查询排除已查询物理 ID |
| 标签隔离 | 选样、观察、审计均不接收 evaluation_labels；真实标签只用于计分及 oracle.query；未到期标签不进入 received |
| D=0 顺序 | 先主预测，再构造审计预测，随后 oracle.release；当批新标签不能训练该审计 |
| 审计参数隔离 | 每个候选 deepcopy 同一主模型；复制 optimizer.state_dict；分支丢弃，hook finally 移除 |
| 随机状态隔离 | fork_rng 隔离 CPU 和当前 GPU，Python/NumPy finally 恢复；独立审计抽样 Generator；不操作其他 GPU 的 RNG |
| 候选可比 | 共享预先准备的历史/伪监督 minibatch、初始参数、动量、随机状态与短更新次数；动作 0 也是一次更新 |
| 奖励时序 | on_labels_released 先读取固定软预测算奖励、记录此前估计，再 policy.add；不回写查询时类别 |
| 统计与梯度 | EMA 统计 no_grad；迁移 hook 返回含历史特征梯度的 Tensor；原始与迁移 CE 保持 rho=0.5 |
| 回退 | 无记忆不更新；无同类可信风格不迁移/审计；无证据/支持不足默认动作 0 |
| 预算 | 主步数=30−实际审计步数；每批和全流检查上限；无标签额度不回填 |
| 全部记忆 | received 只增加；内存缓存按返回数量失效；源伪标签仍使用既有 FCRA 机制 |
| 连续协议 | 不按域重置任何学习状态；末尾无 oracle flush；未返回审计也不会形成 reward |
| 反馈口径 | 语义域判定 cross_domain，visit 判定 cross_stage；返回分母排除 pending，比例按 seed 后汇总 |
| 汇总完整性 | 检查全部九个 (delay,seed) 身份与 method；样本标准差；无返回 N/A |
| 设备入口 | GPU 参数互斥，非负且唯一，worker 验证 CUDA；复用每卡一个进程的 scheduler |
| 计算日志 | 计数键在初始化固定，避免 budget.csv 后期新增字段；临时候选单独计数 |

## 静态修正

- 编码器计算计数字段初始化为固定 schema，避免第一批无梯度、后期有梯度造成 CSV 字段不一致。
- 候选 CUDA RNG 只处理当前设备，避免各 worker 为其他 GPU 创建额外上下文。
- 缓存完整已返回图像记忆，避免每个优化步重复复制整个历史图像池。
- 审计优先借用可信同类风格，避免要求高熵主动查询样本自身必须高置信。
- 记录每个候选动作的预估奖励，而不把所选动作的预测值重复当作所有候选预测。

## 尚需服务器验证

- 当前环境的 Torch/torchvision/sklearn 版本、checkpoint 兼容性和实际 CUDA 内存占用。
- FCRA 聚类在五阶段长流中的耗时和数值行为。
- 审计是否有足够同类历史监督与可信当前风格；长延迟会进一步推迟奖励可用时刻。
- 候选短更新奖励是否能预测实际混合记忆训练效果。
- 风格统计迁移的类别保持性、策略是否退化为动作 0，以及最终在线准确率。
