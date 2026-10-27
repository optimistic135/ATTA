# 静态审查记录

仅依据源代码、复用接口和差异审查；未执行 Python、导入、编译、测试、CLI help 或实验。

| 检查项 | 代码检查依据 |
| --- | --- |
| 未返回标签隔离 | select_queries 只有图像/ID；保护目标来自 online logits.detach；标签仅进入原 oracle 和 batch-end 评价 |
| 原图同视图缓存 | 保护图像和目标来自同一 positions，均不翻转；源翻转只作筛选；目标整个批内不刷新 |
| 原 SP 基础路径 | 父类 group 固定 SP-ATTA；正式 cluster_train/抽样/优化器直接调用父实现；真值权重1、原 λ 门控不改 |
| 只保护正式训练 | protection_training 仅在 cluster_train 范围开启，并要求 model is self.model；finally 清除；影子模型不加 KL |
| 首步近零 | 参考来自本批更新前在线模型；原配置 dropout0/freeze_bn；初始 KL 近零不是额外训练信号 |
| 空集合与无标签 | 空 positions 时保护关闭；无已返回真值则父循环不调用 cluster_train；没有等待期纯 KL 训练 |
| 非有限回退 | 选择概率、KL 和总损失有有限性检查；非有限时失败，不悄悄输出结果 |
| 不扰动抽样 RNG | 平局随机使用专属 CPU generator；额外源/诊断前向 eval；不添加随机增强或优化器 |
| 在线评分时序 | 基础 select_queries 原 logits 返回父循环；当前更新后前向只写诊断，不覆盖正式在线结果 |
| 即时评价真值 | source_correct / pre/post_correct 在 batch-end 获取评价标签，只用于记录和离线汇总 |
| 稳定 CSV 字段 | 新记录分表，所有行同一字段模板；对照无 KL 行时显式保存表头 |
| 配对控制 | 汇总核对 query/return/selection/formal/pseudo 哈希及逐批配额；首次反馈前预测哈希一致；不要求后续门控相同 |
| 统计重复数 | 先聚合每 seed，再算配对差值及样本 SD；未来重叠窗口不用于独立重复推断 |
| 计算成本 | 父 main/source hook 覆盖额外前向、KL 反向；单独记录保护参与的步骤，不将其计为新增 optimizer.step |
| 六任务与断点 | 两组×三seed，一卡一任务；result.json 最后写；输出锁和归档避免直接覆盖；resume 校验设置/哈希 |

服务器仍需验证：源码能否实际导入运行；CUDA/数值稳定性；基线复现和抽样哈希匹配；初始 KL 的量级、后续 KL 变化；六任务汇总与断点跳过；真实在线效果。源模型高置信不保证当前预测正确，保护错误预测和抑制必要纠错均是待检验风险。
