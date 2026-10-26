# 静态审查记录

未运行Python、CLI、编译、测试、推理或训练。按用户要求，仅阅读原SimATTA/DelayedSimATTA、oracle、loader、scheduler及新代码，核对接口和差异；以下不是运行通过证明。

| 项目 | 源码检查 |
| --- | --- |
| 实验范围 | DELAYS仅100，三seed，最多五个预设诊断窗口；不跑其他算法或搜索 |
| 原轨迹 | 继承DelayedSimATTA，不覆盖cluster_train/select_queries；one_step_train仅计数后直接调用官方实现 |
| Snapshot时序 | 在oracle release之后、received.update和当批主训练之前；available显式合并当批已返回标签 |
| 数据对照 | TRUE/MIX真池及draw完全一致；PSEUDO/MIX伪池及draw一致；SKIP不训练 |
| 原alpha | 按完整真/伪记忆计数及cold_start算；空伪池不伪造PSEUDO而是跳过 |
| 优化器状态 | 原官方每批新建SGD；所有诊断分支也从相同空状态开始，lr/momentum沿用 |
| 真标签隔离 | 分支只训练触发时已返回真标签和原伪标签；未来target只用于hook计分 |
| 未来窗口 | 不含触发批；同样的后十批、四分支冻结、不进行后续训练或分支查询 |
| 随机性 | CPU私有Generator采draw；Python/NumPy/CPU及当前GPU finally恢复，避免改变主轨迹采样 |
| 模型状态 | deepcopy完整参数/BN缓冲；哈希检查主模型没有被候选更新；未来branch.eval不更改主模型 |
| 梯度 | 在额外clone上计算全可训练参数梯度；不用head-only梯度冒充全模型；零范数为N/A，非有限报错 |
| 计分 | 改对/改错相对于同快照SKIP、同未来样本；历史准确率明确作为拟合代理 |
| 缺失窗口 | 无条件事件记录skip，未完成未来窗口不进主汇总；无有效事件为N/A |
| 统计 | seed内先配对聚合、三seed样本std；不把多个probe事件当成独立seed重复 |
| 输出保护 | 新目录、完整三结果身份检查、日志保留；无主oracle最终flush |
| 双卡 | 沿用一卡一任务scheduler；分支逐个GPU训练/评分、CPU暂存；不创建其他GPU RNG上下文 |

静态修正包括：移除拷贝入口中不存在的stream模块引用；TRUE/PSEUDO不额外执行未用的前向；原MIX保持source先、truth后；释放候选grad以减小CPU占用；末次窗口放在240附近覆盖可能的Sketch本域返回阶段。

仍需服务器验证：checkpoint和Torch/sklearn版本、原SimATTA早停轨迹、实际触发数、短更新效应大小、显存/耗时及三个seed配对结果。根因结论必须根据结果作条件判断，不提前确认。
