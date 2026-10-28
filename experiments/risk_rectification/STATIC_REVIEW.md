# 静态审查

未运行 Python、导入、compile、CLI help、测试、实验或训练。以下仅为阅读代码后的检查记录。

| 项目 | 静态依据 |
| --- | --- |
| 三个正式目标 | 原SP；SP+η当前有界软CE；SP+η(当前软CE+加权历史有界真值−软CE) |
| 未到期标签隔离 | 源缓存只收到图像/ID；只从release后的received读取监督；当前真值仅batch-end评价 |
| 时序与时钟 | parent select_queries返回原SP logits；源缓存/诊断在训练前；first-return批仍属before-feedback；不冲刷pending |
| 缓存对应关系 | anchor_index与原trace对应；完整记忆按sorted(received)堆叠q/features；缓存总数核对queried |
| 共同抽样 | 镜像父循环的私有生成器、tpos/ppos顺序；planned hash与formal hash一致；两辅助组使用相同真值抽样估计残差 |
| 原训练保留 | 阴影候选直接调用原_loss；正式基础loss保留pseudo先、truth后的前向顺序，原CE、优化器、权重、配额不变 |
| 原真实监督系数 | 保持1，残差增加有界真实项并减软项；不声称真实监督强度相同，额外系数单独记录 |
| 当前目标 | 原冻结源模型完整软分布，不硬化、不筛置信度、不用未来标签修正 |
| 密度训练 | 只用源归一化特征与历史/当前二元指示，CPU解析梯度、零初始化，无全局随机数和真实类别 |
| 分折与先验 | 两类BCE均值等权；每折用另一折拟合；ratio=exp(logits)；log-odds限幅后截断、自归一化 |
| 稳定性与条件 | 两辅助组共同min-memory/min-current/ESS/正式quota启用；ESS不能证明支持；截断前后ESS均记录 |
| 梯度边界 | 当前q/记忆q/权重都无梯度；新增当前forward可更新原在线模型；残差复用该步truth logits |
| 有界与符号 | 新辅助logp下限log(1e-4)；原CE不改；保留负残差；非有限权重/损失失败 |
| 风险真值只评分 | actual_current_risk来自预更新缓存logits，在batch-end计算；仅写表，不进入训练、阈值或density |
| 不混淆估计和梯度 | 完整记忆用于诊断，minibatch用于训练；risk MAE不是梯度校正证明 |
| 配对验证 | 三组核对reference/formal/pseudo/source-cache哈希、density次数及逐批启用/权重统计；before-feedback预测一致 |
| 三seed统计 | 先seed内聚合与相减，再跨seed样本SD；不将batch/图像当独立seed |
| CSV与resume | 每表固定字段模板；空权重表有header；result最后写；九任务一卡一个；完整跳过、部分归档重跑 |
| 理论边界 | 原主动查询无新探索，feature ratio/clip/有限样本存在偏差；不宣称无偏或双重稳健 |

服务器需验证：实际语法导入/依赖兼容；SP基线复现；计划与实际抽样hash；GPU数值/显存；density权重范围及有效启用次数；九任务汇总和resume；实际在线收益。没有本地运行证据。
