# SP-ATTA 静态审查记录

未执行Python、CLI、编译、测试、推理或实验，仅阅读源码和检查差异；以下不是运行通过记录。

| 项目 | 静态依据 |
| --- | --- |
| 范围 | 新增sp_atta包；两个入口独立；诊断3任务，主实验4组×3seed；均仅D100 |
| 原文件 | 原SimATTA/rootcause不改；scaled用子类补第五分支，不修改原模块ARMS全局变量 |
| Scaled配对 | 从原probe_draws还原同一真标签索引、相同alpha和snapshot；不推进原诊断采样Generator |
| 原四分支 | 仍由原RootCause方法创建；新增分支在原RNG隔离范围内；所有五分支同后续窗口 |
| Reference | 要求原诊断complete、正确method/seed/delay、checkpoint哈希匹配；核对查询、返回、phases成本 |
| 协议重放 | 查询收到ID而非truth；实际oracle事件全表核对；正式步数逐batch与参考一致 |
| 伪样本身份 | sample_select旁路记录所选stream IDs；与伪池长度对齐；Fresh ID从候选伪抽样排除 |
| Holdout | 候选真实抽样只读此前received；新返回F只做候选验证，随后全部加入正式记忆 |
| 状态隔离 | 候选deepcopy参数/BN；新建SGD，空动量；RNG finally恢复，哈希检查主模型未改 |
| 权重 | TRUE/Fixed/SP真实系数1；Original原alpha；SP无合格新反馈时lambda=0 |
| 控制计算 | 四组均执行两候选；独立候选与正式Generator；所有组都生成真/伪正式索引，即使某损失不用 |
| 反向 | loss有限检查；实际采用的真/伪目标才反向；没有把验证标签送入候选loss |
| 更新预算 | 正式计数数组逐batch检查；候选另计；不声称FLOPs相等 |
| 未来标签 | local_future仅记录冻结候选的输出，不读回gate或formal update；不完整窗口不进入局部汇总 |
| 汇总 | 方法×seed完整性检查；主实验抽样/伪池hash与候选/正式步数跨组一致；参考变化拒绝汇总 |
| 比率 | 门控无有效样本为N/A；辅助实际采用与假设门控采用分开；逐seed均值样本std |
| 设备与保护 | 一卡一任务、分支CPU暂存逐个GPU；新输出目录；保留原outputs；无主oracle末尾flush |

静态修正：标准PACS dropout/BN检查；候选验证非有限分数报错；正式计数不重复扫描整表；局部观察加完整窗口标记；reference核对训练minibatch。

仍需服务器验证：原参考settings字段兼容、跨GPU数值/伪池一致性、Torch版本与source checkpoint、候选G的噪声、实际资源开销、TRUE-scaled效果和SP的配对收益。不能仅凭静态检查确认算法有效。
