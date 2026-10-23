# 第一步：只增加教师置信状态匹配

独立入口，仅运行一个新变体 confidence_match 的 VLCS D=0 三个 seed。
不重跑已有 simatta_original、simatta_control、calibration_only，不做审计选样、
Oracle、随机保护、针对性重放、快速适配器或参数搜索。

```bash
python -u -m experiments.confidence_match.run \
  --dataset-root datasets \
  --reference-dir storage/vlcs_calibration/d0_seeds110_123_456 \
  --gpus 0 1 \
  --output-dir storage/vlcs_confidence_match/d0_entropy_gap005
```

参考必须为完整 VLCS 三组结果。使用其三个 seed、源权重、查询与逐批更新次数，
检查旧设置、文件哈希和标注/优化成本。不修改旧结果，输出必须为新目录。
只运行三个新任务；相对三组旧结果的比较只读 result.json。新目录暂不支持续跑。

## 唯一机制变化

固定教师状态定义为 h(x)=H(p_T(x))/log(C)，取值范围约 [0,1]。
查询时在标签释放前保存状态；伪标签状态来自固定教师原始选样前向，不使用真值。
旧反馈匹配条件：教师预测类别相同，特征余弦>=0.8，再取最多8邻居。
新条件在 top-k 前增加 abs(h_pseudo-h_feedback)<=0.05。
门控原公式、相似度权重、先验4、缓存512及所有训练参数不变。
没有合格邻居时权重为1；只增加这一项匹配，不叠加新的支持量门限或随机审计。
默认0.05在本次验证前固定；不要根据本次测试结果搜索最佳值。

query_teacher_states.csv 保存查询时状态；state_matching.csv 逐更新记录原门控均值、
新门控均值、匹配邻居数及无匹配比例。updates.csv 使用实际新门控。
额外查询教师前向只读取固定模型，不使用标签、不更新模型，也不消耗采样随机数。
但其前向和统计有计算开销，同更新次数不代表同 FLOPs。

summary.md 汇总唯一新变体与复用的旧结果，paired_deltas.csv 按 seed 配对。
优先看相对 calibration_only 的变化，再看相对 control/原始SimATTA是否获得收益。
匹配邻居更少可能只令方法接近无门控控制组；不能把回退等同于更准确的可靠性估计。
本实验不记录隐藏真值的伪标签正确性，因而不能单独证明减少误伤正确伪标签。
三个已用于开发的 seed 不构成独立确认，也不保证提升或学术新颖性。

本地仅静态分析与 Git 差异检查，不执行 Python、模型、测试或实验。
实际实验须在服务器运行；失败保留日志，不重跑任何已有基线。

下一步的更新方向检查及整体放松对照见 [gradient_compatibility](../gradient_compatibility/README.md)。
该入口直接复用本实验的完整结果，不会重跑 confidence_match。
