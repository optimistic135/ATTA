# SP-ATTA：两个小实验入口

依据 [实验方案](../../docs/supervision_progress_atta_experiment.md) 新增实现，不修改原SimATTA、旧诊断或旧结果。本地只做静态检查，未执行Python、CLI、测试、推理或实验。

所有任务固定PACS Photo(source) → Art → Cartoon → Sketch、batch32、D=100、seed=110/123/456。每张GPU同时一个任务，显示总任务和逐域进度。默认不增加其他delay、数据集或参数搜索。

## 1. 先补TRUE-scaled诊断（3条轨迹）

```bash
git pull --ff-only origin codex/pacs-delayed-feedback
conda activate df-atta
python -u -m experiments.sp_atta.diagnose \
  --dataset-root datasets \
  --gpus 0 1 \
  --seeds 110 123 456 \
  --output-dir storage/sp_atta_scaled/pacs_d100_seeds110_123_456
```

在原四分支诊断上仅增加TRUE-scaled=alpha×TRUE。保留原主轨迹、五个时刻、相同抽样和十步短更新，在相同后十批上评价冻结分支。原四分支不因新分支改变，额外分支不改变原模型。

先查看summary.md中的TRUE−TRUE-scaled与TRUE-scaled−MIX；完整逐事件数据在TRUE-scaled-diagnosis/seed_*/probes.csv。若结果不支持缩放解释，先分析，不必立刻运行主验证。

此入口不用reference-dir。它的输出根目录不是下述主实验所需的旧原诊断reference。

## 2. 四组主验证（12条轨迹）

需要你已经完成的**原simatta_rootcause输出根目录**，其中含manifest.json与D100/seed_*/的settings.json、result.json、batches.csv、feedback.csv、phases.csv。

```bash
python -u -m experiments.sp_atta.run \
  --dataset-root datasets \
  --reference-dir storage/simatta_rootcause/pacs_d100_bs32_seeds110_123_456 \
  --gpus 0 1 \
  --seeds 110 123 456 \
  --output-dir storage/sp_atta/pacs_d100_seeds110_123_456
```

如果原诊断存放位置不同，只修改reference-dir。不要传FCRA、Relay、CSR或上面的新增scaled诊断结果。参考只读取协议与成本，不读取准确率来决定模型更新。

两入口都支持单卡`--gpu-idx 0`，与`--gpus`互斥。可选`--checkpoint-dir /path/to/photo_checkpoint`，默认沿用round1 Photo checkpoint；主实验会验证与参考的encoder_2.pth、fc_2.pth哈希完全一致。输出目录必须不存在，重跑换目录。

| 组别 | 正式目标 |
| --- | --- |
| Original-MIX | 原alpha×真实CE + (1-alpha)×伪CE |
| TRUE-only | 真实CE |
| Fixed-Aux | 真实CE + 0.25×伪CE |
| SP-ATTA | 真实CE + 本批选定lambda×伪CE |

## 3. 主实验怎样控制变量

- 按原feedback.csv重放查询ID和时间，严格验证到期、返回和跨域事件。
- 按phases.csv逐batch重放原正式更新数，不再让各目标的早停改变成本。
- 同seed四组使用相同原图流、真/伪监督抽样和伪标签池。主汇总会检查正式抽样与伪池SHA256一致，及正式/候选步数一致，否则停止。
- 对照不是逐位复现旧原结果：使用统一预生成抽样和固定成本，Original-MIX应重新测量，不能直接拿旧准确率替代。
- 所有四组都执行相同两候选计算；固定目标组忽略分数，仅SP-ATTA用G决定lambda。
- 每个有新反馈、既往真实记忆和可用伪池的batch，从同一参数/BN快照和空SGD状态做一次TRUE与TRUE+0.25P候选。原SimATTA每批也重建优化器，因此候选初始动量为空。
- 新反馈F排除在候选真实抽样之外；通过跟踪伪样本物理ID，也从候选伪抽样中排除F，避免同一验证图像参与候选更新。F在此前可能作为无标签输入或伪样本出现，不将它称作完全独立测试集。
- 用F的平均CE计算G=CE_F(TRUE)-CE_F(AUX)。G>0.001才允许SP本批使用0.25辅助项，否则0；无新反馈/无旧记忆/无合格伪池时SP回退纯真实监督。
- 候选全部丢弃，再把F全部并入正式真实记忆。候选不偷换成主模型，不减少真实标签使用机会。
- λ没有跨batch沿用：无新反馈的batch中SP使用0。Fixed-Aux仍保持0.25；Original-MIX仍保持原alpha规则。
- 无伪池时正式训练回退真实重放。默认只支持标准PACS dropout=0、freeze_bn=True，避免随机前向差异混入目标对比。

## 4. 额外诊断和开销

在原五个诊断窗口附近，保留已计算的TRUE/AUX候选，冻结并评价其后十个batch。未来标签仅计分，不影响G、lambda或正式模型。local_future.csv记录完整窗口标记、净改对/改错及G，用于检验验证分数是否能预测后续局部收益。

候选比较每个合格反馈batch额外两次短更新，四组都支付；正式训练次数保持参考不变。两者分别计数。TRUE-only每正式步少一次伪前向，因此相同优化步数不等于相同FLOPs；compute.json和budget.csv记录处理图像数与前后向调用，另报运行时间和显存。

## 5. 输出

| 文件 | 含义 |
| --- | --- |
| summary.md / summary.csv | 四组域准确率、Macro/Stream、反馈比例、更新开销与辅助采用率；三个seed均值±样本std |
| per_seed.csv / paired.csv / paired_summary.csv | 逐seed结果、SP相对三个对照的Stream配对差及配对均值/样本std |
| manifest.json | Git、checkpoint哈希、参考文件哈希、固定设置与运行状态 |
| 各方法/seed_*/feedback.csv | 实际查询/返回事件，结束后逐条与参考核对 |
| gates.csv | 新反馈数量、候选CE、G、是否合格、假设门控是否接受、本批选定lambda |
| updates.csv | 每步alpha、真实/伪系数和损失 |
| budget.csv / compute.json | 正式更新与参考逐batch核对、前后向成本 |
| local_future.csv | 两个冻结候选在共同后续样本上的结果及完整窗口标记 |

对照组的gate_acceptance_pct表示“其影子检验本来会接受多少”，不表示该组采用了门控。实际辅助使用看aux_update_pct、updates.csv；Original-MIX的原伪系数不是SP形式的selected_lambda。

返回率=returned/queried，跨域比例=cross-domain returned/returned，pending不入后者分母。没有返回或有效门控时对应比例为N/A，不填0。所有平均与配对统计以seed为重复单位。

## 6. 解释边界

TRUE-scaled补充实验用于区分缩放和伪项作用；SP四组实验用于检验反馈门控是否比纯真实或固定辅助更好。少量新反馈来自历史环境，不等于当前域的无偏验证集。若SP几乎总拒绝辅助，它退化为TRUE-only，不应仅凭准确率较高就声称门控有价值。

这些模块尚未运行验证，不保证性能恢复或新颖性。先看诊断结果，再决定是否运行12条主实验；两个入口互不自动触发。
