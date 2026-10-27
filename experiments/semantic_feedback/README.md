# 冻结语义分支与延迟反馈融合：PACS 最小实验

本实验检验一个具体假设：**同源伪标签缺少新的纠错信息；独立预训练语义分支能够在等待期间提供互补预测，返回真值进一步决定如何使用这份信息。**

仅新增本目录，不改动 SP、原 SimATTA、既有检查点或结果。这里只做 D=100、三个 seed 的最小研究，不扩展延迟扫描，不做参数搜索。

## 方法

### 1. 保留监督路径，增加冻结语义信息

SP-ATTA 沿用原有完整真值记忆、历史伪标签池、新反馈门控、查询 ID/时间和逐批更新额度。设其当前输出为

\[
p_t(x)=\operatorname{softmax}(f_{\theta_t}(x)).
\]

独立语义分支使用官方冻结 CLIP ViT-B/32。对已知 PACS 类别，用同一个提示 `an image of a {}.` 编码，不使用域名、域边界或目标标签。类别文本特征归一化，得到

\[
q(c\mid x)=\operatorname{softmax}_c\left[s\,\langle
\widehat{E_I(x)},\widehat{E_T(\mathrm{prompt}(c))}\rangle\right],
\]

其中 s 使用检查点自带的 `exp(logit_scale)`，不利用目标真值调温度。图像和文本编码器全部冻结。该分支从第一批开始工作，不等待标签，也不生成训练 SP 的伪标签。

两分支接收相同的在线图像：原加载器已将 RGB 缩放为 224×224 并做 ImageNet 归一化；语义分支反归一化后改用 CLIP 归一化。**这并非在原始长宽比图像上执行官方 Resize/CenterCrop 的标准 CLIP 评测。** 不额外读取原图、增强图像或使用测试域提示。

### 2. 七参数反馈融合

每个类别有一个权重 \(\alpha_c=\sigma(b_c)\)，全部初始化为 0.5。

\[
u_t(c\mid x)=(1-\alpha_c)p_t(c\mid x)+\alpha_c q(c\mid x),\qquad
r_t(c\mid x)=\frac{u_t(c\mid x)}{\sum_k u_t(k\mid x)}.
\]

类别权重不同，因此必须再次归一化。每个当前 batch 使用更新前的参数预测。

在该 batch 预测完成、标签真正到期后，取本批新返回集合 \(R_t\)。在这些标签进入 SP 真值池和参与任何 SP 训练之前，重新计算当前 SP 在返回图像上的预测 \(p_t\)。冻结语义分支的 \(q\) 使用查询时缓存：模型、提示和输入均未改变，故不需要重复编码。

只训练七个 \(b_c\)：

\[
L_F=-\frac1{|R_t|}\sum_{(x_i,y_i)\in R_t}\log r_t(y_i\mid x_i)
+0.1\frac17\sum_c(\alpha_c-0.5)^2.
\]

每个有返回的 batch 做一步 SGD，学习率 0.1、momentum=0。两个分支的概率均停止梯度，SP 获得原有全部训练额度；七参数更新和返回图像额外前向单独计费，不宣称总计算预算相同。无返回时不更新融合权重。

校准完成后，真值进入原 SP 流程。新融合参数最早影响下一批预测。全流连续计时，不按真实域边界重置。

只用本批返回信息，避免大历史池稀释新反馈；向 0.5 的正则减少少量查询反馈使权重剧烈漂移。不过，历史查询存在选择偏差和时滞，**本方法不保证校准能泛化到当前域**。保存查询时预测与返回时预测，二者不混用。

## 四个必要对照

| 预测分支 | 输出 | 要回答的问题 |
| --- | --- | --- |
| SP-ATTA | 原 SP 当前输出 p | 原方法参照 |
| CLIP-only | 冻结语义输出 q | 更强预训练模型本身能贡献多少 |
| Fixed-Fusion | 0.5p+0.5q | 固定集成能贡献多少 |
| Feedback-Fusion | 七参数归一化融合 r | 延迟真值校准是否在固定融合之外有价值 |

每个 seed 只运行一条 SP 训练轨迹，同时在线评分四个分支。融合不反过来改变查询、SP 损失、抽样、伪标签池或模型状态，因此共享轨迹适用于这个预测层面对照。CLIP-only 的预测不用任何目标真值；它在本程序中与 SP 同时计算，是为了统一样本和统计，不表示 CLIP-only 部署需要训练 SP。

固定配置：P(source) → A → C → S，batch=32，D=100，seed=110/123/456。三个 GPU 任务，双卡时每卡至多一个任务。原逐域进度条显示事实 SP 准确率；四分支结果在独立结果表中报告。流末不冲刷 pending 标签。

`--reference-dir` 必须是已完成的原 SimATTA 根因诊断目录，而非 FCRA 或 SP 结果；复用其中 D100 的查询计划、时钟、逐批更新额度和源检查点哈希。它不会为融合提供未来标签。

## 服务器准备

在仓库根目录、现有实验环境内操作。先安装官方 CLIP；下面使用 `--no-deps` 保留原有 PyTorch/torchvision 环境，补齐 CLIP 的轻量依赖：

```bash
pip install ftfy regex tqdm
pip install --no-deps git+https://github.com/openai/CLIP.git
mkdir -p storage/pretrained
curl -fL --retry 3 \
  https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt \
  -o storage/pretrained/ViT-B-32.pt
```

已经有相同文件时直接复用，不需要重新下载。入口必须接收本地文件，并核对官方 SHA256；工作进程不自动下载模型。CLIP 包源码哈希、模型哈希、原检查点哈希、参考文件哈希、实验代码哈希和 PyTorch/CUDA 版本写入记录。

## 服务器运行

```bash
python -u -m experiments.semantic_feedback.run \
  --dataset-root datasets \
  --reference-dir storage/simatta_rootcause/pacs_d100_bs32_seeds110_123_456 \
  --clip-checkpoint storage/pretrained/ViT-B-32.pt \
  --gpus 0 1 \
  --seeds 110 123 456 \
  --output-dir storage/semantic_feedback/pacs_d100_seeds110_123_456
```

源检查点位置不同时加 `--checkpoint-dir 路径`；单卡改为 `--gpu-idx 0`。

中断后在同一命令末尾加 `--resume`：跳过完整 seed，保留并归档不完整目录，然后重新运行该 seed；不恢复半途模型状态。配置、参考或本实验 Python 源码改变时，使用新的输出目录。完成标记为最后写入的 `result.json`，不是日志进度达到 100%。

## 输出与判断

- `summary.md`：四分支各域、宏平均、全流、首次反馈前/后准确率；互补纠错、配对收益和共享反馈/成本。
- `summary.csv`、`per_arm.csv`：分域及前后反馈阶段的逐 seed/聚合统计，每项含有效 n。标准差为三个 seed 的样本标准差，不把图像当作重复实验。
- `paired_gains.csv`：逐 seed、逐域、逐阶段的 Feedback-Fusion 相对三个对照的差值。
- `cost_summary.csv`、`per_seed.csv`：查询、返回、未返回、跨域比例、SP 正式/影子步数、融合步数、CLIP 图像编码数、共享流耗时等。
- 每个 `seed_*` 下：原 SP 的 `batches/domains/feedback/gates/updates/budget/local_future.csv` 与 `compute.json`；新增 `arm_batches.csv`、`fusion_feedback.csv`、`fusion_weights.csv`、`fusion_state.json`。
- `arm_batches.csv` 包含每分支相对 SP 的纠错/致错数量；CLIP 的两项给出独立信息的互补性。当前图像真值仅在评分阶段使用。
- `fusion_feedback.csv` 记录查询时 SP/CLIP/融合的预测和真类概率，以及当前返回图像上 SP/融合的预更新预测。待返回标签不出现在该表。

先看等待期 CLIP 是否能纠正 SP 的错误及固定融合是否受益，再看 Feedback-Fusion 相对 Fixed-Fusion 的逐 seed 收益。若只优于 SP 而不能优于 CLIP-only/Fixed-Fusion，结论只能是额外预训练或固定集成有效，不能声称延迟反馈机制有效。若校准在 Cartoon/Sketch 反复伤害固定融合，应停止扩展，分析标签时滞和查询选择偏差；不要再堆叠筛选模块。

这是一项可失败的最小研究：没有运行结果之前，不预报提升，不宣称新颖性已确立。引入 CLIP 改变了预训练资源；它既有额外计算，也可能有预训练数据重叠风险。与原 ResNet 源模型的比较不属于相同预训练资源的公平优越性证明；CLIP 单独预测和固定融合是必要控制。

官方依据：[CLIP 论文](https://arxiv.org/abs/2103.00020)、[官方仓库](https://github.com/openai/CLIP)、[加载、本地文件、归一化及权重地址实现](https://github.com/openai/CLIP/blob/main/clip/clip.py)。本代码未复制其模型实现，只通过官方包加载。

本地只做静态阅读及差异检查，未运行 Python、测试、实验或训练。
