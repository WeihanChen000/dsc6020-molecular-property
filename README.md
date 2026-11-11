# 分子性质预测与结构泛化 · DSC6020

我们想知道，遇到训练时没见过的分子骨架，哪种模型还能把性质预测准。我们比较了随机森林（RF）、从分子图学习的 D-MPNN、联合微调的分子语言模型 MoLFormer，以及用 4-bit QLoRA 微调的 Qwen3-1.7B。

## 三个任务的正式测试

D-MPNN 在六种设置中有五种取得最低的平均测试 RMSE 和 MAE。亲脂性的骨架划分最稳，三个划分种子上都是 D-MPNN 第一、MoLFormer 第二。ESOL 的骨架划分最接近，RF 和 D-MPNN 的平均 RMSE 只差 0.0046 logS。

| 性质 | 划分 | RF | D-MPNN | MoLFormer | Qwen |
|---|---|---:|---:|---:|---:|
| ESOL | Group Random | 0.7076 | **0.6022** | 0.6867 | 0.8011 |
| ESOL | Scaffold | **0.9429** | 0.9475 | 0.9868 | 1.0384 |
| Lipophilicity | Group Random | 0.6907 | **0.5995** | 0.6215 | 0.7346 |
| Lipophilicity | Scaffold | 0.7599 | **0.5963** | 0.6425 | 0.7720 |
| FreeSolv | Group Random | 0.9231 | **0.7777** | 0.9655 | 1.3152 |
| FreeSolv | Scaffold | 2.5514 | **2.0301** | 2.3119 | 3.2035 |

表中是三个划分种子（3101、3102、3103）的平均测试 RMSE，越低越好。训练种子固定为 0。Group Random 将相同分子连接结构放在一起，Scaffold 将整个骨架组留出。各性质单位不同，只在同一行比较。

每份划分先在验证集选参数和轮次，再用锁定的权重做测试。18 份划分中，验证第一在测试时仍然第一的有 **10 份**，四个模型顺序完全相同的有 **4 份**。

[正式报告：RMSE、MAE、标准差和逐份排名](results/README.md) · [72 组成绩总表](results/all_scores.csv) · [候选、计时与验证记录](results/validation/README.md)

## ESOL 为什么换了排名？

在 3101 和 3103 的骨架划分上，验证集都是 MoLFormer 领先，测试集却都是 RF 第一。细看同一组 21 条巴比妥酸骨架记录，3101 上 RF 更准，3103 上 MoLFormer 更准。即使把两次训练都固定在第 2 轮，或都固定在第 10 轮，这组的排名仍然翻转。这个结论针对这 21 条记录，正式总表仍按验证集选中的轮次和完整测试集计分。

1. [哪些分子和骨架拉开了差距](results/esol-scaffold/README.md)
2. [同一组 21 条的逐分子预测和训练近邻](results/esol-neighbors/README.md)
3. [最新补跑：对齐第 2、10 轮后的四组结果](results/esol-epochs/README.md) · [21 条逐项对照表](results/esol-epochs/paired_21.csv)

## 协议与早期实验

[超参数、GPU pilot 和训练预算问答](validation/3101/QA.md) · [早期冻结模型、BACE 扩展与旧版结果导航](README-previous.md)
