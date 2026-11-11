# 三批验证结果 · 3101、3102、3103

[下载 36 份入选权重，查看 CPU 和 GPU 加载示例](../../weights/README.md)

三个划分种子共用同一份[正式训练脚本](../../formal_3101/train.py#L18)。MoLFormer 的编码器与回归头联合微调，Qwen 的 4-bit 量化和 LoRA 层在[模型初始化](../../revised_gpu_pilot/gpu.py#L30)中定义。[候选参数](../../formal_3101/protocol.py#L16)和[入选权重清单](selected_checkpoints.csv)可以按候选编号对照。运行原脚本需要本地实验归档的训练输入、基础模型与 GPU 环境。

三批共 144 个候选，正常完成 143 个。每份划分从正常完成者中独立选参，共锁定 36/36 个语言模型 checkpoint。RF 和 D-MPNN 取同一划分上已核验的验证成绩。

| 性质 | 划分 | RF | D-MPNN | MoLFormer | Qwen |
|---|---|---:|---:|---:|---:|
| FreeSolv | random_group | 0.9725 ± 0.1709 | 0.8422 ± 0.1967 | 0.9059 ± 0.0996 | 1.3336 ± 0.3321 |
| FreeSolv | random_scaffold | 3.0205 ± 0.6763 | 1.8513 ± 0.6385 | 2.5044 ± 0.5761 | 2.9260 ± 0.7676 |
| ESOL | random_group | 0.6393 ± 0.0594 | 0.6076 ± 0.0483 | 0.6145 ± 0.0801 | 0.7448 ± 0.0293 |
| ESOL | random_scaffold | 0.8530 ± 0.0891 | 0.8329 ± 0.0935 | 0.7932 ± 0.0886 | 0.8623 ± 0.1105 |
| Lipophilicity | random_group | 0.6667 ± 0.0498 | 0.5739 ± 0.0525 | 0.5977 ± 0.0233 | 0.6871 ± 0.0413 |
| Lipophilicity | random_scaffold | 0.7365 ± 0.0538 | 0.6174 ± 0.0325 | 0.6539 ± 0.0415 | 0.7633 ± 0.0485 |

表中为验证 RMSE 的均值 ± 样本标准差。每个分数都是在该划分的验证集上选好配置后读出的，三个划分的波动不能当作最终测试成绩。

## 换骨架后哪种模型领先

| 性质 | 划分种子 | Group Random 最低者 | Scaffold 最低者 |
|---|---:|---|---|
| FreeSolv | 3101 | MoLFormer（0.9709） | D-MPNN（2.4820） |
| FreeSolv | 3102 | D-MPNN（0.6778） | D-MPNN（1.8664） |
| FreeSolv | 3103 | D-MPNN（0.7886） | D-MPNN（1.2053） |
| ESOL | 3101 | MoLFormer（0.5417） | MoLFormer（0.6960） |
| ESOL | 3102 | MoLFormer（0.6014） | D-MPNN（0.8370） |
| ESOL | 3103 | D-MPNN（0.6053） | MoLFormer（0.8144） |
| Lipophilicity | 3101 | D-MPNN（0.5660） | D-MPNN（0.6503） |
| Lipophilicity | 3102 | MoLFormer（0.6167） | D-MPNN（0.6167） |
| Lipophilicity | 3103 | D-MPNN（0.5258） | D-MPNN（0.5853） |

按三份骨架验证集分别计数：

- FreeSolv：D-MPNN 领先 3 次。
- ESOL：MoLFormer 领先 2 次、D-MPNN 领先 1 次。
- Lipophilicity：D-MPNN 领先 3 次。

## 实际耗时与失败记录

- 3101 用时 13.87 小时，47/48 个候选正常完成。
- 3102 用时 13.23 小时，48/48 个候选正常完成。
- 3103 用时 15.05 小时，48/48 个候选正常完成。
- 3101 · Lipophilicity · random_scaffold · M2：failed_zero_gradient，已排除。

Lipophilicity Scaffold 的 MoLFormer 大学习率候选 M2 在三个划分上的状态依次为 3101（因零梯度退出）、3102（正常完成）、3103（正常完成）。
新增两批共占用约 28.28 小时，各批计时均从本批启动到核验结束。

训练种子统一为 0；最终测试见[三个任务的正式成绩](../README.md)。详见[144 个候选](candidates.csv)、[逐划分成绩](validation_scores.csv)、[用时](timing.csv)和[入选权重清单](selected_checkpoints.csv)。
