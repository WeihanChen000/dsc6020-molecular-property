# 三个分子性质任务的最终测试

18 份划分里，验证集上 RMSE 最低的模型有 10 次在测试集上仍然第一，四个模型的完整顺序有 4 次相同。固定训练种子 0，每次更换划分种子都会重新分配训练、验证和测试分子。

D-MPNN 在六种「任务 × 划分方式」中的五种取得最低的平均测试 RMSE 和 MAE。最稳的是 Lipophilicity 的骨架划分，三个种子上 D-MPNN 都在验证集和测试集排第一，MoLFormer 都排第二。

ESOL 的骨架划分变化最大。MoLFormer 在两份验证集领先，到了测试集，这两份都是 RF 第一。RF 的平均测试 RMSE 为 0.9429，D-MPNN 为 0.9475，两者只差 0.0046 logS。FreeSolv 的骨架划分也有例外，3102 的测试集由 MoLFormer 领先，另外两份由 D-MPNN 领先。

这次用了联合微调的 MoLFormer 和 4-bit QLoRA 的 Qwen3-1.7B。Qwen 在六种设置里的平均测试 RMSE 都排第四；这个结果适用于本次三个性质任务、输入表示和训练预算。

[结构泛化主图、Tanimoto 分档 MAE 与样本量](structure/README.md)

四条路线的搜索、入选训练和推理耗时已按同一份测试结果对齐。可以逐项查看加载、预处理、推理和显存，也能查到各次候选训练花了多久。[四模型成本和测试误差](cost/README.md)

## 哪些排名保持住了

每行先对三个划分的误差取平均，再比较模型。“冠军一致”表示三份划分中，有几份的验证和测试 RMSE 由同一个模型领先。末列是平均测试 RMSE 前两名的差值，使用各任务自身的单位。

| 数据集 | 划分 | 验证 RMSE 第一 | 测试 RMSE 第一 | 测试 MAE 第一 | 冠军一致 | 前两名差值 |
| --- | --- | --- | --- | --- | --- | --- |
| ESOL | Group Random | D-MPNN | D-MPNN | D-MPNN | 1/3 | 0.0845 |
| ESOL | Scaffold | MoLFormer | RF | RF | 1/3 | 0.0046 |
| FreeSolv | Group Random | D-MPNN | D-MPNN | D-MPNN | 2/3 | 0.1454 |
| FreeSolv | Scaffold | D-MPNN | D-MPNN | D-MPNN | 2/3 | 0.2817 |
| Lipophilicity | Group Random | D-MPNN | D-MPNN | D-MPNN | 1/3 | 0.0221 |
| Lipophilicity | Scaffold | D-MPNN | D-MPNN | D-MPNN | 3/3 | 0.0462 |

## 四个模型的测试成绩

数值为三个划分种子（3101、3102、3103）的均值 ± 样本标准差。每份划分内按测试分子等权算误差，三个划分各占三分之一。ESOL 预测 logS，Lipophilicity 预测 logD，FreeSolv 的水合自由能单位为 kcal/mol。请在同一任务内比较。

| 数据集 | 划分 | 模型 | RMSE ↓ | MAE ↓ |
| --- | --- | --- | --- | --- |
| ESOL | Group Random | D-MPNN | 0.6022 ± 0.0335 | 0.4575 ± 0.0194 |
| ESOL | Group Random | MoLFormer | 0.6867 ± 0.1145 | 0.5038 ± 0.0961 |
| ESOL | Group Random | RF | 0.7076 ± 0.0648 | 0.5164 ± 0.0267 |
| ESOL | Group Random | Qwen | 0.8011 ± 0.0882 | 0.6084 ± 0.0796 |
| ESOL | Scaffold | RF | 0.9429 ± 0.0267 | 0.6853 ± 0.0428 |
| ESOL | Scaffold | D-MPNN | 0.9475 ± 0.0836 | 0.6896 ± 0.0638 |
| ESOL | Scaffold | MoLFormer | 0.9868 ± 0.0117 | 0.7622 ± 0.0303 |
| ESOL | Scaffold | Qwen | 1.0384 ± 0.0212 | 0.8208 ± 0.0395 |
| FreeSolv | Group Random | D-MPNN | 0.7777 ± 0.1321 | 0.5333 ± 0.0948 |
| FreeSolv | Group Random | RF | 0.9231 ± 0.1071 | 0.5907 ± 0.0844 |
| FreeSolv | Group Random | MoLFormer | 0.9655 ± 0.1324 | 0.6630 ± 0.1168 |
| FreeSolv | Group Random | Qwen | 1.3152 ± 0.0307 | 0.9018 ± 0.1083 |
| FreeSolv | Scaffold | D-MPNN | 2.0301 ± 0.3575 | 1.4328 ± 0.1960 |
| FreeSolv | Scaffold | MoLFormer | 2.3119 ± 0.6524 | 1.5546 ± 0.4042 |
| FreeSolv | Scaffold | RF | 2.5514 ± 0.5801 | 1.6394 ± 0.2116 |
| FreeSolv | Scaffold | Qwen | 3.2035 ± 0.3316 | 1.9866 ± 0.1948 |
| Lipophilicity | Group Random | D-MPNN | 0.5995 ± 0.0657 | 0.4107 ± 0.0028 |
| Lipophilicity | Group Random | MoLFormer | 0.6215 ± 0.0139 | 0.4522 ± 0.0194 |
| Lipophilicity | Group Random | RF | 0.6907 ± 0.0172 | 0.5272 ± 0.0132 |
| Lipophilicity | Group Random | Qwen | 0.7346 ± 0.0225 | 0.5408 ± 0.0110 |
| Lipophilicity | Scaffold | D-MPNN | 0.5963 ± 0.0313 | 0.4452 ± 0.0076 |
| Lipophilicity | Scaffold | MoLFormer | 0.6425 ± 0.0403 | 0.4869 ± 0.0257 |
| Lipophilicity | Scaffold | RF | 0.7599 ± 0.0458 | 0.5825 ± 0.0321 |
| Lipophilicity | Scaffold | Qwen | 0.7720 ± 0.0343 | 0.5977 ± 0.0174 |

## 逐份划分看排名

从左到右 RMSE 逐渐增大。排名使用未四舍五入的数值。

| 数据集 | 划分 | 种子 | 验证集顺序 | 测试集顺序 |
| --- | --- | --- | --- | --- |
| ESOL | Group Random | 3101 | MoLFormer > D-MPNN > RF > Qwen | D-MPNN > MoLFormer > RF > Qwen |
| ESOL | Group Random | 3102 | MoLFormer > D-MPNN > RF > Qwen | D-MPNN > MoLFormer > RF > Qwen |
| ESOL | Group Random | 3103 | D-MPNN > RF > MoLFormer > Qwen | D-MPNN > RF > MoLFormer > Qwen |
| ESOL | Scaffold | 3101 | MoLFormer > Qwen > D-MPNN > RF | RF > D-MPNN > MoLFormer > Qwen |
| ESOL | Scaffold | 3102 | D-MPNN > MoLFormer > Qwen > RF | D-MPNN > RF > MoLFormer > Qwen |
| ESOL | Scaffold | 3103 | MoLFormer > RF > D-MPNN > Qwen | RF > MoLFormer > D-MPNN > Qwen |
| FreeSolv | Group Random | 3101 | MoLFormer > D-MPNN > RF > Qwen | D-MPNN > RF > MoLFormer > Qwen |
| FreeSolv | Group Random | 3102 | D-MPNN > MoLFormer > RF > Qwen | D-MPNN > MoLFormer > RF > Qwen |
| FreeSolv | Group Random | 3103 | D-MPNN > RF > MoLFormer > Qwen | D-MPNN > MoLFormer > RF > Qwen |
| FreeSolv | Scaffold | 3101 | D-MPNN > MoLFormer > RF > Qwen | D-MPNN > MoLFormer > RF > Qwen |
| FreeSolv | Scaffold | 3102 | D-MPNN > MoLFormer > Qwen > RF | MoLFormer > D-MPNN > RF > Qwen |
| FreeSolv | Scaffold | 3103 | D-MPNN > MoLFormer > RF > Qwen | D-MPNN > MoLFormer > Qwen > RF |
| Lipophilicity | Group Random | 3101 | D-MPNN > MoLFormer > Qwen > RF | D-MPNN > MoLFormer > RF > Qwen |
| Lipophilicity | Group Random | 3102 | MoLFormer > D-MPNN > RF > Qwen | D-MPNN > MoLFormer > RF > Qwen |
| Lipophilicity | Group Random | 3103 | D-MPNN > MoLFormer > Qwen > RF | MoLFormer > RF > D-MPNN > Qwen |
| Lipophilicity | Scaffold | 3101 | D-MPNN > MoLFormer > Qwen > RF | D-MPNN > MoLFormer > RF > Qwen |
| Lipophilicity | Scaffold | 3102 | D-MPNN > MoLFormer > RF > Qwen | D-MPNN > MoLFormer > Qwen > RF |
| Lipophilicity | Scaffold | 3103 | D-MPNN > MoLFormer > RF > Qwen | D-MPNN > MoLFormer > RF > Qwen |

## 怎么复核

36 个语言模型权重均通过 SHA-256 核对。每次重新加载基础模型、入选权重和回归头，用训练集保存的均值与尺度还原预测。完整验证集上，逐分子预测与原记录的最大绝对差为 1.01e-06（容许差 0.00001）。随后用同一个模型预测该划分的测试分子。

RF 与 D-MPNN 的 36 组测试预测沿用已核验的存档，早期研究中已查看过。此次按分子 ID 核对四个模型的测试名单与标签，重算原单位 RMSE、MAE。共有 14,304 条模型预测记录，含不同划分间重复出现的分子。

验证集同时用于选参数和轮次。验证和测试的差别也受两批分子的组成影响。三个划分只描述这次划分带来的波动，均值接近时不宜断言模型普遍占优。

[all_scores.csv](all_scores.csv) 是总表，每行对应一份划分上的一个模型。[summary.csv](summary.csv) 给出跨划分均值和标准差，[rank_changes.csv](rank_changes.csv) 列出每份划分的排名。[test_predictions.csv](test_predictions.csv) 和 [checkpoint_audit.csv](checkpoint_audit.csv) 供复核。
