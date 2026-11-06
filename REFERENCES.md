# 参考文献与实验设计

下面列出选择数据集、模型和划分方法时用到的研究。

### [1] MoleculeNet：数据集和任务来源

Wu et al. *MoleculeNet: a benchmark for molecular machine learning*. Chemical Science, 2018, 9:513–530；在线首发 2017。DOI：[10.1039/C7SC02664A](https://pubs.rsc.org/en/content/articlehtml/2018/sc/c7sc02664a)。核对范围：出版社原文摘要及任务说明。

论文事实：建立统一的数据、表征、算法和指标评测框架，并指出数据稀缺等条件会限制学习型表征。

对本项目的意义：ESOL、Lipophilicity 是合理的分子性质预测任务；采用监督回归头有明确研究基础。这个出处不意味着本地划分就是官方统一 folds，也不意味着两个任务自动满足老师的自拟题最低要求。

### [2] D-MPNN 的原始研究：强基线与任务依赖

Yang et al. *Analyzing Learned Molecular Representations for Property Prediction*. JCIM, 2019, 59:3370–3388。DOI：[10.1021/acs.jcim.9b00237](https://doi.org/10.1021/acs.jcim.9b00237)；核对[作者公开稿](https://arxiv.org/abs/1904.01561)的引言、方法与小数据讨论。

论文事实：研究固定描述符与学习表征，采用有向键消息传递；讨论小数据下指纹模型的竞争力，以及调参、数据划分和混合表征的重要性。

对本项目的意义：采用 D-MPNN 有依据；但调用成熟组件不等于复现论文完整训练方案。小样本下传统方法强，并不自动说明 GNN 实现错误。

### [3] MoLFormer 原论文：冻结与微调必须分开

Ross et al. *Large-scale chemical language representations capture molecular structure and properties*. Nature Machine Intelligence, 2022, 4:1256–1264。DOI：[10.1038/s42256-022-00580-7](https://doi.org/10.1038/s42256-022-00580-7)。核对[作者稿 v3](https://arxiv.org/abs/2106.09553v3)：Derivation of MOLFORMER Embeddings、Table 2、Fine-tuned versus Frozen，以及补充材料训练设置。

论文事实：明确区分固定编码器训练下游网络与联合微调；主要回归表中的 MoLFormer 使用微调。官方 [both-10pct 模型卡](https://huggingface.co/ibm-research/MoLFormer-XL-both-10pct)说明本项目使用的版本在两个预训练数据集各 10% 的数据上训练，允许冻结特征用于预测。

对本项目的意义：当前路线合理，但研究对象是“冻结 MoLFormer 的下游适配”，不能用它代表微调 MoLFormer 或通用聊天 LLM 的性能上限。论文表中的分数与本地分数存在训练方式、checkpoint 和划分差异，不直接排名。

### [4] 与本项目最直接相关的系统评测

Deng et al. *A systematic study of key elements underlying molecular property prediction*. Nature Communications, 2023, 14:6395。[原文及 DOI：10.1038/s41467-023-41948-6](https://www.nature.com/articles/s41467-023-41948-6)。核对范围：固定表征对照、数据划分与不确定性讨论。

论文事实：ESOL/FreeSolv 上 PhysChem 与 RDKit2D 表现较强；比较了 Morgan 指纹的位向量和计数形式；将划分、实验测量和训练过程视为不同不确定性来源。

对本项目的意义：当前 ESOL 结果具有先例，适合做复核与条件分析。增加训练种子不能解决只有一套划分的问题；ECFP 计数指纹是值得评测的低成本对照。不能据此预言它一定胜过现有 RF。

### [5] 最新冻结表征评测：传统指纹仍不能省

Praski, Adamczyk & Czech. *Benchmarking Pretrained Molecular Embedding Models for Molecular Representation Learning*. [arXiv:2508.06199v5](https://arxiv.org/html/2508.06199v5)，初稿 2025，核对版本更新于 2026-11-03；本次按预印本引用。

论文事实：比较 28 种表征、25 个分类数据集，用共享下游分类器评价冻结特征；ECFP count 是重要基线，MoLFormer 在其统计框架中未显示出相对 ECFP 的明确优势。

对本项目的意义：冻结特征研究本身有实际价值，也应比较强传统表征。但这篇是分类评测，不能直接证明我们的回归结果。其规模和统计模型也不能用本地三个训练种子替代。

### [6] Scaffold 不等于完全陌生的化学空间

Guo, Hernandez-Hernandez & Ballester. *Scaffold Splits Overestimate Virtual Screening Performance*. ICANN 2024, pp.58–72。[出版记录：10.1007/978-3-031-72359-9_5](https://link.springer.com/chapter/10.1007/978-3-031-72359-9_5)；核对[公开作者稿 v2](https://arxiv.org/abs/2406.00873v2)及出版摘要。

论文事实：在其虚拟筛选数据中，不同骨架之间仍可能有较高结构相似度；划分方法会影响难度判断。

对本项目的意义：保留 Scaffold，同时说明残余相似度和标签分布；不将其成绩称为“真实新药发现能力”。不能因为其他任务上的聚类划分更难，就宣称它对 ESOL 必然更真实。

### [7] OOD 评测应与应用目标相符

Fooladi et al. *Evaluating Machine Learning Models for Molecular Property Prediction: Performance and Robustness on Out-of-Distribution Data*. JCIM, 2025, 65:9871–9891。DOI：[10.1021/acs.jcim.5c00475](https://pmc.ncbi.nlm.nih.gov/articles/PMC12529777/)。核对范围：正式版本摘要、数据与划分方法；不混用更早预印本的模型数量。

论文事实：比较 14 个模型、8 个数据集与 10 类划分；ID 与 OOD 成绩的关系依赖生成 OOD 的方式。其任务为分类。

对本项目的意义：支持检验排名对划分的敏感性，并限制“结构泛化”这一表述的范围；不支持直接搬用其分类指标或效果量到当前回归任务。

### [8] 失败案例分析的参考：Activity cliffs

van Tilborg, Alenicheva & Grisoni. *Exposing the Limitations of Molecular Machine Learning with Activity Cliffs*. JCIM, 2022, 62:5938–5951。[原文：10.1021/acs.jcim.2c01073](https://pmc.ncbi.nlm.nih.gov/articles/9749029/)。核对范围：摘要、Study Setup 和失败模式相关片段。

论文事实：考察结构相近但生物活性差异较大的分子对，说明整体指标可能遮蔽特定困难样本。

对本项目的意义：可以借鉴“分组评价＋真实失败案例”。ESOL/Lipophilicity 不是这篇论文的靶点活性任务；单个高误差分子也不是一对分子。因此当前案例不能直接命名为 activity cliff。若做性质突变分析，需另行定义相似度、性质差阈值和观测单位。

### [9] 第三个数据集的 BACE-1 抑制活性来自哪里？

BACE 数据提供同一靶点的连续活性值和二分类标签。我们的第三项任务读取分子结构，用实验 pIC50 训练回归模型。原表另有作者的分区标记。为检验两条模型路线在留出骨架上的排名，我们事先另行固定了分区。报告中的数值对应本组协议。[原始研究（JCIM，2016）](https://doi.org/10.1021/acs.jcim.6b00290) · [DeepChem 的 BACE 回归数据说明](https://deepchem.readthedocs.io/en/latest/api_reference/moleculenet.html#deepchem.molnet.load_bace_regression)
