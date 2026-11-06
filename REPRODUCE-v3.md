# 怎样核对结果

只看图表可以直接打开仓库里的 CSV。想核对某个具体分子，可以下载记录包，在对应实验目录里打开 `seed0_test_predictions.csv`，按 `sample_id` 查找。每行同时有实测值 `y_true` 和预测值 `y_pred`。[实验文件下载](https://github.com/WeihanChen000/dsc6020-molecular-property/releases/tag/study-v3)

| 文件 | 什么时候用 |
|---|---|
| `study-v3-records.zip` | 包含训练代码、已缓存的分子特征、训练成员、各候选的验证记录、结果图表和逐分子预测。 |
| `molecular-study-v3-selected-checkpoints.zip` | 包含选中模型的权重和缩放器。重载核验和下方完整复跑命令会用到。 |
| `SHA256SUMS.txt` | 核对两个下载包的 SHA256。Windows 上可使用 `Get-FileHash 文件名`。 |

把两个压缩包解压到同一个空目录，合并其中的 `results/` 和 `inherited_v2/`。解压后，`verify.py`、`inputs/` 和这两个结果目录应当在同一层。下面的命令从这个目录运行。

## 核对保存的预测

原实验环境为 Windows、Python 3.12.10 和 CPU 版 PyTorch。用 PowerShell 建环境并安装锁定版本：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe audit_provenance.py
.\.venv\Scripts\python.exe verify.py --stage all
```

第一项检查逐条对齐原始分子和标签，并核对第二轮继承文件的哈希。第二项重载第三轮的选中模型，检查划分、预处理、预测和误差。本机原记录为 679 次拟合、187 个选中模型和 115,244 条验证或测试预测。[核验记录](study/verification_all.json)

## 从固定输入重新训练

```powershell
.\.venv\Scripts\python.exe run_new.py --workers 1
```

新结果写到 `runs/` 下的时间戳目录。原实验最多使用三个训练进程，五个阶段累计约 62 分钟，单进程耗时会有所不同。包中已经有冻结的 MoLFormer 向量，可以直接重跑下游训练。重载同一检查点可复算预测，从头训练的 GNN 轨迹会有浮点和训练过程差异，其具体来源仍待核查。

数据来自 MoleculeNet。模型使用 `ibm-research/MoLFormer-XL-both-10pct`，固定版本和源文件哈希记录在清单里。D-MPNN 使用 Chemprop 2.2.1 的消息传递、聚合和预测组件。[数据来源](https://pubs.rsc.org/en/content/articlehtml/2018/sc/c7sc02664a) · [MoLFormer 模型卡](https://huggingface.co/ibm-research/MoLFormer-XL-both-10pct) · [版本清单](study/model_source_manifest.json)
