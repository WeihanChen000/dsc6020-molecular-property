# 怎样复核

## 用公开预测复算 BACE 种子 0 的排名变化

在仓库根目录运行以下命令，只需 Python 标准库。脚本读取相似分组实验已经公开的预测，按骨架汇总平方误差，与主表核对两个模型在三个种子下的两种 RMSE，然后逐个去掉骨架重算差值。

```powershell
python final/bace/similarity/scaffold_weights.py --out checked-scaffold-weights
```

seed0_scaffolds.csv 每行对应一种骨架，两个模型名下的数值均为该骨架内的 MSE。record_contribution 与 scaffold_contribution 分别是它对两种总体 MSE 差的贡献；weight_shift 是后者减前者。差值统一为 D-MPNN 减 MoLFormer。seed_sensitivity.csv 汇总每次仅去掉一种骨架后的胜负次数。[仓库中的输出](final/bace/similarity/weights_review)

## BACE 相似组整组留出

使用下文的同一份 bace.csv、第三轮的 records 目录、相同的锁定环境。把原 BACE 包解压到 bace-runs，这次只借用里面按 ID 存放的冻结向量。两种模型都用新分区重新训练。要复算发布的权重，把新包解压到 grouped-runs，其顶层应直接有 dmpnn 和 mlp 两个目录。[原 BACE 包](https://github.com/WeihanChen000/dsc6020-molecular-property/releases/tag/study-v7) · [相似分组模型包](https://github.com/WeihanChen000/dsc6020-molecular-property/releases/tag/study-v8)

```powershell
$py = '.\.venv\Scripts\python.exe'
$scripts = 'final/bace/similarity'
$cache = 'bace-runs/mlp/embeddings.npz'
& $py "$scripts/partition_groups.py" --raw bace.csv --study records --out rebuilt-partitions.csv
& $py "$scripts/compare.py" --raw bace.csv --study records --cache $cache --runs grouped-runs --out checked-grouped
```

第一行 Python 命令重建成员表。第二行重新检查分区成员、冻结向量与训练样本 ID、两条路线的验证选参记录和模型文件，再生成新旧划分的对照表。如果要从头训练，用一个新的输出目录执行下面三行。每个学习率都从头拟合，三个种子共用同一批验证与测试记录。

新包 grouped-runs.zip 的 SHA256 为 `f9056fb1bf0a05174b73e5c6cc29f7ea93a7565eef936f29ee804895ec54ec2a`。发布前从解压后的模型包、冻结向量和第三轮数据包重新运行了评分脚本。

```powershell
& $py "$scripts/fit.py" dmpnn --raw bace.csv --study records --cache $cache --runs grouped-retrain
& $py "$scripts/fit.py" mlp --raw bace.csv --study records --cache $cache --runs grouped-retrain
& $py "$scripts/compare.py" --raw bace.csv --study records --cache $cache --runs grouped-retrain --out rescored-grouped
```

## 复算 BACE 的独立验证

想看某条记录的预测，可以打开[测试预测表](final/bace/results/predictions.csv)，用 sample_id 对照[分区表](final/bace/partitions.csv)。例如 BACE_0000 对应原数据的第 0 行，从首条数据开始计数。三个种子共用测试 ID。

要从模型重新得到整张表，先按下文装好环境。把第三轮数据包解压到 records，把 bace-runs.zip 解压到 bace-runs，解压后应直接看到 dmpnn 和 mlp 两个目录。模型包保留了八次拟合的权重、训练曲线、验证预测，以及按样本 ID 保存的冻结向量。下载原始 CSV 后，在仓库根目录执行下列命令。[第三轮数据包](https://github.com/WeihanChen000/dsc6020-molecular-property/releases/tag/study-v3) · [BACE 模型包](https://github.com/WeihanChen000/dsc6020-molecular-property/releases/tag/study-v7)

```powershell
curl.exe -fL https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv -o bace.csv
.\.venv\Scripts\python.exe final/bace/evaluate.py --raw bace.csv --study records --runs bace-runs --out checked-bace --published-weights
```

预期打印三个种子的成绩，最后一项为 `reversed`。脚本核对原始文件、预设分区和训练配置，检查八次拟合的验证记录，然后重载六个入选模型。我们从解压包重建了结果，五份输出与仓库结果逐字节相同，又单独用公开预测复算了 RMSE、骨架等权 RMSE 和 MAE。模型包 SHA256 为 `12788de198d257cdf3b545732de7e131d27b2d3ce7e35cb0d29bbe14fd19957c`。

复核公开包时，`--published-weights` 会在加载权重前，把六个入选模型及其预处理文件与仓库记录的 SHA256 逐项对照。若要从头训练，先按仓库的 IBM 文件清单下载编码器。下面的命令用同一份原始 CSV 和第三轮数据目录，写入新的 retrained-bace 目录。训练代码只读取训练和验证记录的标签。

```powershell
.\.venv\Scripts\python.exe -c "import json; from huggingface_hub import snapshot_download; m=json.load(open('study/model_source_manifest.json')); snapshot_download(m['repository'],revision=m['revision'],allow_patterns=[f['filename'] for f in m['files']],local_dir='models/molformer')"
.\.venv\Scripts\python.exe final/bace/train.py dmpnn --raw bace.csv --study records --out retrained-bace
.\.venv\Scripts\python.exe final/bace/train.py mlp --raw bace.csv --study records --out retrained-bace --model models/molformer --manifest study/model_source_manifest.json
.\.venv\Scripts\python.exe final/bace/evaluate.py --raw bace.csv --study records --runs retrained-bace --out retrained-scores
```

## 复算 BACE 最近邻和逐骨架检查

这一项只需上面的原始 bace.csv、仓库里的分区与预测，以及已安装的 RDKit、NumPy 和 pandas。在仓库根目录运行以下命令。

```powershell
.\.venv\Scripts\python.exe final/bace/review.py --raw bace.csv --out checked-bace-review
```

输出三张表。neighbors.csv 列出 300 条测试记录各自最相似的训练记录，并列时保留全部 ID。scores.csv 按相似度分档，all 行可与主报告核对。omit_one_scaffold.csv 每行去掉一个测试骨架，差值为 D-MPNN 的 RMSE 减去 MoLFormer 的 RMSE。可将输出与[仓库里的三张表](final/bace/review)对照。

## 只用 Python 看清 0.7330 从哪里来

下载仓库或用 Git 克隆后，在仓库根目录运行下面的 PowerShell 命令。它只读一份公开 CSV，用 ESOL 骨架划分 3102 的 112 个测试分子复算 RMSE。列名 rdkit_mlp 是原 210 项描述符，without_ipc 是去掉 Ipc 后重新训练的预测。

```powershell
@"
import csv, math
with open('final/results/predictions.csv', encoding='utf-8') as f:
    rows = [r for r in csv.DictReader(f) if r['context'] == 'ESOL_random_scaffold_split3102']
print('n =', len(rows))
for m in ['train_mean', 'dmpnn', 'molformer_mlp', 'rdkit_mlp', 'without_ipc']:
    mse = sum((float(r[m]) - float(r['y_true']))**2 for r in rows) / len(rows)
    print(m, round(math.sqrt(mse), 4))
"@ | python -
```

预期输出依次为 112 条，以及 3.4469、0.8603、0.9259、0.7330、0.6854。若只想看哪些分子误差大，可直接筛选[逐分子预测表](final/results/predictions.csv)。

## 重算所有成绩，再重画主图

原实验使用 Windows、Python 3.12 和 CPU 版 PyTorch。先按锁定版本安装环境，然后把公开预测复制到空目录；此步只重算图表。

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r study/requirements-lock.txt
New-Item -ItemType Directory -Force check | Out-Null
Copy-Item final/results/predictions.csv check/
.\.venv\Scripts\python.exe final/summarize.py check
```

## 从保存的模型重建预测，或重新训练

下载旧版数据包 study-v3-records.zip，以及本版的 descriptor-runs.zip。先把两个包分别解压到 records 和 weights。前者顶层应能看到 core.py 和 inputs，后者顶层应有 all 和 no_ipc。数据包含训练成员、特征缓存和 D-MPNN、MoLFormer 的既有预测。本版模型包含第四轮和第五轮的全部 48 个候选文件夹、缩放器、曲线和验证选参记录。[数据包](https://github.com/WeihanChen000/dsc6020-molecular-property/releases/tag/study-v3) · [两组描述符模型](https://github.com/WeihanChen000/dsc6020-molecular-property/releases/tag/study-v5)

```powershell
.\.venv\Scripts\python.exe final/analyze.py --study records --runs weights --full weights/all --out checked
```

这一步从训练分区重算均值，重载 12 个入选的去 Ipc 模型，核对验证与测试预测，并对齐另外三条路线的已存预测。完整输入特征以样本 ID 对齐。模型包的 SHA256 为 `109ebed4c84716259de0fdf95bf7d914b090df2efab1b930f814435346a9e2e9`。

要从头训练两种描述符模型，可以另外指定一个空输出目录。代码按原训练索引和验证 RMSE 选参，CPU 浮点运算可能带来细微的重训差异。

```powershell
foreach ($variant in 'all','no_ipc') {
    foreach ($dataset in 'ESOL','Lipophilicity') {
        .\.venv\Scripts\python.exe final/train.py $dataset --study records --variant $variant --output retrained
    }
}
.\.venv\Scripts\python.exe final/analyze.py --study records --runs retrained --full retrained/all --out rechecked
```

D-MPNN、冻结 MoLFormer 和早期小样本实验的运行方法见[第三轮复现说明](REPRODUCE-v3.md)。

## 固定骨架划分，复核训练种子 0、1、2

仓库里的逐分子预测已经按测试 ID 对齐。按前文安装环境后，在仓库根目录运行：

```powershell
.\.venv\Scripts\python.exe final/seeds/summarize.py final/seeds/results/predictions.csv check-seeds
```

输出的 scores.csv 每行对应同一划分和同一训练种子的成对结果；scaffold_rmse 先在每个骨架内求均方误差，再对骨架取平均并开平方。split_summary.csv 的标准差只表示同一划分内三次训练的波动。seed_comparison.png 可和首页图对照。

若要核验检查点，把[种子复核模型包](https://github.com/WeihanChen000/dsc6020-molecular-property/releases/tag/study-v6)解压到 seed-weights，把[第三轮数据包](https://github.com/WeihanChen000/dsc6020-molecular-property/releases/tag/study-v3)解压到 records。前者顶层为六个划分目录，后者含 core.py、inputs 和 results。下列命令读取种子 0 的原始结果，并重新载入种子 1、2 的全部 24 个检查点，核对验证与测试预测。

```powershell
.\.venv\Scripts\python.exe final/seeds/collect.py --study records --runs seed-weights --reference final/results/predictions.csv --out checked-seeds
```

如需从头训练，将输出写到一个空目录。每个模型沿用本划分种子 0 已选定的学习率，早停只看本次验证集。训练完后将上一条命令的 --runs 改为 seed-rerun 即可复算。

```powershell
foreach ($split in 3101,3102,3103) {
    .\.venv\Scripts\python.exe final/seeds/train.py --study records --out seed-rerun --split $split
}
```
