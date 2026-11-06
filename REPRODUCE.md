# 怎样复核

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
