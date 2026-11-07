# 怎样复核

核对整个 BACE 相似组划分时，用原始 `bace.csv`、作者补充压缩包和仓库里已有的测试预测即可。在仓库根目录、项目的 CPU Python 环境执行：

```powershell
python final/bace/qwen/assay_audit.py --raw bace.csv --supplement ci6b00290_si_001.zip --out checked-assays
```

脚本先核对源文件摘要、CID、发布标签和 1,503 条成员，再按种子和测试 ID 合并两份预测表，检查共有的 MoLFormer 预测是否一致。复现完整测试集成绩后，分别计算严格 IC50 子集和补入缺浓度值记录后的对照成绩。所有指标使用原实验的发布标签；换算浓度只用于检查标签来源。[原始 CSV](https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv) · [作者补充压缩包](https://doi.org/10.1021/acs.jcim.6b00290.s001) · [核对结果](final/bace/qwen/assay_audit)

要找八条难例在 1,053 条训练记录中的最近邻，准备下文的原始 CSV、补充 SDF 压缩包和 [study-v7](https://github.com/WeihanChen000/dsc6020-molecular-property/releases/tag/study-v7)、[study-v9](https://github.com/WeihanChen000/dsc6020-molecular-property/releases/tag/study-v9) 的冻结向量。在已有 CPU 环境运行：

```powershell
python final/bace/qwen/training_neighbors.py --raw bace.csv --supplement ci6b00290_si_001.zip --qwen qwen-bace/features/vectors.npz --molformer bace-runs/mlp/embeddings.npz --out checked-training-neighbors
```

Morgan 指纹沿用原划分的半径 2、2,048 位和无手性输入；两种编码器分别在自身未标准化的冻结向量上计算余弦相似度。各自只比较训练样本。前五名若在边界同分则全部计入，各记录等权；余弦值仅用于各自模型内部排序。活性匹配使八对标签绝对差的总和最小，平局按有序 ID 和当前求解器返回的解保留。脚本校验原始文件和向量摘要，并核对 203 条低活性记录的最高指纹相似度是否与原报告一致。[输出表和结构图](final/bace/qwen/training_neighbors)

要核对哪些分子拉高了 Qwen 的 RMSE，在仓库根目录用已有的 CPU Python 环境执行 `python final/bace/qwen/error_review.py --out checked-qwen-errors`。脚本读取已发布的两模型预测、训练均值和原始成绩，按分子、骨架及实测活性分组。所有差值均按 Qwen 减 MoLFormer 计算。共同难例取种子 1、2 的配对平方误差差值各自前十名的交集；骨架表用成员中编号最小者作索引，按整组对逐分子误差的总贡献排名。[核对用的四张表](final/bace/qwen/error_review)

## 在同一批 BACE 分子上比较冻结 Qwen 和 MoLFormer

只想看两者对同一个分子的预测，打开[逐条对照表](final/bace/qwen/results/predictions.csv)，按 `sample_id` 和 `seed` 查找。要从回归头权重重算，用原实验的 CPU Python 环境即可。下载 [BACE 原始 CSV](https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv)、[第三轮数据包](https://github.com/WeihanChen000/dsc6020-molecular-property/releases/tag/study-v3)和 [Qwen 实验包](https://github.com/WeihanChen000/dsc6020-molecular-property/releases/tag/study-v9)。从仓库根目录执行，假定第三轮包已解压到 `records`，Qwen 包已解压到 `qwen-bace`。

```powershell
$cpu = '.\.venv\Scripts\python.exe'
$scripts = 'final/bace/qwen'
$shared = @('--study','records','--raw','bace.csv','--features','qwen-bace/features')
& $cpu "$scripts/heads.py" evaluate @shared --runs qwen-bace/heads --out checked-qwen
```

若要重新训练四个候选回归头，先执行 `fit`，再执行 `evaluate`，两步均把 `--runs` 换成一个新的空目录。评分脚本按样本 ID 核对标签和旧预测，也检查三个种子的 MoLFormer 分数能否重现。

若要从原始结构重新生成 Qwen 向量，需要单独的 CUDA 环境。此次用 Python 3.12.10、PyTorch 2.11.0+cu128、Transformers 4.57.6、Accelerate 1.10.1、tokenizers 0.22.1、safetensors 0.6.2、NumPy 2.5.3 和 huggingface-hub 0.36.0。将 [Qwen 官方固定版本](https://huggingface.co/Qwen/Qwen3-1.7B/tree/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e)中[清单](final/bace/qwen/model_manifest.json)列出的文件放到 `Qwen3-1.7B`。下面的 `$gpu` 指向这个环境。编码器读取结构文件，不读取活性标签；它会逐个校验权重和 tokenizer 文件。

```powershell
$gpu = '.\.venv-gpu\Scripts\python.exe'
& $cpu "$scripts/prepare.py" --raw bace.csv --study records --out structures.csv
& $gpu "$scripts/encode.py" --model Qwen3-1.7B --structures structures.csv --out fresh-features
```

用新向量重新训练时，`--features` 传 `fresh-features`，再选一个空的回归头目录。发布的权重只和当时保存的向量配套。显存读数取自 GPU 编码阶段，CPU 训练耗时另记。

发布包 SHA256 为 `e31bfe1fcdf8e1b3f38f0c2f45d92f153056cad620ae6fa854f1521ee0427a0d`。我们从解压后的包重载了三个入选回归头，重新生成的预测、成绩和训练表与仓库文件逐字节一致。若只想用标准库复算两种模型的 18 个评分，执行 `python final/bace/qwen/check_scores.py` 即可。

## 核对 BACE 的早停轮次和预测分布

从相似组留出实验的发布页下载 grouped-runs.zip，放在仓库根目录。在项目 Python 环境中运行下面的命令，无需解压或训练。脚本校验压缩包摘要，从六个入选模型的逐轮日志定位验证 RMSE 最低点，再核对已发布的验证和测试 RMSE。[相似组模型包](https://github.com/WeihanChen000/dsc6020-molecular-property/releases/tag/study-v8)

```powershell
.\.venv\Scripts\python.exe final/bace/similarity/stopping_review.py --archive grouped-runs.zip --out checked-stopping
```

early_stops.csv 保存最佳轮、停止轮和对应损失；prediction_spread.csv 分别统计验证集和测试集，标准差按样本标准差计算。constant_pred_mean_rmse 用同一批预测的平均值替代每条预测，只用于解释误差。图中圆点标出验证集选中的轮次。

## 复算 BACE 全测试集的偏差方向，追溯两条相同标签

第一条命令使用仓库内已有的 300 条测试记录及三个种子的预测。第二条使用原 DeepChem bace.csv 和 2016 年论文的 ci6b00290_si_001.zip，从作者的 SDF 按 CID 对照 IC50、单位、关系和来源论文。两个文件可放在仓库外并传入对应路径。[DeepChem 文件](https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv) · [作者补充包](https://doi.org/10.1021/acs.jcim.6b00290.s001)

```powershell
.\.venv\Scripts\python.exe final/bace/similarity/mean_pull.py --out checked/mean_pull.csv
python final/bace/similarity/trace_pair.py --raw bace.csv --supplement ci6b00290_si_001.zip --out checked/pair_source.csv
```

toward 统计误差方向朝训练均值，between 要求预测落在实测值和均值之间，closer 允许跨过均值但要求绝对距离缩短。mean_error 为预测减去实测的平均值。scope 为 all 时保留全部记录，其他两种取值分别排除指定两条或种子 0 的 13 种翻转骨架。

## 对照 13 种骨架在三个种子下的表现

用项目的 Python 环境运行下面的命令。输入是仓库里已发布的测试预测、最近邻相似度和主评分表。RDKit 用于读取骨架重原子数和环数，无环记录的 Murcko 骨架按空结构计数。

```powershell
.\.venv\Scripts\python.exe final/bace/similarity/flip_profile.py --out checked-flip13
```

flip13.csv 固定种子 0 的 13 种骨架，三列分别对应三个种子；正值表示删组后 MoLFormer 的相对优势缩小。flip_cohorts.csv 给出 13 种与其余 170 种的描述统计。scaffold_influence.csv 保留全部 549 行，模型偏差是预测值减实测值，rank_for_lm 按有符号影响降序，rank_absolute 按影响绝对值降序。[13 组对照](final/bace/similarity/weights_review/flip13.csv)

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
