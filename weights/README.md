# 下载并加载 36 份正式权重

每个链接是一份独立 ZIP，解压后可以看到训练时保存的 `best.pt`、入选配置和上游模型版本。MoLFormer 每份约 158 MiB，包含微调后的编码器和回归头。Qwen 每份约 7–12 MiB，包含 4-bit QLoRA 训练得到的适配器和回归头。

| 数据集 | 测试划分 | MoLFormer · 划分种子 | Qwen · 划分种子 |
|---|---|---|---|
| ESOL | Group Random | [3101](https://github.com/WeihanChen000/dsc6020-molecular-property/releases/download/formal-weights-v1/MoLFormer-ESOL-random_group-3101.zip) · [3102](https://github.com/WeihanChen000/dsc6020-molecular-property/releases/download/formal-weights-v1/MoLFormer-ESOL-random_group-3102.zip) · [3103](https://github.com/WeihanChen000/dsc6020-molecular-property/releases/download/formal-weights-v1/MoLFormer-ESOL-random_group-3103.zip) | [3101](https://github.com/WeihanChen000/dsc6020-molecular-property/releases/download/formal-weights-v1/Qwen-ESOL-random_group-3101.zip) · [3102](https://github.com/WeihanChen000/dsc6020-molecular-property/releases/download/formal-weights-v1/Qwen-ESOL-random_group-3102.zip) · [3103](https://github.com/WeihanChen000/dsc6020-molecular-property/releases/download/formal-weights-v1/Qwen-ESOL-random_group-3103.zip) |
| ESOL | Scaffold | [3101](https://github.com/WeihanChen000/dsc6020-molecular-property/releases/download/formal-weights-v1/MoLFormer-ESOL-random_scaffold-3101.zip) · [3102](https://github.com/WeihanChen000/dsc6020-molecular-property/releases/download/formal-weights-v1/MoLFormer-ESOL-random_scaffold-3102.zip) · [3103](https://github.com/WeihanChen000/dsc6020-molecular-property/releases/download/formal-weights-v1/MoLFormer-ESOL-random_scaffold-3103.zip) | [3101](https://github.com/WeihanChen000/dsc6020-molecular-property/releases/download/formal-weights-v1/Qwen-ESOL-random_scaffold-3101.zip) · [3102](https://github.com/WeihanChen000/dsc6020-molecular-property/releases/download/formal-weights-v1/Qwen-ESOL-random_scaffold-3102.zip) · [3103](https://github.com/WeihanChen000/dsc6020-molecular-property/releases/download/formal-weights-v1/Qwen-ESOL-random_scaffold-3103.zip) |
| Lipophilicity | Group Random | [3101](https://github.com/WeihanChen000/dsc6020-molecular-property/releases/download/formal-weights-v1/MoLFormer-Lipophilicity-random_group-3101.zip) · [3102](https://github.com/WeihanChen000/dsc6020-molecular-property/releases/download/formal-weights-v1/MoLFormer-Lipophilicity-random_group-3102.zip) · [3103](https://github.com/WeihanChen000/dsc6020-molecular-property/releases/download/formal-weights-v1/MoLFormer-Lipophilicity-random_group-3103.zip) | [3101](https://github.com/WeihanChen000/dsc6020-molecular-property/releases/download/formal-weights-v1/Qwen-Lipophilicity-random_group-3101.zip) · [3102](https://github.com/WeihanChen000/dsc6020-molecular-property/releases/download/formal-weights-v1/Qwen-Lipophilicity-random_group-3102.zip) · [3103](https://github.com/WeihanChen000/dsc6020-molecular-property/releases/download/formal-weights-v1/Qwen-Lipophilicity-random_group-3103.zip) |
| Lipophilicity | Scaffold | [3101](https://github.com/WeihanChen000/dsc6020-molecular-property/releases/download/formal-weights-v1/MoLFormer-Lipophilicity-random_scaffold-3101.zip) · [3102](https://github.com/WeihanChen000/dsc6020-molecular-property/releases/download/formal-weights-v1/MoLFormer-Lipophilicity-random_scaffold-3102.zip) · [3103](https://github.com/WeihanChen000/dsc6020-molecular-property/releases/download/formal-weights-v1/MoLFormer-Lipophilicity-random_scaffold-3103.zip) | [3101](https://github.com/WeihanChen000/dsc6020-molecular-property/releases/download/formal-weights-v1/Qwen-Lipophilicity-random_scaffold-3101.zip) · [3102](https://github.com/WeihanChen000/dsc6020-molecular-property/releases/download/formal-weights-v1/Qwen-Lipophilicity-random_scaffold-3102.zip) · [3103](https://github.com/WeihanChen000/dsc6020-molecular-property/releases/download/formal-weights-v1/Qwen-Lipophilicity-random_scaffold-3103.zip) |
| FreeSolv | Group Random | [3101](https://github.com/WeihanChen000/dsc6020-molecular-property/releases/download/formal-weights-v1/MoLFormer-FreeSolv-random_group-3101.zip) · [3102](https://github.com/WeihanChen000/dsc6020-molecular-property/releases/download/formal-weights-v1/MoLFormer-FreeSolv-random_group-3102.zip) · [3103](https://github.com/WeihanChen000/dsc6020-molecular-property/releases/download/formal-weights-v1/MoLFormer-FreeSolv-random_group-3103.zip) | [3101](https://github.com/WeihanChen000/dsc6020-molecular-property/releases/download/formal-weights-v1/Qwen-FreeSolv-random_group-3101.zip) · [3102](https://github.com/WeihanChen000/dsc6020-molecular-property/releases/download/formal-weights-v1/Qwen-FreeSolv-random_group-3102.zip) · [3103](https://github.com/WeihanChen000/dsc6020-molecular-property/releases/download/formal-weights-v1/Qwen-FreeSolv-random_group-3103.zip) |
| FreeSolv | Scaffold | [3101](https://github.com/WeihanChen000/dsc6020-molecular-property/releases/download/formal-weights-v1/MoLFormer-FreeSolv-random_scaffold-3101.zip) · [3102](https://github.com/WeihanChen000/dsc6020-molecular-property/releases/download/formal-weights-v1/MoLFormer-FreeSolv-random_scaffold-3102.zip) · [3103](https://github.com/WeihanChen000/dsc6020-molecular-property/releases/download/formal-weights-v1/MoLFormer-FreeSolv-random_scaffold-3103.zip) | [3101](https://github.com/WeihanChen000/dsc6020-molecular-property/releases/download/formal-weights-v1/Qwen-FreeSolv-random_scaffold-3101.zip) · [3102](https://github.com/WeihanChen000/dsc6020-molecular-property/releases/download/formal-weights-v1/Qwen-FreeSolv-random_scaffold-3102.zip) · [3103](https://github.com/WeihanChen000/dsc6020-molecular-property/releases/download/formal-weights-v1/Qwen-FreeSolv-random_scaffold-3103.zip) |

## 普通电脑先跑一条

1. 下载并解压[加载工具包](https://github.com/WeihanChen000/dsc6020-molecular-property/releases/download/formal-weights-v1/load-tools.zip)。已有最新版仓库的队友可以直接在仓库根目录操作。
2. 下载 [MoLFormer · ESOL · Scaffold · 3101](https://github.com/WeihanChen000/dsc6020-molecular-property/releases/download/formal-weights-v1/MoLFormer-ESOL-random_scaffold-3101.zip)，将权重 ZIP 原样放在工具包解压后的根目录，与 `weights` 文件夹并列。
3. 在这个目录打开终端。建议使用 Python 3.12 的独立环境，首次安装依赖后，MoLFormer 可离线运行。

```sh
python -m pip install torch==2.11.0 transformers==4.57.6
python weights/predict.py MoLFormer-ESOL-random_scaffold-3101.zip --device cpu --smiles CCO
```

本机 CPU 输出为 `ESOL  CCO  -2.159689`，单位是 logS。输入沿用正式实验的分子连接结构 SMILES（规范化、去立体标记）；可在 `--smiles` 后连续填写多条。预测会用该权重保存的训练集均值和尺度还原到原单位。加载器从 ZIP 读权重，并与公开清单中的 SHA-256 核对。

## 在 NVIDIA GPU 上跑 Qwen

复测环境为 RTX 5060 Laptop 8 GB、支持 CUDA 的 PyTorch 2.11.0+cu128 和 BF16。先在支持 BF16 的 NVIDIA 机器上装好 CUDA 版 PyTorch；下面第一行可以检查当前环境。把要测的 Qwen 权重 ZIP 放在同一位置。

```sh
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.is_bf16_supported())"
python -m pip install transformers==4.57.6 peft==0.17.1 accelerate==1.10.1 bitsandbytes==0.50.2 huggingface-hub==0.36.0
python weights/predict.py Qwen-ESOL-random_scaffold-3101.zip --device cuda --smiles CCO
```

首次运行会从 Hugging Face 下载锁定版本的 Qwen3-1.7B，约 4.08 GB，之后 18 个 Qwen 包共享同一缓存。已经有该版本的本地文件时，可追加 `--base-dir <基础模型目录>`。加载器会按原设置做 NF4 量化、恢复 LoRA 和回归头，并使用各自入选的 pooling。[基础模型版本与文件清单](../revised_gpu_pilot/qwen/source.json)

## 想核对某一份结果

- 分子性质、划分方式和种子共同决定用哪份权重。比如做 ESOL 的 Scaffold / 3101 对照时，两种模型都点该行的 3101。训练种子统一为 0。
- 逐份入选的候选编号、最佳轮次、原归档相对路径和 `best.pt` 的 SHA-256 在[原始入选表](../results/validation/selected_checkpoints.csv)。[下载包清单](packages.csv)另列 ZIP 大小和 ZIP 自身的 SHA-256。
- 验证方法是每个 ZIP 独立加载，从对应正式测试集取第一个完整批次，并与原预测按分子 ID 对照。[逐包复测结果](replay.csv)
- [训练代码](https://github.com/WeihanChen000/dsc6020-molecular-property/blob/e6a63e2/formal_3101/train.py#L18)和[正式测试误差](https://github.com/WeihanChen000/dsc6020-molecular-property/blob/main/results/all_scores.csv)对应同一批入选模型。

工具包附带 IBM MoLFormer 固定版本的配置、分词器和模型定义，保留上游说明和 Apache 2.0 许可。预测时直接恢复本实验微调后的完整权重。[IBM 模型版本](https://huggingface.co/ibm-research/MoLFormer-XL-both-10pct/tree/7b12d946c181a37f6012b9dc3b002275de070314) · [Qwen 版本](https://huggingface.co/Qwen/Qwen3-1.7B/tree/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e)
