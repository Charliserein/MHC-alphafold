# MHCalphafold

从 [PMGen](https://github.com/soedinglab/PMGen) 单独抽取的 **肽–MHC（pMHC）AlphaFold2 结构预测模块**。输入肽序列和 MHC 序列，完成模板检索、锚点对齐、结构预测、置信度计算与候选选优，输出复合物 PDB。

本目录包含独立的预测源码、模板库、权重、比对工具和本机 Python 环境，运行不依赖原来的 `pmgen` 文件夹。它不是新训练的 AlphaFold 模型；默认使用 **AF2 原版 `model_2_ptm` 参数**。不包含 ProteinMPNN 肽生成、BioEmu 和结合亲和力筛选流程。

英文介绍（恰好 300 characters，包含空格和标点，另见 [INTRODUCTION.txt](INTRODUCTION.txt)）：

> MHCalphafold is a standalone pipeline for peptide-MHC structure prediction built on AlphaFold2. It combines pMHC templates, anchor-guided initial coordinates, and confidence-based model selection. It provides reproducible setup and inference workflows while retaining the original AlphaFold2 weights.

## 1. 相对于 AlphaFold2 做了什么适配

核心是把通用 AF2 推理改成 **pMHC 模板和锚点引导的推理流程**。默认配置不重新训练权重，保留 Evoformer、Structure Module 和 AF2 原有置信度计算。

| 适配 | 原理与作用 | 主要代码 |
| --- | --- | --- |
| pMHC 专用模板检索 | 使用本地 PANDORA pMHC 数据库；按 MHC 结合域序列检索模板，再结合肽比对得分选择模板 | `PANDORA/PANDORA/Pandora/Modelling_functions.py` |
| 锚点优先对齐 | 先对应目标肽与模板肽的锚点，再对齐其余位置，使结合位置符合 pMHC 几何先验 | `PANDORA/PANDORA/Pandora/Pandora.py` |
| Initial Guess（IG） | 从选中的模板保留可用的 MHC 坐标和肽锚点坐标；非锚点肽坐标与原子有效性 mask 置零 | `AFfine/af2_util.py` |
| 修改初始 recycling 输入 | `RunModel` 接收 `initial_guess` / `initial_guess_mask`，填入首次 `prev_pos`；经伪 β 原子距离特征进入 Evoformer | `AFfine/alphafold/model/model.py`、`modules.py` |
| 对无效几何使用 mask | 初始坐标缺失的残基对不提供距离信息，避免把零坐标解释为真实几何 | `AFfine/alphafold/model/modules.py` |
| 多链输入 | 将 MHC 链和肽拼接成模型序列，在链边界增加 residue index 偏移 200 来表达分链；默认走 monomer pTM 网络，不是 AF-Multimer 权重 | `AFfine/predict_utils_ig.py` |
| 单序列 MSA | 默认不执行大规模同源 MSA 数据库搜索，仅提供查询序列；仍使用模板特征与初始坐标 | `AFfine/run_prediction.py` |
| 锚点候选搜索 | 可手动指定锚点，也可枚举合法组合并分别推理；按肽核心区域 pLDDT 选优 | `mhcalphafold/inputs.py`、`outputs.py` |
| 可选 Template Engineering（TE） | 使用 PANDORA + Modeller 生成同源模型，再把这些结构送入 AF2 模板通道；需另行配置 Modeller | `mhcalphafold/backend.py` |

**锚点是起始几何引导，不是最终结构的硬性锁定。** 后续 recycling 使用上一轮预测坐标，因此锚点与 MHC 可以随网络输出继续调整。IG 路线仍使用模板，只是绕过 Modeller 同源建模步骤。

这些科学方法来自 PMGen 及其引用的 AFfine / binder-design 工作。抽取时新增的输入校验、结果整理、断点恢复、多进程调度，以及部署时的 JAX/Haiku 兼容修复，属于运行工程，不能据此宣称结构准确率提高。pLDDT 是模型置信度，也不是与实验结构比较得到的准确率。

## 2. 目录与功能范围

```text
MHCalphafold/
├── README.md                     # 本说明
├── INTRODUCTION.txt              # 300 字符英文介绍
├── mhcalphafold.sh                # 推荐命令入口
├── predict.py                    # Python 命令入口
├── mhcalphafold/                  # 输入、预测编排、校验、导出
├── AFfine/                       # 完整 AF2 预测实现与高级 IG/采样接口
│   └── af_params/                 # 原版权重和已有微调权重
├── PANDORA/                      # 模板检索、对齐和可选模板工程
├── resources/
│   ├── pandora_db/               # 模板 PDB、序列库、BLAST 数据库
│   └── tools/                    # 本机 Linux x86_64 BLAST+/MUSCLE
├── examples/                     # MHC-I / MHC-II 的 TSV 与 FASTA
├── scripts/                      # 环境、检查、并行、资源恢复和验证脚本
├── tests/                        # 输入、分数与检查点协议测试
├── validation/                   # 安装、抽取一致性及样例验证记录
├── SOURCE_MANIFEST.json          # 上游提交、资源 SHA256、原部署兼容补丁
├── requirements*.txt             # 独立部署的固定依赖版本
├── .venv/                        # 本机独立 Python 环境
└── .runtime/python/              # 本机独立 Python 3.10 解释器
```

默认支持 MHC-I 和 MHC-II；输出包含 **所提供的 MHC 序列 + 肽**。例如，输入 MHC-I 结合域时，得到的是结合域–肽复合物，不会自动补齐跨膜区、β2-microglobulin、TCR 或其他未输入组分。

这里“独立预测模块”指 PMGen 中的 AF2 结构预测链路。它没有打包通用 AF2 的大型 MSA 数据库和全流程数据库搜索服务；默认输出未经 Amber relaxation 的 AF2 结构。`AFfine/alphafold/relax` 的上游代码虽保留，默认部署未配置 OpenMM/PDBFixer，也不会自动调用该步骤。

## 3. 怎么部署

### 3.1 当前这台机器：已经部署

```bash
cd /home/c/czh/MHCalphafold
./mhcalphafold.sh check
./mhcalphafold.sh predict --help
```

`check` 检查实际导入的 AF2/PANDORA 路径、权重、模板库、BLAST/MUSCLE 和 JAX GPU，并将记录写到 `validation/install-check.json`。

本机固定栈：Linux x86_64、Python 3.10.22、JAX 0.4.35 / jaxlib 0.4.34、CUDA 12 插件 0.4.34、Haiku 0.0.13、TensorFlow CPU 2.17.1、NumPy 1.26.4、Biopython 1.80。**预测不需要 PyTorch。** 本机环境是实际文件副本，没有指向原 PMGen 目录的环境或资源符号链接。

当前随附 `model_2_ptm` 原版权重和 PMGen 已有的 `model_ft_mhc_20640.pkl` 微调权重；默认仅使用前者。模板对象共 MHC-I 864 个、MHC-II 136 个，资源目录还包含反向模板等附属结构。

### 3.2 另一台 Linux GPU 机器

GitHub 仓库只上传源码、样例和文档，不包含本机 `.venv/`、`.runtime/`、`resources/`、模型权重或预测输出。通过 GitHub 部署时先克隆代码：

```bash
git clone https://github.com/Charliserein/MHC-alphafold.git
cd MHC-alphafold
```

随后准备 Python 3.10、NVIDIA/CUDA 环境，以及已加入 `PATH` 的 BLAST+ 和 **MUSCLE 5**，再下载模板库与权重并安装 Python 依赖：

```bash
blastp -version
muscle -version               # 本实现的对齐参数需要 MUSCLE 5
python3.10 scripts/fetch_resources.py
export CUDA_ROOT=/usr/local/cuda-12.8   # 按实际 CUDA 目录调整
bash scripts/install.sh
```

没有 `resources/tools/bin/` 时，启动脚本会使用系统 `PATH` 中的比对工具。下面的复制部署方式适用于从已经准备好资源的本地目录迁移。

1. 复制本项目的源码、`resources/` 和 `AFfine/af_params/`。`.venv/` 与 `.runtime/` 是本机运行环境；换目录或换机器时建议重新创建虚拟环境。
2. 准备 Python **3.10**、NVIDIA 驱动及可用的 CUDA 12 工具链。用 `nvidia-smi` 和 `ptxas --version` 检查；本机验证使用 CUDA 12.8。
3. 安装固定 Python 依赖并执行检查：

```bash
cd /你的路径/MHCalphafold
export MHC_AF_BOOTSTRAP_PYTHON=/你的/python3.10
export CUDA_ROOT=/usr/local/cuda-12.8   # 按本机实际 CUDA 目录修改
bash scripts/install.sh
```

安装脚本将创建 `.venv`，安装 `requirements.txt` 和 `requirements-cuda.txt`，运行 `pip check` 及 GPU 检查；不会安装 ProteinMPNN、BioEmu 或 PyTorch。系统资源不足时先使用单进程推理。

`scripts/activate.sh` 设置项目自己的 Python、模板库、工具路径、线程数和最多 1 GiB 的 JAX 编译缓存。可在运行前覆盖：

```bash
export CUDA_VISIBLE_DEVICES=1
export CUDA_ROOT=/usr/local/cuda-12.8
export MHC_AF_DATABASE=/你的/pandora_db   # 可选：使用其他兼容模板库
```

本机携带的 BLAST/MUSCLE 二进制用于 Linux x86_64；如果目标系统动态库不兼容，安装相应工具并替换 `resources/tools/bin/` 的入口，随后重新执行 `check`。

### 3.3 缺少权重或模板库时

完整目录已包含资源，无需重新下载。如果只复制了代码，可以从 PMGen 上游的固定归档恢复：

```bash
python3.10 scripts/fetch_resources.py

# 或使用已下载的本地归档
python3.10 scripts/fetch_resources.py \
  --database-zip /路径/pandora_db.zip \
  --weights-zip /路径/af_params.zip
```

脚本核对已知归档 SHA256 和 ZIP CRC 后解压。下载入口及哈希写在脚本中；若上游归档发生变化，校验会失败，需要先确认新资源版本。该脚本恢复模板库与权重；比对工具由随目录的 `resources/tools/` 提供。

### 3.4 哪些许可证是可选的

| 使用方式 | Modeller | NetMHCpan / NetMHCIIpan |
| --- | --- | --- |
| IG + 手动锚点 | 不需要 | 不需要 |
| IG + 枚举锚点（默认无锚点时使用） | 不需要 | 不需要 |
| IG + NetMHC 预测锚点 | 不需要 | 需要对应授权安装包 |
| TE 模板工程 | 需要已安装且有效授权的 Modeller | 使用手动/枚举锚点时不需要 |

Modeller 从[官方注册入口](https://salilab.org/modeller/registration.html)申请密钥后，按其官方步骤安装到本项目 Python 环境。只有 `--mode te` 才会调用它。

NetMHC 工具安装后设置**可执行文件路径**：

```bash
export NETMHCPAN_PATH=/你的/netMHCpan-4.1/netMHCpan
export NETMHCIIPAN_PATH=/你的/netMHCIIpan-4.3/netMHCIIpan
```

DTU 官方入口：[NetMHCpan](https://services.healthtech.dtu.dk/services/NetMHCpan-4.1/)、[NetMHCIIpan](https://services.healthtech.dtu.dk/services/NetMHCIIpan-4.3/)。授权工具及密钥不包含在本目录中。

## 4. 怎么推理

### 4.1 最快跑通一个 MHC-I 样例

```bash
cd /home/c/czh/MHCalphafold
CUDA_VISIBLE_DEVICES=1 ./mhcalphafold.sh predict \
  --input examples/mhc1.tsv \
  --output outputs/mhc1_demo
```

这使用样例给定的锚点、4 个模板、`model_2_ptm` 和 3 次 recycling。首次出现某个序列长度时需要 JAX 编译，耗时会增加。GPU 编号通过 `CUDA_VISIBLE_DEVICES` 选择，不会更改原项目的后台任务。

### 4.2 直接输入一条肽与 MHC FASTA

```bash
CUDA_VISIBLE_DEVICES=1 ./mhcalphafold.sh predict \
  --id my_sample \
  --peptide TQGTPLKY \
  --mhc-type 1 \
  --mhc-fasta examples/mhc1.fasta \
  --anchors '2;8' \
  --output outputs/my_sample
```

也可以用 `--mhc-seq '完整的MHC结合域序列'` 替代 `--mhc-fasta`。这里应提供实际氨基酸序列，**不能把 HLA 名称或 34 aa 伪序列直接当作 MHC 输入**。伪序列必须先映射到真实 MHC 序列；有歧义时需要记录映射选择。

### 4.3 批量 TSV / CSV 输入

至少提供以下列；`anchors` 可以留空：

| 列 | 含义 |
| --- | --- |
| `id` | 每条样本唯一标识，以字母/数字开头，只含字母、数字、`_`、`.`、`-` |
| `peptide` | 标准氨基酸肽序列 |
| `mhc_seq` | MHC-I：一条 MHC 序列；MHC-II：`Alpha序列/Beta序列` |
| `mhc_type` | `1` 或 `2` |
| `anchors` | 肽内 **1-based** 位置；MHC-I 两个，MHC-II 四个；使用分号分隔 |

可直接修改 `examples/mhc1.tsv` 或 `examples/mhc2.tsv`；运行：

```bash
CUDA_VISIBLE_DEVICES=1 ./mhcalphafold.sh predict \
  --input /路径/input.tsv \
  --output outputs/my_batch \
  --batch-size 50
```

`--batch-size` 是每个 AF2 子进程处理的原始样本数，每条样本可展开成多个锚点候选；它不是同时在 GPU 上执行的样本数。每批内顺序推理并复用模型和编译结果。

### 4.4 没有 NetMHC 授权：枚举锚点

```bash
CUDA_VISIBLE_DEVICES=1 ./mhcalphafold.sh predict \
  --input examples/mhc1_enumerate.tsv \
  --anchor-mode enumerate \
  --output outputs/enumerated
```

- `auto`（默认）：已有锚点直接使用，否则枚举。
- `given`：必须给定锚点。
- `enumerate`：忽略输入锚点，枚举全部合法组合。
- `netmhc`：已有锚点直接使用，否则调用对应 NetMHC 授权工具。

MHC-I 枚举两个锚点，要求位置差至少为 6；8 肽对应 `(1,7)`、`(1,8)`、`(2,8)` 三个候选。MHC-II 枚举 `(i,i+3,i+5,i+8)` 的四锚点组合。两者分别预测后按核心 pLDDT 选优，候选越多，总耗时越长。

### 4.5 MHC-II

```bash
CUDA_VISIBLE_DEVICES=1 ./mhcalphafold.sh predict \
  --input examples/mhc2.tsv \
  --output outputs/mhc2_demo
```

直接使用 FASTA 时，先写 Alpha，再写 Beta，提供 `--mhc-type 2` 和四个锚点。输出还包含肽链。样例来自模板库，用于验证功能；不能当作无模板泄漏的准确率评测。

### 4.6 断点续跑

在原命令中增加 `--resume`：

```bash
CUDA_VISIBLE_DEVICES=1 ./mhcalphafold.sh predict \
  --input examples/mhc1.tsv \
  --output outputs/mhc1_demo \
  --resume
```

输入、模型和推理配置必须保持一致。检查点会核对任务指纹及已保存 PDB/分数文件的 SHA256，完整样本直接跳过；未完整完成的批次会重新准备并推理。结果按原始输入顺序写入 `results.tsv`。相同输出目录禁止两个进程同时写入。

### 4.7 断开终端或 AI 后继续

```bash
mkdir -p outputs
nohup env CUDA_VISIBLE_DEVICES=1 ./mhcalphafold.sh predict \
  --input /路径/input.tsv \
  --output outputs/background_job \
  > outputs/background_job.log 2>&1 < /dev/null &
echo $!
```

查看进度：

```bash
cat outputs/background_job/progress.json
tail -f outputs/background_job.log
```

`nohup` 适合关闭终端后继续；若希望由用户 systemd 管理并在失败时重试，可按实际绝对路径执行：

```bash
systemd-run --user --unit=mhcalphafold-myjob \
  --property=Restart=on-failure --property=RestartSec=30 \
  --setenv=CUDA_VISIBLE_DEVICES=1 \
  /home/c/czh/MHCalphafold/mhcalphafold.sh predict \
  --input /绝对路径/input.tsv \
  --output /home/c/czh/MHCalphafold/outputs/systemd_job --resume

systemctl --user status mhcalphafold-myjob
journalctl --user -u mhcalphafold-myjob -f
```

用户退出登录后继续运行还取决于系统是否允许用户服务驻留；`loginctl show-user "$USER" -p Linger` 可检查。本机原 PMGen 服务与这里的示例是不同任务。本说明没有自动创建或启动额外批量服务。

### 4.8 多 GPU / 多进程

```bash
.venv/bin/python scripts/predict_parallel.py \
  --input /路径/input.tsv \
  --output outputs/parallel_job \
  --workers 2 --gpus 1,0 \
  --anchor-mode auto --batch-size 50
```

按原始样本分配 worker，每个 worker 使用独立目录、日志、检查点和 AF2 进程。`--gpus 1,0` 表示轮流分配到 GPU 1 和 0；顶层 `results.tsv` 汇总结果并恢复原始样本顺序。续跑在该命令增加 `--resume`，保持 worker 数、GPU 配置和输入一致。

需要后台运行时将整个命令交给 `nohup` 或 systemd。并行数应依据空闲显存、内存、序列长度和其他任务决定；这个独立调度器不会自动探测最优并行数或动态缩减 worker。

### 4.9 只准备输入，或者直接使用完整 AF2 预测接口

只做模板检索/对齐，不占用 GPU 推理：

```bash
./mhcalphafold.sh prepare \
  --input examples/mhc1.tsv \
  --output outputs/prepared
```

生成的 `work/batch00000/targets.tsv` 可以交给完整 AFfine 预测入口：

```bash
CUDA_VISIBLE_DEVICES=1 ./mhcalphafold.sh af2 -- \
  --targets "$PWD/outputs/prepared/work/batch00000/targets.tsv" \
  --data_dir "$PWD/AFfine/af_params/params_original" \
  --model_names model_2_ptm --model_params_files classic \
  --num_recycles 3 --num_templates 4 --ignore_identities
```

该底层接口按 targets 中的 `outfile_prefix` 写文件，不自动生成统一的 `results.tsv`。需要完整整理时使用高层 `predict`；从 `prepare` 的目录继续预测可在同样参数下执行 `predict --resume`。

高级接口完整保留了上游的自定义 IG PDB/坐标数组、原子/残基 mask、模板 mask、结构采样、返回表示等参数：

```bash
./mhcalphafold.sh af2 -- --help
```

高层也保留 `--sampling-mode --n-times-sampling 50 --sampling-seed 42` 及 `--pep-sampling` 接口。这些是上游高级模式，本轮独立抽取仅验证默认 IG 推理；使用前请单独验证产物和采样语义。

### 4.10 可选微调权重和 TE

```bash
# 使用目录内已有的 PMGen 微调权重；默认不会启用它
CUDA_VISIBLE_DEVICES=1 ./mhcalphafold.sh predict \
  --input examples/mhc1.tsv \
  --models model_2_ptm_ft \
  --output outputs/finetuned

# 已安装并授权 Modeller 后，选择模板工程路线
CUDA_VISIBLE_DEVICES=1 ./mhcalphafold.sh predict \
  --input examples/mhc1.tsv --mode te \
  --output outputs/template_engineering
```

其他 AF2 pTM 模型可通过 `--models model_1_ptm model_2_ptm` 等指定，但必须在 `--params-dir/params/` 下提供各自权重。本目录仅附带 `model_2_ptm` 的原版权重。`--fine-tuned-params` 可以覆盖微调权重路径。

## 5. 输出怎么看

```text
outputs/my_batch/
├── input.tsv                  # 标准化的原始样本
├── expanded.tsv               # 初始锚点候选表及 source_id
├── run.json                   # 配置及任务指纹
├── progress.json              # 状态、已完成/剩余数量
├── results.tsv                # 每条原始样本的一条最佳结果
├── best_structures/<id>.pdb    # 最佳复合物结构
├── scores/<id>_plddt.npy       # 每残基 pLDDT，已裁剪至真实长度
├── scores/<id>_pae.npy         # PAE 方阵，已裁剪至真实长度
├── scores/<id>_ptm.npy         # pTM
├── checkpoints/<id>.json      # 完成记录及文件 SHA256
├── logs/batch00000.log         # AF2 详细日志
└── work/batch00000/
    ├── candidates.tsv         # 实际使用的候选锚点（含 NetMHC 返回值）
    ├── targets.tsv            # AF2 输入
    ├── pandora/               # 模板与对齐日志
    ├── alignment/             # 模板对齐表
    └── alphafold/             # 全部候选 PDB/原始分数
```

`results.tsv` 的主要列：

| 列 | 含义 |
| --- | --- |
| `selected_candidate` / `anchors` / `model` | 胜出的锚点候选及模型 |
| `core_plddt` | 用于选优的肽核心区域平均 pLDDT |
| `peptide_mean_plddt` | 整条肽平均 pLDDT |
| `overall_mean_plddt` | MHC + 肽的整体平均 pLDDT |
| `ptm` | AF2 的 pTM 置信度 |
| `pdb_path` / `*_path` | 相对于该结果目录的产物路径 |

核心区域沿用 PMGen 定义。设 MHC 总长度为 `M`，首末锚点为 1-based 的 `a,b`，使用 Python 切片：

```python
core = plddt[M + max(0, a - 2) : M + b - 1]
```

它通常包括首锚点前一位，但不包含末锚点，不能直接解释成“两个锚点之间所有残基的平均”。选优只使用核心分数，不按整个复合物 pLDDT 排序。

结构文件通过序列一致性、真实残基分数长度、有限坐标及置信度范围校验后才记为完成。AF2 为同批长度做的 padding 不进入最终分数。工作目录保留全部候选便于审查，较大批次需要额外磁盘空间；本独立版不会自动清理它们。

最佳 PDB 导出时规范化分链：MHC-I 为 **A=MHC、B=肽**，MHC-II 为 **A=Alpha、B=Beta、C=肽**，各链残基从 1 编号；坐标和 B-factor 不变。`work/` 中保留 AFfine 原始 PDB，其输入链可能共用一个 PDB chain ID，以 residue index 跳变区分。

## 6. 验证与已知限制

```bash
.venv/bin/python -m unittest discover -s tests -v
./mhcalphafold.sh check

# 仅当还保留原项目时，可重新核对抽取一致性；日常推理不需要原目录
.venv/bin/python scripts/verify_extraction.py --source /home/c/czh/pmgen
```

抽取一致性记录见 `validation/source-parity.json`：AFfine 预测文件及权重逐文件比较；资源逐文件核对 SHA256；检查所有资源和环境符号链接均未指向原项目。`SOURCE_MANIFEST.json` 保存来源提交、资源清单和原部署中的兼容补丁。

本轮验证覆盖输入约束、MHC-I/MHC-II 链格式、锚点枚举、分数 padding 裁剪、损坏检查点拒绝、默认 IG GPU 推理和续跑。实际样例输出保存在 `outputs/`，详细结果见 `validation/smoke-validation.json`。

尚未做独立验证的路线包括 Modeller TE、授权 NetMHC 锚点预测、微调权重效果和高级结构采样。接口与对应上游实现保留，不应把“代码可调用”当作这些路线已通过本轮端到端测试。

常见情况：

- **找不到 GPU / ptxas：** 检查 `CUDA_VISIBLE_DEVICES`、驱动和 `CUDA_ROOT`，再运行 `check`。
- **首次预测慢：** JAX 编译和模板准备需要时间；从 `logs/` 查看进度，后续同形状输入可复用缓存。
- **输入仅有 HLA 伪序列：** 先映射到真实 MHC 结合域序列，再使用本工具。
- **NetMHC 锚点失败：** 上游 PANDORA 可能回退到 canonical anchors；检查 `pandora.log` 和实际 `candidates.tsv`，或直接选择枚举路线。
- **分数高但肽姿态仍有疑问：** pLDDT 不能代替实验验证；结合 PAE、结合槽位置和结构检查判断。
- **输出目录已有任务：** 同任务使用 `--resume`；改变输入或模型参数时使用新输出目录。

## 7. 来源与许可

本项目是功能抽取与部署封装，科学方法归原作者。保留 [PMGen 许可证](licenses/PMGen-LICENSE)、[AFfine 许可证](AFfine/LICENSE)、[原始 AF2 许可证](AFfine/original_alphafold_LICENSE) 和 [PANDORA 许可证](PANDORA/LICENSE)；模型参数、模板数据及可选授权工具遵循各自上游条款。

- PMGen：<https://github.com/soedinglab/PMGen>，抽取提交 `628d9e807c6a2408add8969770b06c059c2b9ce1`。
- PMGen 方法论文：<https://pmc.ncbi.nlm.nih.gov/articles/PMC13308714/>。
- AlphaFold2：<https://github.com/google-deepmind/alphafold>。
- AFfine：<https://github.com/phbradley/alphafold_finetune>。
- Initial Guess 相关实现：<https://github.com/nrbennet/dl_binder_design>。
- PANDORA：<https://github.com/X-lab-3D/PANDORA>。
