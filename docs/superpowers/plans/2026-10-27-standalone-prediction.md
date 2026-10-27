# MHCalphafold 独立预测模块实施计划

> 执行方式：在当前会话逐项实施并验收；用户已经授权抽取、部署说明和推理说明，不另设审批步骤。

**目标：** 在 `/home/c/czh/MHCalphafold` 提供可独立运行的 pMHC AlphaFold2 结构预测目录及完整中文 README。

**架构：** 复制经验证的 AFfine 预测代码与 PANDORA 模板准备代码，移除顶层肽生成入口。新增统一 CLI 完成输入校验、锚点枚举、模板准备、预测和结果整理；所有资源使用目录内相对路径，并提供独立环境安装脚本。

**技术栈：** Python 3.10、JAX CUDA 12、Haiku、TensorFlow CPU、Biopython、BLAST+、MUSCLE。

**需求来源：** 用户要求完整单独抽取 AlphaFold2 预测功能，命名 MHCalphafold，并在 README 说明相对 AlphaFold2 的适配、部署和推理；另外提供 300 characters 英文介绍。

## 全局约束

- 原 PMGen 源文件、后台服务及现有预测结果不修改、不停止。
- 支持 MHC-I / MHC-II、Initial Guess / 可选 Template Engineering、原版 / 已有微调权重和上游高级预测入口。
- 默认原版 `model_2_ptm`、4 个模板、3 次 recycling，无 ProteinMPNN / BioEmu 依赖。
- 资源、Python 环境和代码不通过指向原 PMGen 目录的符号链接实现独立性。
- README 使用简体中文；如实区分已验证功能、需要许可证的可选功能和未经本轮验证的高级模式。

## 文件职责与执行步骤

### 任务 1：提取代码与资源

- [x] 复制 `AFfine/` 的完整预测实现和原始许可证，排除训练入口、缓存和旧样例。
- [x] 复制 `PANDORA/`，将数据库路径改为 `resources/pandora_db`，支持环境变量覆盖。
- [x] 在 `mhcalphafold/backend.py` 保留结构建模与批量模板准备类；`utils/processing_functions.py` 只保留预测依赖的函数闭包。
- [x] 复制原版及已有微调权重、模板库、BLAST/MUSCLE 工具；生成来源与 SHA256 清单。
- [x] 校验新目录没有指向原项目的资源路径或符号链接。

### 任务 2：独立 CLI 和部署

- [x] 创建 `predict.py`、`mhcalphafold/inputs.py`、`mhcalphafold/outputs.py`、`mhcalphafold/cli.py`，支持 TSV/CSV、单样本、MHC FASTA、手动/枚举/NetMHC 锚点、仅准备输入、已有输入推理、候选择优及完整日志。
- [x] 创建 `mhcalphafold.sh`、`scripts/activate.sh`、`scripts/install.sh`、`requirements.txt`、`scripts/check_install.py`，在本机建立独立运行环境并验证 GPU。
- [x] 加入最小 MHC-I/MHC-II 样例；所有命令可从其他工作目录执行。

### 任务 3：验证与交付文档

- [x] 验证非法序列/重复 ID/锚点越界、枚举数量、MHC-II 分链和分数补齐裁剪。
- [x] 使用独立目录真实完成一个 MHC-I 样例 GPU 推理，校验 PDB 残基、有限坐标和分数数组。
- [x] 对比原实现与抽取代码，记录模板准备与预测核心一致性；检查旧后台服务仍正常。
- [x] README 写入功能范围、AF2 适配原理、环境与资源部署、单条/批量/后台推理、输出格式、许可证、故障处理、验证记录和上游来源。
- [x] 提供恰好 300 字符（包含空格及标点）的英文介绍。

## 验收结果

- 已生成独立目录，包含源码、约 1.1 GiB 权重与资源，以及独立 Python 运行环境。
- MHC-I 的 3 个锚点候选、MHC-II、两进程预测及续跑均通过。
- 8 项协议测试通过；AFfine 的 81 个文件与原实现一致，1413 个资源文件哈希通过。
- 导出的最佳 PDB 使用独立 MHC/肽链，坐标保持不变。
- 原批量服务 PID 4028587 保持运行，12 路 worker 未调整。
