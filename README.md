# CrispEdit / CurpEdit

本仓库用于研究大语言模型知识编辑中的能力保持问题。当前主线是在 EasyEdit 风格的编辑框架上实现基于 K-FAC 曲率统计的 **CrispEdit**，以及使用软投影 / Newton 预条件的 **CurpEdit-Adam** 和 **CurpEdit-SGD**；仓库同时保留多种基线算法、实验性 LoRA 变体、vLLM 评测和结果汇总工具。

CrispEdit 论文：<https://arxiv.org/abs/2602.15823>

> 本仓库是研究代码快照，不是开箱即用的软件包。模型、编辑数据和 K-FAC 缓存未纳入版本控制，部分旧入口与配置仍有已知问题。首次运行前请先阅读[当前限制](#当前限制)。

## 功能概览

| 范围 | 内容 | 主要位置 |
| --- | --- | --- |
| 主编辑方法 | CrispEdit、CurpEdit-Adam、CurpEdit-SGD | `run_crispedit.py`、`crispedit.py`、`easyeditor/models/{crispedit,curpedit}/` |
| 上游编辑算法 | ROME、MEMIT、FT、MEND、LoRA、WISE、AlphaEdit、UltraEdit 等 | `easyeditor/models/`、`easyeditor/util/alg_dict.py` |
| 实验性方法 | 投影 LoRA、Curvature/Leaky LoRA、SAFE-LoRA、参数投影 | `easyeditor/mymodels/` |
| 编辑质量评测 | rewrite、rephrase、locality；exact match 或 LLM judge | `run_edited_benchmarks.py`、`easyeditor/evaluate/` |
| 基础能力评测 | IFEval、TruthfulQA MC2、MMLU、GSM8K CoT、ARC Challenge | `run_base_benchmarks.py` |
| 结果分析 | 汇总 `capability.json` 和两个上下文的 `mean_metrics.json` | `skills/analyze-results/collect_runs.py` |
| 小规模验证 | LeNet 上的 K-FAC、Hessian、Gauss-Newton 投影对比 | `toy_experiment/` |

`easyeditor/` 还包含多模态、人格、安全和概念编辑基础设施。源码中存在某个模型或算法分支，不等于仓库已经提供相应配置、checkpoint 或完成该组合的验证。

## 仓库结构

```text
.
|-- easyeditor/
|   |-- dataset/          # 数据集封装
|   |-- editors/          # 通用编辑器接口
|   |-- evaluate/         # vLLM 及专项评测
|   |-- models/           # EasyEdit 基线与 CrispEdit/CurpEdit
|   |-- mymodels/         # 实验性 LoRA/投影方法
|   |-- tools/            # W&B / SwanLab 追踪器
|   `-- trainer/          # MEND/SERAC/MALMEN 等训练基础设施
|-- hparams/              # 按算法和模型组织的 YAML 配置
|-- skills/               # 实验规划与结果分析工具
|-- toy_experiment/       # 小模型曲率投影实验
|-- run_crispedit.py      # 主编辑入口
|-- run_base_benchmarks.py
|-- run_edited_benchmarks.py
`-- utils.py              # 根入口共用的数据与保存逻辑
```

`logs/`、`docs/experiments/`、`data/`、模型目录和统计缓存都是本地生成或外部准备的内容，默认不应提交。

## 环境安装

完整编辑和 vLLM 评测面向 Linux、NVIDIA GPU 与 CUDA 环境。历史文档使用 Python 3.9；依赖固定了 `torch==2.4.0`、`transformers==4.46.2`、`vllm==0.6.3.post1` 和 `lm_eval==0.4.8`。

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

若安装时遇到 `pyarrow` 与 `datasets` 的兼容问题，可在确认环境约束后尝试：

```bash
python -m pip install --upgrade datasets pyarrow
```

模型编辑、K-FAC 统计和 vLLM 评测通常需要大显存 GPU；仅结果收集器和部分诊断可以在 CPU 上运行。

## 环境变量

先从模板创建本地配置：

```bash
cp .env.example .env
```

建议至少配置：

```dotenv
HF_CACHE_DIR=/absolute/path/to/models/
HF_DATASETS_DIR=/absolute/path/to/hf_datasets
STATS_DIR=/absolute/path/to/kfac_stats
EDIT_DATA_DIR=/absolute/path/to/this/repo/data
HF_ENDPOINT=https://huggingface.co
```

| 变量 | 用途 | 何时需要 |
| --- | --- | --- |
| `HF_CACHE_DIR` | 模型查找前缀和编辑后模型的保存根目录 | 编辑与编辑质量评测必需 |
| `HF_DATASETS_DIR` | Hugging Face 数据缓存；部分模块导入时就会读取 | 主入口必需 |
| `STATS_DIR` | K-FAC 协方差 / 投影缓存根目录 | CrispEdit、CurpEdit 和相关诊断必需 |
| `EDIT_DATA_DIR` | K-FAC 任务统计所使用的 JSON 数据目录 | 使用编辑数据计算统计时必需 |
| `API_KEY` | LLM-as-a-Judge 的访问密钥 | `llm_judge` 评测必需 |
| `SWANLAB_API_KEY` | SwanLab 登录密钥 | 启用 SwanLab 时需要 |
| `WANDB_API_KEY` | W&B 登录密钥；模板中尚未列出 | 启用 W&B 时需要 |

当前若干脚本用字符串拼接处理 `HF_CACHE_DIR`，因此该值应为绝对目录并以 `/` 结尾。`run_crispedit.py` 使用 `local_files_only=True`，不会在编辑时自动下载模型；请预先下载模型，并在对应 YAML 的 `model_name` 中填写可解析的本地路径或已缓存的 Hub ID。不要把令牌或机器路径提交到仓库。

## 数据准备

编辑入口直接从仓库根目录下的 `data/` 读取 JSON；本仓库当前不包含这些文件。

| `data_type` | 期望文件 | 当前入口 |
| --- | --- | --- |
| `zsre` | `data/zsre_mend_3k.json` | 编辑、编辑质量评测 |
| `zsre10k` | `data/zsre_mend_10k.json` | `run_crispedit.py` |
| `counterfact` | `data/counterfact-edit_3k.json` | 编辑、编辑质量评测 |
| `wiki` | `data/wiki_big_edit_3k.json` | 编辑、编辑质量评测 |
| `safeedit_train` | `data/SafeEdit_train.json` | 主编辑入口、安全评测 |
| `safeedit_test` | `data/SafeEdit_test.json` | 主编辑入口、安全评测 |
| `multi_counterfact` | `data/multi_counterfact.json` | 编辑质量评测 |

主要字段约定：

- ZsRE：`src`、`subject`、`rephrase`、`alt`、`loc`、`loc_ans`。
- CounterFact / Wiki：`prompt`、`subject`、`rephrase_prompt`、`target_new`、`locality_prompt`、`locality_ground_truth`。
- SafeEdit：`adversarial prompt`、`safe generation`、`unsafe generation`、`generalization test`、`question`。

K-FAC 的 `mom2_dataset: wikipedia` 或 `wikitext` 通过 Hugging Face Datasets 加载并优先使用 `HF_DATASETS_DIR` 中的缓存；任务统计数据则可能通过 `EDIT_DATA_DIR` 查找。

## 模型与超参数

`--model` 不是任意模型路径，而是配置文件名。例如：

```text
--model llama3-8b
        -> hparams/CrispEdit/llama3-8b.yaml
        -> hparams/CurpEdit/llama3-8b.yaml
```

运行前至少核对 YAML 中的 `model_name`、`layers`、`rewrite_module_tmp`、`device`、学习率和统计数据设置。当前已有配置如下；“已有 YAML”不代表该组合已经在当前机器上验证。

| 配置族 | 已有 YAML |
| --- | --- |
| CrispEdit、CurpEdit、FT、MEMIT | `llama3-8b`、`mistral-7b`、`qwen2.5-7b` |
| AlphaEdit | `llama3-8b`、`qwen2.5-7b` |
| AlphaEditFT | `llama3-8b`、`mistral-7b`、`qwen2.5-7b` |
| LoRA、MEND、MyEdit、MyLoRA | `llama3-8b`、`qwen2.5-7b` |
| ROME | `llama3-8b copy`、`mistral-7b`、`qwen2.5-7b` |
| WISE | `llama3-8b`、`mistral-7b`、`qwen2.5-7b` |
| UltraEdit | `gemma-3-27b`、`gpt-j-6B`、`llama3-8b`、`mistral-7b`、`phi-4`、`qwen2.5-7b` |

部分基线配置仍包含原作者机器路径、`your/path/...` 占位符或外部 checkpoint；使用前必须改成本机资源。`MyEdit` 和 `MyLoRA` 配置目前没有接入主 CLI。

## 运行编辑

所有命令都应从仓库根目录执行。当前接线最完整的是 Llama/Qwen 的 CurpEdit 分支。下面以 CurpEdit-Adam 为例：

```bash
CUDA_VISIBLE_DEVICES=0 python run_crispedit.py \
  --model llama3-8b \
  --data_type wiki \
  --alg_name curpedit_adam \
  --cache_sample_num 10000 \
  --energy_threshold 0.5 \
  --batch_size 32 \
  --lr 5e-4 \
  --plat_name none
```

将 `--alg_name` 改为 `curpedit_sgd` 可使用 SGD 版本。CurpEdit 会用 CLI 的 `--lr` 覆盖 YAML，而该参数默认值是 `0.7`，因此务必显式给出合理学习率。

主入口可选算法只有：

```text
crispedit | curpedit_adam | curpedit_sgd
```

CrispEdit 的预期命令形式如下，但当前 YAML 的 `alg_name` 大小写与加载器不一致；需先按[当前限制](#当前限制)修正配置后再运行：

```bash
CUDA_VISIBLE_DEVICES=0 python run_crispedit.py \
  --model llama3-8b \
  --data_type zsre \
  --alg_name crispedit \
  --cache_sample_num 10000 \
  --energy_threshold 0.5 \
  --batch_size 32 \
  --plat_name none
```

模型会保存到 `HF_CACHE_DIR` 下，目录名由模型、算法、数据集和关键超参数自动生成。当前保存的是合并后的完整模型与 tokenizer，而不是只保存投影缓存或 LoRA adapter。

## 运行评测

### 基础能力

直接调用 `run_base_benchmarks.py`。相对模型目录会与 `HF_CACHE_DIR` 拼接，绝对路径也可使用。

```bash
CUDA_VISIBLE_DEVICES=0 python run_base_benchmarks.py \
  --edited_model_dir <edited-model-directory> \
  --model_name llama3-8b \
  --alg_name CurpEdit \
  --data_type wiki \
  --tasks all \
  --eval_num 200 \
  --no_wandb
```

`--tasks all` 包括 `ifeval`、`truthfulqa_mc2`、`mmlu`、`gsm8k_cot` 和 `arc_challenge`。当前 `--eval_num` 只限制 MMLU，其他任务仍会运行完整数据。结果写入：

```text
logs/<sanitized-model-directory>/capability.json
```

### 编辑质量

直接调用 `run_edited_benchmarks.py`。该入口使用 vLLM，当前固定为 bfloat16、单卡和 90% GPU 显存利用率；`edited_model_dir` 应传入 `HF_CACHE_DIR` 下的相对目录名。

```bash
CUDA_VISIBLE_DEVICES=0 python run_edited_benchmarks.py \
  --edited_model_dir <edited-model-directory> \
  --model_name llama3-8b \
  --alg_name CurpEdit \
  --data_type wiki \
  --context_type qa_inst \
  --evaluation_criteria exact_match \
  --eval_num 3000 \
  --no_wandb
```

可选上下文为 `qa_inst`、`chat_temp`、`no_context`，判据为 `exact_match` 或 `llm_judge`。使用 LLM judge 时设置 `API_KEY`，并将判据替换为：

```bash
--evaluation_criteria llm_judge --judge_batch_size 16
```

SafeEdit 评测路径要求同时使用 `--context_type chat_temp` 和 `--evaluation_criteria llm_judge`。

评测过程中持续写入 `results_pending_judge.json`，结束后生成 `results.json` 和 `mean_metrics.json`：

```text
logs/<edited-model-directory>_eval_<criterion>_<context>/
```

### 汇总实验

结果收集器把一个基础能力目录与 `no_context`、`qa_inst` 两个 LLM-judge 目录视为同一实验：

```bash
python skills/analyze-results/collect_runs.py
```

筛选运行：

```bash
python skills/analyze-results/collect_runs.py --run <run-name>
python skills/analyze-results/collect_runs.py --run-pattern '*curpedit*'
```

默认报告写入 `docs/experiments/YYYYMMDD_HHMMSS_results_analysis.md`。收集器只把 `capability.json` 和 `mean_metrics.json` 视为权威汇总文件。

## 辅助工具

检查已有 K-FAC 因子的条件数：

```bash
python check_kfac_condition.py \
  --model_name Meta-Llama-3-8B-Instruct \
  --base_ds wikipedia \
  --base_sample_size 10000 \
  --layers 19,20,21,22,23 \
  --factor_damping 1e-5
```

`toy_experiment/run_exp_sweep_recalculate.py` 在 LeNet 上比较 SGD、K-FAC、EKFAC、Hessian 和 Gauss-Newton 等投影；`toy_experiment/plot_roc.py` 读取其缓存并绘图。两者属于独立实验，不参与大模型主流程。其数据、缓存和图片路径都相对于当前工作目录，建议进入 `toy_experiment/` 后运行。当前 sweep 没有 CPU fallback，且保存文件的 `_special.pth` 后缀与绘图脚本默认读取名不一致，运行前需先统一路径。

`scripts/run_0.sh` 是依赖 `nvidia-smi`、`bc`、awk/sed 的 Bash GPU 轮询队列。脚本中的任务名与部分旧参数已经过时，使用前应重新核对 `TASK_LIST`，不要直接批量提交。

## 当前限制

以下问题已从当前代码确认，文档中的命令不会将其描述为已验证流程：

1. `hparams/CrispEdit/*.yaml` 使用 `alg_name: CrispEdit`，而加载器断言要求 `CRISPEDIT`；CurpEdit 的 Mistral 配置也有同类问题。
2. `run_benchmarks.py` 的 base 分支导入未提供的 `run_base_benchmarks_vllm.py`，edited 分支要求 `run_edited_benchmarks.run()`，但该函数不存在。当前请直接调用两套评测脚本。
3. `edit.py` 在导入阶段引用未定义的 `Scheme*` 名称，当前不能作为 EasyEdit 基线统一入口；库内算法实现和 YAML 仍可供修复与二次接线。
4. CrispEdit 顺序编辑路径调用了未传入的 `tracker.log()`，当前 `--sequential_edit` 会失败；CurpEdit 则显式拒绝顺序编辑。
5. `run_crispedit.py` 中 `--edit_sample_num`、`--newton_damping` 未写回超参数；CrispEdit 还忽略 CLI 的 `--lr`。需要修改对应 YAML 的值。
6. `--target_modules` 使用 `type=list`，不适合从命令行传入逗号分隔列表；应保留默认值或从配置修正。
7. `run_crispedit.py` 传给追踪器的 `mode` 与 `--no_wandb` 语义相反。无追踪运行建议使用 `--plat_name none`，不要依赖该开关。
8. LLM judge 当前硬编码 DeepSeek endpoint 与 `deepseek-v4-flash`，`.env` 中的 `BASE_URL` 和 `MODEL` 不会改变实际调用。
9. `calculate_AB_layer.py` 导入了不存在的 helper；AlphaEditFT、ROME、MEND、UltraEdit 的部分配置也依赖缺失文件或机器专用路径。
10. SafeEdit 主编辑路径会重新映射安全数据字段，其训练目标语义尚未在本仓库中给出可靠说明，使用前需结合实验设计复核。

## 开发与检查

当前仓库的聚焦 CPU 测试是结果收集器：

```bash
python -m unittest skills/analyze-results/test_collect_runs.py -v
```

修改 Python 文件后至少执行：

```bash
python -m py_compile path/to/changed_file.py
```

若环境中已安装 Ruff，再运行 `ruff check path/to/changed_file.py`。

完整实验应记录命令、模型、数据集、随机种子、设备、超参数和指标差异。不要把未运行的 GPU 基准写成已验证结果。

## 代码来源说明

仓库基于 EasyEdit 风格的编辑器与算法实现继续开发，并内嵌了 PEFT、Knowledge Neurons、BLIP2 等上游/第三方代码子树。CrispEdit、CurpEdit 和 `easyeditor/mymodels/` 中的研究变体应与这些基础设施区分；使用或再分发时请同时检查各子目录中的原始许可证和版权声明。
