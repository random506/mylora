# RhoEdit

本仓库用于研究大语言模型知识编辑中的“编辑效果与原有能力保持”问题。当前主方法是 **RhoEdit**：利用编辑数据与能力保持数据的 K-FAC 曲率统计，在广义特征空间中对梯度进行软约束预条件，再交给 Adam 或 SGD 更新模型参数。

仓库同时保留 CrispEdit、FT、LoRA、LocBF-FT、MEND、MEMIT、AlphaEdit、AlphaEditFT、ROME、UltraEdit 和 WISE，用作对比方法或消融实验。本文档以 2026-09-08 的代码状态为准；存在但尚未完整接线的入口会在“当前限制”中明确说明。

## 方法概览

RhoEdit 在每个待编辑权重块上同时考虑两类曲率：

- 编辑曲率：由 `task_mom2_dataset` 或当前编辑请求得到，记为 `A_e`、`B_e`。
- 能力曲率：由 `mom2_dataset` 的预训练/保持数据得到，记为 `A_c`、`B_c`。

代码中的软约束 Newton 系统为：

```text
B_e * dW * A_e + lambda * B_c * dW * A_c = -G
```

`ProjectedAdam` 和当前软约束版 `ProjectedSGD` 会构造广义基，使编辑曲率被白化、能力曲率被对角化：

```text
Q_A^T A_e Q_A = I
Q_A^T A_c Q_A = diag(a)

Q_B^T B_e Q_B = I
Q_B^T B_c Q_B = diag(b)
```

随后按下面的谱过滤形式处理梯度：

```text
G_projected = Q_B [ (R_B^T G R_A) / (1 + lambda * outer(b, a)) ] Q_A^T
```

其中：

- `soft_lambda` 控制能力保持约束。值越大，高能力曲率方向的更新收缩越强。
- `newton_damping` 为编辑曲率的相对阻尼，用于改善 Cholesky 分解和广义特征分解的数值稳定性。
- `lr` 控制预条件后 Adam/SGD 的实际更新步长。
- `mom2_n_samples` 和 `task_mom2_n_samples` 分别控制能力统计与编辑任务统计的样本数。

## RhoEdit 实现

| 文件 | 作用 |
| --- | --- |
| `run_rhoedit.py` | RhoEdit 命令行入口、模型加载、CLI 参数覆盖、模型保存 |
| `hparams/RhoEdit/*.yaml` | 不同模型的编辑层、数据集、学习率和 soft K-FAC 参数 |
| `easyeditor/models/rhoedit/utils.py` | Adam 路径：统计加载、缓存构造、训练循环、顺序编辑实验实现 |
| `easyeditor/models/rhoedit/projected_adam.py` | 广义特征分解、谱过滤、Adam 动量重投影和诊断统计 |
| `easyeditor/models/rhoedit/utils_sgd.py` | SGD 路径的缓存构造与训练循环 |
| `easyeditor/models/rhoedit/projected_adam_sgd.py` | 当前 soft K-FAC SGD 优化器实现 |
| `easyeditor/models/rhoedit/CrispEdit_hparams.py` | Adam 超参数 dataclass 和 YAML 加载器 |
| `easyeditor/models/rhoedit/CrispEdit_hparams_sgd.py` | SGD 超参数 dataclass 和 YAML 加载器 |
| `easyeditor/models/rome/layer_stats.py` | 当前 RhoEdit 实际调用的 K-FAC 数据加载与统计实现 |
| `easyeditor/tools/tracker.py` | W&B、SwanLab 或无跟踪模式的统一接口 |

当前 Adam 流程如下：

```text
run_rhoedit.py
  -> 读取 hparams/RhoEdit/<model>.yaml
  -> 加载本地模型与 tokenizer
  -> 计算或读取能力数据 K-FAC 统计
  -> 计算或读取编辑任务 K-FAC 统计
  -> 构造 ProjectedAdam
  -> 只训练 YAML layers 对应的 down_proj 权重
  -> 保存完整模型与 tokenizer 到 HF_CACHE_DIR
```

`ProjectedAdam` 在缓存切换时还会重新投影已有 Adam 一阶动量，避免动量继续停留在旧曲率基中。默认会记录广义特征值和投影前后梯度范数，文件位置由实现和 `STATS_DIR` 决定。

## 目录结构

```text
.
|-- easyeditor/
|   |-- editors/                 # EasyEdit 风格编辑器接口
|   |-- evaluate/                # 编辑质量和安全性评测
|   |-- models/
|   |   |-- rhoedit/             # 主方法 RhoEdit
|   |   |-- crispedit/           # CrispEdit 对比方法
|   |   `-- ...                  # 其他 EasyEdit 基线
|   |-- tools/                   # W&B / SwanLab 跟踪器
|   `-- trainer/                 # MEND 等训练组件
|-- hparams/                     # 按方法和模型组织的 YAML
|-- data/                        # 编辑数据集
|-- scripts/experiments/         # GPU 调度和批量实验脚本
|-- scripts/hparam_search.py     # RhoEdit 历史超参搜索辅助器
|-- run_rhoedit.py               # RhoEdit 主入口
|-- run_crispedit.py             # CrispEdit / FT / LoRA 对比入口
|-- edit.py                      # EasyEdit 基线统一入口
|-- run_base_benchmarks.py       # 原有能力评测
|-- run_edited_benchmarks.py     # 编辑成功率与泛化评测
`-- utils.py                     # 数据读取与模型保存
```

`logs/`、模型目录、K-FAC 缓存和 `docs/experiments/` 下的结果分析属于运行产物，不应默认作为源代码提交。

## 环境安装

完整训练和评测主要面向 Linux、NVIDIA GPU 与 CUDA 环境。依赖中固定了 `torch==2.4.0`、`transformers==4.46.2`、`vllm==0.6.3.post1` 和 `lm_eval==0.4.8`。

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

仅查看脚本或运行结果收集器不需要 GPU；RhoEdit 训练、K-FAC 统计和 vLLM 评测通常需要较大显存。

## 环境变量

从模板创建本地配置：

```bash
cp .env.example .env
```

建议配置：

```dotenv
HF_CACHE_DIR=/absolute/path/to/models/
HF_DATASETS_DIR=/absolute/path/to/hf_datasets
STATS_DIR=/absolute/path/to/kfac_stats
EDIT_DATA_DIR=/absolute/path/to/repo/data

BASE_URL=https://openrouter.ai/api/v1
MODEL=deepseek/deepseek-v4-flash
API_KEY=

WANDB_API_KEY=
SWANLAB_API_KEY=

HF_DATASETS_OFFLINE=1
HF_HUB_OFFLINE=1
```

注意事项：

- `HF_CACHE_DIR` 必须以 `/` 结尾。训练保存和编辑质量评测仍使用字符串拼接路径。
- `run_rhoedit.py` 使用 `local_files_only=True`。模型必须已经下载到本地，或 YAML 的 `model_name` 必须能在本地 Hugging Face 缓存中解析。
- 如果本地没有数据集缓存，应暂时移除或关闭 `HF_DATASETS_OFFLINE`、`HF_HUB_OFFLINE`。
- `STATS_DIR` 保存 K-FAC 统计；不同模型、数据集、层和样本数会生成不同缓存。
- `EDIT_DATA_DIR` 供 K-FAC 数据加载器查找 CounterFact、ZsRE 和 WikiBigEdit 等本地 JSON。
- 只有使用 LLM-as-a-Judge 时才需要 `BASE_URL`、`MODEL` 和 `API_KEY`。

不要提交 API 密钥、服务器绝对路径、模型 checkpoint 或原始实验日志。

## 数据集

| `data_type` | 编辑数据文件 | `run_rhoedit.py` |
| --- | --- | --- |
| `zsre` | `data/zsre_mend_3k.json` | 支持 |
| `zsre10k` | `data/zsre_mend_10k.json` | 支持 |
| `counterfact` | `data/counterfact-edit_3k.json` | 支持 |
| `wiki` | `data/wiki_big_edit_3k.json` | 支持 |
| `safeedit_train` | `data/SafeEdit_train.json` | 实验性支持 |
| `safeedit_test` | `data/SafeEdit_test.json` | 实验性支持 |

RhoEdit 同时读取 YAML 中的两个统计数据集参数：

```yaml
mom2_dataset: "wikipedia"
task_mom2_dataset: "counterfact-edit_3k"
```

`--data_type` 只决定本次训练请求，不会自动修改 `task_mom2_dataset`。运行不同数据集实验前，必须检查 YAML 中的任务统计数据是否符合实验设计。例如运行 ZsRE 时，如需使用 ZsRE 编辑曲率，应将 `task_mom2_dataset` 配置为相应的 ZsRE 数据名，而不是继续沿用 CounterFact。

## 模型配置

`--model` 表示配置文件名，不是任意 Hugging Face 模型路径：

```text
--model llama3-8b
    -> hparams/RhoEdit/llama3-8b.yaml

--model llama3.2-3b
    -> hparams/RhoEdit/llama3.2-3b.yaml
```

当前 `hparams/RhoEdit/` 包含：

| 配置 | 模型 | 编辑层 | 当前状态 |
| --- | --- | --- | --- |
| `llama3-8b.yaml` | Llama 3 8B Instruct | 19-23 | Adam 加载器所需的 `RHOEDIT` 标识已配置 |
| `llama3.2-3b.yaml` | Llama 3.2 3B Instruct | 16-20 | 根据 28 层结构生成的初始参数，尚需实验调优 |
| `qwen2.5-7b.yaml` | Qwen2.5 7B Instruct | 15-19 | 当前仍写为 `CRISPEDIT`，不能直接通过 RhoEdit Adam 加载器 |

所有对比方法目录也提供了 `llama3.2-3b.yaml`，用于后续统一比较；这些配置是根据 Llama 3 8B 和 Qwen2.5 7B 相对层深生成的初始值，不代表已经完成 GPU 基准验证。

## 运行 RhoEdit

下面的例子使用当前 Llama 3 8B 配置，并选择与 YAML 任务统计一致的 CounterFact：

```bash
CUDA_VISIBLE_DEVICES=0 python run_rhoedit.py \
  --model llama3-8b \
  --data_type counterfact \
  --alg_name rhoedit_adam \
  --cache_sample_num 10000 \
  --edit_sample_num 3000 \
  --batch_size 32 \
  --lr 5e-4 \
  --newton_damping 1e-2 \
  --soft_lambda 0.1 \
  --plat_name none \
  --no_wandb
```

主要参数：

| 参数 | 含义 |
| --- | --- |
| `--model` | `hparams/RhoEdit/` 下的配置文件名 |
| `--data_type` | 本次编辑数据集 |
| `--alg_name` | 当前建议使用 `rhoedit_adam` |
| `--cache_sample_num` | 能力保持 K-FAC 统计样本数 |
| `--edit_sample_num` | 任务 K-FAC 统计样本数 |
| `--batch_size` | 编辑训练 batch size |
| `--lr` | Adam/SGD 学习率，覆盖 YAML |
| `--newton_damping` | 广义特征分解阻尼，覆盖 YAML |
| `--soft_lambda` | 能力保持软约束强度，覆盖 YAML |
| `--recalculate_cache` | 权重变化超过阈值时重新计算能力统计 |
| `--recalculate_weight_threshold` | 触发重算的相对权重变化阈值 |
| `--plat_name` | `wandb`、`swanlab` 或 `none` |

输出目录名由模型、算法、数据集、阻尼、`lambda` 和学习率组成，例如：

```text
llama3-8b_rhoedit_adam_counterfact_0_01_0_1_0_0005
```

完整模型和 tokenizer 会保存到：

```text
${HF_CACHE_DIR}<run-id>/
```

### 诊断输出

`ProjectedAdam` 当前默认启用谱统计与梯度范数统计。常见输出包括：

```text
projected_adam_factor_stats.json
${STATS_DIR}/projected_adam_grad_norm_stats.json
```

这些文件用于检查广义特征值分布、阻尼是否足够，以及投影前后梯度收缩比例。批量实验时应按 run 隔离或及时归档，避免后一次运行覆盖前一次结果。

## 超参数搜索

RhoEdit 的主要搜索轴是：

```text
lr -> newton_damping -> soft_lambda
```

`scripts/hparam_search.py` 实现了命令生成、结果读取和按 WILD/能力下降约束排序的逻辑，历史网格为：

```text
lr:              5e-5, 1e-4, 2e-4, 5e-4, 1e-3
newton_damping:  1e-5, 1e-3, 1e-2, 1e-1
soft_lambda:     0.1, 1.0, 9.0
```

查看搜索协议：

```bash
python scripts/hparam_search.py print --protocol
```

查看某一轴：

```bash
python scripts/hparam_search.py print --axis lr --verbose
```

注意：该辅助器和两个 2026-08-26 调度脚本仍使用历史名称 `curpedit_adam`，并检查已经不存在的 `hparams/CurpEdit/llama3-8b.yaml`。它们不能直接用于当前 `run_rhoedit.py`；使用前需统一为 `rhoedit_adam` 和 `hparams/RhoEdit/llama3-8b.yaml`。

## 评测

### 原有能力

```bash
CUDA_VISIBLE_DEVICES=0 python run_base_benchmarks.py \
  --edited_model_dir <run-id> \
  --model_name llama3-8b \
  --alg_name RhoEdit \
  --data_type counterfact \
  --tasks all \
  --eval_num 200 \
  --no_wandb
```

`--tasks all` 包括：

```text
ifeval, truthfulqa_mc2, mmlu, gsm8k_cot, arc_challenge
```

结果保存到：

```text
logs/<run-id>/capability.json
```

### 编辑质量

Exact Match：

```bash
CUDA_VISIBLE_DEVICES=0 python run_edited_benchmarks.py \
  --edited_model_dir <run-id> \
  --model_name llama3-8b \
  --alg_name RhoEdit \
  --data_type counterfact \
  --context_type qa_inst \
  --evaluation_criteria exact_match \
  --eval_num 3000 \
  --no_wandb
```

LLM-as-a-Judge：

```bash
CUDA_VISIBLE_DEVICES=0 python run_edited_benchmarks.py \
  --edited_model_dir <run-id> \
  --model_name llama3-8b \
  --alg_name RhoEdit \
  --data_type counterfact \
  --context_type no_context \
  --evaluation_criteria llm_judge \
  --judge_batch_size 16 \
  --no_wandb
```

编辑质量评测使用 vLLM，当前固定为单卡、`bfloat16` 和 `gpu_memory_utilization=0.9`。结果目录为：

```text
logs/<run-id>_eval_<criterion>_<context>/
```

运行过程中写入 `results_pending_judge.json`，结束后生成 `results.json` 和 `mean_metrics.json`。

当前不建议使用 `run_benchmarks.py` 统一入口：它的 base 分支导入仓库中不存在的 `run_base_benchmarks_vllm.py`，edited 分支则要求 `run_edited_benchmarks.run()`，但当前脚本没有提供该函数。请直接调用上面的两个评测脚本。

## 结果汇总

```bash
python skills/analyze-results/collect_runs.py
```

筛选实验：

```bash
python skills/analyze-results/collect_runs.py --run <run-id>
python skills/analyze-results/collect_runs.py --run-pattern '*rhoedit*'
```

收集器读取：

- `logs/<run-id>/capability.json`
- `logs/<run-id>_eval_llm_judge_qa_inst/mean_metrics.json`
- `logs/<run-id>_eval_llm_judge_no_context/mean_metrics.json`

报告写入 `docs/experiments/<timestamp>_results_analysis.md`。缺失任一结果文件时，报告会将对应实验标记为不完整。

## 对比方法

对比方法不是本仓库的主方法，主要用于 Table 1 和消融实验。

| 方法 | 入口 | 配置目录 | 说明 |
| --- | --- | --- | --- |
| CrispEdit | `run_crispedit.py` | `hparams/CrispEdit/` | 单一能力子空间投影对比 |
| FT | `run_crispedit.py --no_crisp` 或 `edit.py` | `hparams/FT/` | 无曲率约束微调 |
| LoRA | `run_crispedit.py --no_crisp --perform_lora` 或 `edit.py` | `hparams/LoRA/` | LoRA/AdaLoRA 编辑 |
| LocBF-FT | `locft-bf.py` | `hparams/FT/` | locality-aware baseline |
| MEND | `edit.py --editing_method MEND` | `hparams/MEND/` | EasyEdit 基线 |
| MEMIT | `edit.py --editing_method MEMIT` | `hparams/MEMIT/` | EasyEdit 基线 |
| AlphaEdit | `edit.py --editing_method AlphaEdit` | `hparams/AlphaEdit/` | null-space 投影基线 |
| AlphaEditFT | `alphaedit_ft.py` | `hparams/AlphaEditFT/` | AlphaEdit 风格 FT 实验入口 |
| ROME | `edit.py --editing_method ROME` | `hparams/ROME/` | EasyEdit 基线 |
| UltraEdit | `edit.py --editing_method UltraEdit` | `hparams/UltraEdit/` | EasyEdit 基线 |
| WISE | `edit.py --editing_method WISE` | `hparams/WISE/` | EasyEdit 基线 |

### Table 1 调度脚本

| 脚本 | 模型与数据集 | 状态 |
| --- | --- | --- |
| `20260827_table1-easyeditor-baselines.sh` | Llama 3 8B / ZsRE | 单数据集基线训练 |
| `20260827_table1-easyeditor-baselines-counterfact.sh` | Llama 3 8B / CounterFact | 单数据集基线训练 |
| `20260827_table1-easyeditor-baselines-wiki.sh` | Llama 3 8B / WikiBigEdit | Wiki 脚本未包含 MEMIT 和 AlphaEdit |
| `20260908_table1-easyeditor-baselines-qwen2.5-7b.sh` | Qwen2.5 7B / ZsRE、CounterFact、WikiBigEdit | 三阶段串行运行全部基线 |

调度脚本依赖 Bash 4+、`nvidia-smi`、`bc`、`awk`、`sed` 等工具，并按空闲显存分配 GPU。`skip_or_train` 通过 `${HF_CACHE_DIR}<run-id>` 是否存在决定是否跳过训练。

Qwen2.5 脚本按以下顺序执行：

```text
ZsRE -> CounterFact -> WikiBigEdit
```

每个阶段会等待本阶段所有任务结束。如果阶段内任一任务失败，脚本立即退出，后续数据集不会启动。因此出现“只运行 ZsRE”时，应先查看 ZsRE 阶段中失败的方法，而不是检查 CounterFact/Wiki 任务列表。

## 当前限制

以下限制来自当前代码实现，不代表方法设计本身：

1. `run_rhoedit.py` 的 CLI 列出了 `crispedit`，但 `get_hparams()` 只处理 `rhoedit_adam` 和 `rhoedit_sgd`。不要通过该入口选择 `crispedit`。
2. 当前推荐路径是 `rhoedit_adam`。SGD 加载器仍断言 YAML 的 `alg_name` 为 `CURPEDIT`，而现有 RhoEdit YAML 使用 `RHOEDIT` 或 `CRISPEDIT`，因此 `rhoedit_sgd` 尚不能直接运行。
3. `--sequential_edit` 当前不会进入 Adam 顺序编辑实现：主分发先匹配 `rhoedit_adam`，之后的顺序编辑分支不可达；顺序实现内部还直接调用 `wandb.log()`。
4. `hparams/RhoEdit/qwen2.5-7b.yaml` 的 `alg_name` 仍是 `CRISPEDIT`，不能直接由 Adam RhoEdit 加载器读取。
5. `--data_type` 不会自动同步 YAML 的 `task_mom2_dataset`。不同数据集实验必须手动核对任务曲率来源。
6. `scripts/hparam_search.py` 和 2026-08-26 两个 RhoEdit 搜索脚本仍使用历史名称 `curpedit_adam` 与旧配置路径。
7. `run_benchmarks.py` 当前不是可用的统一评测入口，请直接运行 `run_base_benchmarks.py` 和 `run_edited_benchmarks.py`。
8. `run_rhoedit.py`、`utils.py` 和 `run_edited_benchmarks.py` 仍依赖 `HF_CACHE_DIR` 字符串拼接；路径结尾错误会导致模型查找或保存失败。
9. 部分 Python 源码注释存在历史编码损坏，但不影响本 README 的 UTF-8 内容。
10. 新增的 Llama 3.2 3B 配置是结构适配后的初始超参数，尚不能视为已经验证的最佳参数。

## 开发检查

修改 Python 文件后至少运行：

```bash
python -m py_compile path/to/changed_file.py
```

结果收集器的 CPU 测试：

```bash
python -m unittest skills/analyze-results/test_collect_runs.py -v
```

如果环境安装了 Ruff：

```bash
ruff check <changed-files>
```

GPU 实验应记录模型、数据集、随机种子、设备、编辑层、统计样本数、`lr`、`newton_damping`、`soft_lambda` 和完整评测产物。不要把未实际运行的基准结果描述为已验证结果。
