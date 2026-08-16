# <实验名称>

## 研究问题

<用一个明确、可证伪的问题说明本实验要解决什么。>

## 现有证据

<引用 `skills/analyze-results/collect_runs.py` 新生成的报告和 canonical 指标。没有可用基线时明确说明。>

## 假设

<说明预期结果、原因，以及什么结果会推翻该假设。>

## 对照变量

<列出唯一变化维度、固定参数、control 和随机种子。>

## 实验矩阵

| run_id | model | data_type | alg_name | lr | energy_threshold | batch_size | cache_sample_num | GPUs | 变化说明 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| <run-id> | <model> | <data> | <algorithm> | <lr> | <threshold> | <batch> | <samples> | <gpus> | <change> |

## 评测方案

<明确选择 train、capability、qa_inst、no_context 中的哪些阶段，并记录评测参数。未指定时四项全部执行。>

## 选择规则

- 主要指标：<metric>
- Capability 退化门槛：<value 或 待确认>
- No Context 退化门槛：<value 或 待确认>
- QA Inst 退化门槛：<value 或 待确认>
- 排序规则：先淘汰越过门槛的运行，再按主要指标排序；不计算综合总分。

## 资源与前置条件

<记录 GPU、显存、环境变量、缓存、数据、hparams 和任何代码前置修改。>

## 产物

- 计划：`docs/plan/YYYYMMDD_<name>_plan.md`
- 脚本：`scripts/experiments/YYYYMMDD_<name>.sh`
- 模型：`HF_CACHE_DIR/<run_id>`
- 结果：`logs/<run_id>/capability.json`
- 结果：`logs/<run_id>_eval_llm_judge_qa_inst/mean_metrics.json`
- 结果：`logs/<run_id>_eval_llm_judge_no_context/mean_metrics.json`

## 风险与备注

<只记录简短、事实性的运行风险、缺失信息和恢复方式。>
