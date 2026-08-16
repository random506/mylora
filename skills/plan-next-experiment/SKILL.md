---
name: plan-next-experiment
description: Plan the next mylora experiment from canonical logs and create a Chinese experiment plan plus a standalone GPU-scheduled shell script. Use when Codex needs to choose a controlled experiment, define a training or evaluation matrix, compare CurpEdit variants, or turn an experiment idea into docs/plan and scripts/experiments artifacts.
---

# Plan Next Experiment

## Outputs

Create these linked artifacts from the repository root:

- `docs/plan/YYYYMMDD_<name>_plan.md`
- `scripts/experiments/YYYYMMDD_<name>.sh`

Use the current Asia/Shanghai date and a short ASCII `kebab-case` name. Refuse to overwrite either path. Write the plan in Chinese while preserving code identifiers, metric names, paths, and commands in English.

The only exception to the two-output contract is a requested sweep that the current CLI cannot express independently. In that case, create the plan, document the prerequisite, and do not create a misleading shell script.

Use [assets/plan-template.md](assets/plan-template.md) for the plan and [assets/gpu_scheduler.sh](assets/gpu_scheduler.sh) for the script. Replace every angle-bracket prompt in the plan and every `__UPPER_SNAKE_CASE__` marker in the scheduler. The generated artifacts must reference each other.

## Workflow

1. Locate the repository root and inspect the current training entry point, evaluation entry points, relevant hparams, data files, and `calculate_model_name()` implementation. Treat code as authoritative; do not use README command examples.
2. Resolve execution scope independently for `train`, `capability`, `qa_inst`, and `no_context`. Include all four when the user does not specify a scope.
3. Analyze relevant existing results before choosing the matrix:

```bash
python skills/analyze-results/collect_runs.py
```

Use `--run` for user-named baselines and the default full comparison otherwise. Cite the generated report in `现有证据`. Use only canonical `capability.json` and `mean_metrics.json` values. If no baseline exists, state that explicitly.

4. Define one research question. Prefer the smallest controlled matrix that resolves it. Include a control and change exactly one matrix column relative to that control. For an optimizer comparison, fix `lr`; for an `lr` sweep, fix `alg_name`. Split a proposed `alg_name x lr` grid into separate plans. Keep every other CLI and hparam value explicit and fixed.
5. Record each matrix row with at least `run_id`, `model`, `data_type`, `alg_name`, `lr`, `energy_threshold`, `batch_size`, `cache_sample_num`, and required GPU count.
6. Derive each `run_id` from the current `calculate_model_name()` behavior and verify the explicit value. Never infer training parameters from a log directory name.
7. Establish parameter provenance for the control. Canonical result files prove metrics, not training configuration. If a historical run lacks an authoritative saved config, use it only as motivation and either add a fresh control row or ask the user for the original command. Never present it as a strict control.
8. Verify every swept value can be passed independently through the current CLI and reaches hparams. Never mutate one shared YAML between concurrent runs. If CLI support is missing, create only the plan, mark the code prerequisite, and stop before creating the shell script.
9. Materialize the scheduler by replacing `__PLAN_PATH__`, `__TRAIN_TASKS__`, `__EVAL_TASKS__`, and `__EXPERIMENT_PREFLIGHT__`. When both stages are present, run all training tasks first, require all of them to succeed, then run evaluation tasks.
10. Add preflight checks before scheduling. Check required commands and files, required `.env` variables, relevant hparam/data inputs, and existing model directories for evaluation-only plans.
11. Validate the outputs without starting an experiment.

## Command Contract

Follow these repository entry points, regardless of README examples:

- Training: `python run_crispedit.py`
- Capability: `python run_base_benchmarks.py`
- Edited quality: `python run_edited_benchmarks.py`

Materialize commands with these shapes, adding only current CLI options that the matrix requires:

```bash
python run_crispedit.py \
  --model <model> --data_type <data_type> \
  --cache_sample_num <cache_sample_num> \
  --energy_threshold <energy_threshold> --batch_size <batch_size> \
  --wandb_project <wandb_project> --alg_name <alg_name> --lr <lr>

python run_base_benchmarks.py \
  --edited_model_dir <run_id> --model_name <model> \
  --alg_name Base --data_type <data_type> --eval_num 200

python run_edited_benchmarks.py \
  --edited_model_dir <run_id> --data_type <data_type> \
  --context_type <qa_inst-or-no_context> --alg_name ft_edit \
  --model_name <model> --evaluation_criteria llm_judge \
  --judge_batch_size 16 --no_wandb
```

Default capability arguments are `--eval_num 200` and `--alg_name Base`.

Default edited-quality arguments are `--alg_name ft_edit --evaluation_criteria llm_judge --judge_batch_size 16 --no_wandb`. Generate separate `--context_type qa_inst` and `--context_type no_context` tasks. Leave `--eval_num` unset to use the entry point's default of 3000 unless the user overrides it.

Use the scheduler defaults unless the user overrides them:

```bash
MIN_VRAM_GB=47
POLL_INTERVAL=60
MAX_WAIT_HOURS=72
MAX_CONCURRENT_GPUS=2
```

Default every task to one GPU, with per-row overrides allowed.

## Scope Semantics

- No scope specified: generate training and all three evaluations.
- Training only: generate no evaluation tasks.
- Evaluation only: generate selected evaluation tasks and require every `HF_CACHE_DIR + run_id` directory to exist during preflight.
- Training plus evaluations: place a global stage barrier after training.
- Any training failure: print the failed row and command, exit nonzero, and start no evaluation tasks.

## Selection Rules

Keep capability, no-context editing, and QA-instruction editing separate. Never calculate a combined score.

For each plan:

1. Name one primary metric.
2. Record user-supplied degradation limits for the other metric families.
3. Reject runs that violate a limit.
4. Rank eligible runs by the primary metric.
5. Mark an unspecified degradation limit as `待确认`; never invent a percentage.

If the user asks Codex to choose the experiment, prioritize a missing strict control or the single-variable experiment with the highest expected information value. Explain that choice in `研究问题` and `假设`.

## Preflight Contract

Generate checks for:

- Commands: `python`, `nvidia-smi`, `bc`, `awk`, and `sed`.
- Environment: `HF_CACHE_DIR`, `HF_DATASETS_DIR`, and `STATS_DIR`.
- LLM judge only: `API_KEY`, `BASE_URL`, and `MODEL`.
- Repository inputs: selected Python entry points, `hparams/CurpEdit/<model>.yaml`, and the data file selected by `data_type`.
- Evaluation-only inputs: each edited model directory.

Keep generated scripts standalone. Do not source `scripts/run_0.sh` or any shared scheduler.

## Validation

Perform all applicable checks:

```bash
bash -n scripts/experiments/YYYYMMDD_<name>.sh
python run_crispedit.py --help
python run_base_benchmarks.py --help
python run_edited_benchmarks.py --help
```

Also verify:

- The plan and script paths reference each other exactly.
- Every matrix row has the expected selected-stage commands.
- No angle-bracket template prompt or `__UPPER_SNAKE_CASE__` marker remains.
- Evaluation commands use the verified `run_id`.
- No training or evaluation process was started.

Report unavailable validation commands honestly. Never launch the generated script.
