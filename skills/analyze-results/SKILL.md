---
name: analyze-results
description: Aggregate and compare this mylora project's completed or partial experiment results from logs/, including capability.json and LLM-judge mean_metrics.json artifacts. Use when Codex needs to organize experiments, compare capability and editing accuracy, identify missing result artifacts, filter run names, or generate docs/experiments/*_results_analysis.md.
---

# Analyze Experiment Results

## Workflow

1. Locate the repository root and use `logs/` as the default results root.
2. Treat a base experiment directory and its two evaluation directories as one experiment:
   - `<run>/capability.json`
   - `<run>_eval_llm_judge_no_context/mean_metrics.json`
   - `<run>_eval_llm_judge_qa_inst/mean_metrics.json`
3. Run the bundled collector from the repository root:

```bash
python skills/analyze-results/collect_runs.py
```

4. Use `--run <base-run-name>` for exact selections or `--run-pattern '<glob>'` for pattern selections. Repeat either option to select multiple experiments.
5. Read the generated report path printed by the collector and inspect the single comparison table plus any missing-artifact section.
6. Tell the user where the report was written and summarize objective comparisons. Do not declare one overall best experiment because the report intentionally keeps capability, no-context editing, and QA-instruction editing separate.

## Metric Contract

Read only the two canonical artifact types above. Ignore `results.json`, `results_pending_judge.json`, and other raw files.

Extract these capability metrics:

- IFEval: `results.ifeval["prompt_level_strict_acc,none"]`
- TruthfulQA MC2: `results.truthfulqa_mc2["acc,none"]`
- MMLU: `results.mmlu["acc,none"]`
- GSM8K CoT: `results.gsm8k_cot["exact_match,flexible-extract"]`
- ARC Challenge: `results.arc_challenge["acc_norm,none"]`

Calculate `Capability Mean` only when all five values exist. For each evaluation context, extract only `post.rewrite_acc` and `post.rephrase_acc`, then calculate that context's mean only when both values exist. Never calculate a partial mean.

## Report Contract

- Write a new report to `docs/experiments/YYYYMMDD_HHMMSS_results_analysis.md` using Asia/Shanghai time.
- Refuse to overwrite an existing report.
- Render one wide Markdown table with one row per base experiment.
- Show scores as percentages with two decimal places.
- Sort rows by the complete base run name.
- Bold every tied maximum in `Capability Mean`, `No Context Mean`, and `QA Inst Mean`.
- Include incomplete and evaluation-only experiments with `N/A` values and list their missing or invalid canonical artifacts.
- Do not parse model names, methods, hyperparameters, learning rates, or optimizers from directory names.
- Do not calculate a combined score across capability and editing contexts.

## Rules

- Never fabricate, estimate, or infer a missing metric from a raw result file.
- Treat `capability.json` and `mean_metrics.json` as the only authoritative inputs.
- Preserve raw floating-point precision during calculation; round only for Markdown display.
- Prefer the default full comparison unless the user names a specific experiment scope.
