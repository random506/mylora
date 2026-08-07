#!/usr/bin/env python3
"""Aggregate mylora experiment logs into one Markdown comparison report."""

from __future__ import annotations

import argparse
import fnmatch
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable


SHANGHAI_TZ = timezone(timedelta(hours=8), name="Asia/Shanghai")

EVAL_SUFFIXES = {
    "no_context": "_eval_llm_judge_no_context",
    "qa_inst": "_eval_llm_judge_qa_inst",
}

CAPABILITY_METRICS = {
    "ifeval": ("results", "ifeval", "prompt_level_strict_acc,none"),
    "truthfulqa_mc2": ("results", "truthfulqa_mc2", "acc,none"),
    "mmlu": ("results", "mmlu", "acc,none"),
    "gsm8k_cot": ("results", "gsm8k_cot", "exact_match,flexible-extract"),
    "arc_challenge": ("results", "arc_challenge", "acc_norm,none"),
}

EDIT_METRICS = {
    "rewrite_acc": ("post", "rewrite_acc"),
    "rephrase_acc": ("post", "rephrase_acc"),
}

REPORT_COLUMNS = (
    ("Run", "run"),
    ("IFEval", "ifeval"),
    ("TruthfulQA MC2", "truthfulqa_mc2"),
    ("MMLU", "mmlu"),
    ("GSM8K CoT", "gsm8k_cot"),
    ("ARC Challenge", "arc_challenge"),
    ("Capability Mean", "capability_mean"),
    ("No Context Rewrite", "no_context.rewrite_acc"),
    ("No Context Rephrase", "no_context.rephrase_acc"),
    ("No Context Mean", "no_context.mean"),
    ("QA Inst Rewrite", "qa_inst.rewrite_acc"),
    ("QA Inst Rephrase", "qa_inst.rephrase_acc"),
    ("QA Inst Mean", "qa_inst.mean"),
    ("Status", "status"),
)

BEST_COLUMNS = (
    "capability_mean",
    "no_context.mean",
    "qa_inst.mean",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Group mylora logs by base experiment and write one Markdown "
            "comparison table."
        )
    )
    parser.add_argument(
        "--results-root",
        default="logs",
        help="Repo-relative or absolute directory containing experiment logs.",
    )
    parser.add_argument(
        "--output",
        help=(
            "Optional report path. Defaults to "
            "docs/experiments/YYYYMMDD_HHMMSS_results_analysis.md."
        ),
    )
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="Exact base experiment name to include. Repeat as needed.",
    )
    parser.add_argument(
        "--run-pattern",
        action="append",
        default=[],
        help="Glob matched against base experiment names. Repeat as needed.",
    )
    return parser.parse_args()


def find_repo_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    return current


def resolve_from_repo(repo_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def default_output_path(repo_root: Path, generated_at: datetime) -> Path:
    timestamp = generated_at.strftime("%Y%m%d_%H%M%S")
    return repo_root / "docs" / "experiments" / f"{timestamp}_results_analysis.md"


def load_json_object(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except FileNotFoundError:
        return None, "file is missing"
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON: {exc}"
    except OSError as exc:
        return None, f"could not read file: {exc}"

    if not isinstance(value, dict):
        return None, "top-level JSON value is not an object"
    return value, None


def get_numeric_metric(
    data: dict[str, Any], path: tuple[str, ...]
) -> tuple[float | None, str | None]:
    value: Any = data
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None, f"missing metric {'.'.join(path)}"
        value = value[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None, f"metric {'.'.join(path)} is not numeric"
    return float(value), None


def complete_mean(values: Iterable[float | None]) -> float | None:
    items = list(values)
    if not items or any(value is None for value in items):
        return None
    return sum(value for value in items if value is not None) / len(items)


def base_name_for_directory(directory_name: str) -> tuple[str, str | None]:
    for context, suffix in EVAL_SUFFIXES.items():
        if directory_name.endswith(suffix):
            return directory_name[: -len(suffix)], context
    return directory_name, None


def discover_experiment_names(results_root: Path) -> set[str]:
    names: set[str] = set()
    if not results_root.is_dir():
        return names

    for child in results_root.iterdir():
        if not child.is_dir():
            continue
        base_name, context = base_name_for_directory(child.name)
        if context is not None or (child / "capability.json").is_file():
            names.add(base_name)
    return names


def select_experiment_names(
    discovered: set[str], exact_runs: list[str], patterns: list[str]
) -> list[str]:
    if not exact_runs and not patterns:
        return sorted(discovered)

    selected = set(exact_runs)
    for name in discovered:
        if any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns):
            selected.add(name)
    return sorted(selected)


def collect_artifact_metrics(
    path: Path,
    metric_paths: dict[str, tuple[str, ...]],
    issues: list[str],
    missing_artifacts: list[str],
) -> dict[str, float | None]:
    values = {name: None for name in metric_paths}
    data, load_error = load_json_object(path)
    if load_error:
        issue = f"{path}: {load_error}"
        issues.append(issue)
        missing_artifacts.append(issue)
        return values

    assert data is not None
    for name, metric_path in metric_paths.items():
        value, metric_error = get_numeric_metric(data, metric_path)
        values[name] = value
        if metric_error:
            issue = f"{path}: {metric_error}"
            issues.append(issue)
            missing_artifacts.append(issue)
    return values


def collect_experiment(results_root: Path, run_name: str) -> dict[str, Any]:
    issues: list[str] = []
    missing_artifacts: list[str] = []

    capability_path = results_root / run_name / "capability.json"
    capability = collect_artifact_metrics(
        capability_path,
        CAPABILITY_METRICS,
        issues,
        missing_artifacts,
    )

    metrics: dict[str, float | None] = dict(capability)
    metrics["capability_mean"] = complete_mean(capability.values())

    for context, suffix in EVAL_SUFFIXES.items():
        metrics_path = results_root / f"{run_name}{suffix}" / "mean_metrics.json"
        context_metrics = collect_artifact_metrics(
            metrics_path,
            EDIT_METRICS,
            issues,
            missing_artifacts,
        )
        for metric_name, value in context_metrics.items():
            metrics[f"{context}.{metric_name}"] = value
        metrics[f"{context}.mean"] = complete_mean(context_metrics.values())

    if not issues:
        status = "complete"
    elif missing_artifacts:
        status = "missing"
    else:
        status = "invalid"

    return {
        "run": run_name,
        **metrics,
        "status": status,
        "issues": issues,
        "missing_artifacts": missing_artifacts,
    }


def best_values(rows: list[dict[str, Any]]) -> dict[str, float]:
    best: dict[str, float] = {}
    for key in BEST_COLUMNS:
        values = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
        if values:
            best[key] = max(values)
    return best


def escape_markdown_cell(value: str) -> str:
    return value.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def format_percent(value: float | None, *, bold: bool = False) -> str:
    if value is None:
        return "N/A"
    rendered = f"{value * 100:.2f}%"
    return f"**{rendered}**" if bold else rendered


def render_cell(
    row: dict[str, Any], key: str, maxima: dict[str, float]
) -> str:
    value = row.get(key)
    if key in {"run", "status"}:
        return escape_markdown_cell(str(value))
    is_best = key in maxima and value == maxima[key]
    return format_percent(value, bold=is_best)


def render_report(
    rows: list[dict[str, Any]], results_root: Path, generated_at: datetime
) -> str:
    maxima = best_values(rows)
    lines = [
        "# Experiment Results Analysis",
        "",
        f"- Generated at: `{generated_at.strftime('%Y-%m-%d %H:%M:%S %z')}`",
        f"- Results root: `{results_root}`",
        f"- Experiment count: `{len(rows)}`",
        "",
        "| " + " | ".join(label for label, _ in REPORT_COLUMNS) + " |",
        "| " + " | ".join("---" for _ in REPORT_COLUMNS) + " |",
    ]

    for row in rows:
        cells = [render_cell(row, key, maxima) for _, key in REPORT_COLUMNS]
        lines.append("| " + " | ".join(cells) + " |")

    missing_rows = [row for row in rows if row["missing_artifacts"]]
    if missing_rows:
        lines.extend(["", "## Missing Artifacts", ""])
        for row in missing_rows:
            for issue in row["missing_artifacts"]:
                lines.append(f"- `{escape_markdown_cell(row['run'])}`: {issue}")

    lines.append("")
    return "\n".join(lines)


def write_new_report(path: Path, report: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(report)


def main() -> int:
    args = parse_args()
    repo_root = find_repo_root(Path(__file__).resolve().parent)
    results_root = resolve_from_repo(repo_root, args.results_root)
    generated_at = datetime.now(SHANGHAI_TZ)
    output_path = (
        resolve_from_repo(repo_root, args.output)
        if args.output
        else default_output_path(repo_root, generated_at)
    )

    discovered = discover_experiment_names(results_root)
    selected = select_experiment_names(discovered, args.run, args.run_pattern)
    rows = [collect_experiment(results_root, run_name) for run_name in selected]
    report = render_report(rows, results_root, generated_at)

    try:
        write_new_report(output_path, report)
    except FileExistsError:
        print(f"Refusing to overwrite existing report: {output_path}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"Could not write report {output_path}: {exc}", file=sys.stderr)
        return 1

    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
