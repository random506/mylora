# Repository Guidelines

## Project Structure & Module Organization

Core library code lives in `easyeditor/`. Add algorithms under `easyeditor/models/` or project variants under `easyeditor/mymodels/`; datasets, evaluation, and training utilities belong in their matching subdirectories. Root scripts such as `run_crispedit.py` and `run_benchmarks.py` are CLI entry points. Store settings in `hparams/<Algorithm>/<model>.yaml`. `toy_experiment/` holds isolated experiments, while `skills/` contains agent workflows and focused tests. Treat `logs/` and timestamped `docs/experiments/` files as generated artifacts.

## Build, Test, and Development Commands

Create and activate a Python environment before installing dependencies:

```bash
python -m pip install -r requirements.txt
python run_crispedit.py --help
python run_benchmarks.py --help
python skills/analyze-results/collect_runs.py
```

The install command sets up the pinned ML stack. The `--help` commands inspect options without launching experiments. The collector aggregates `logs/` into Markdown. Full runs require configured model/data caches and usually CUDA GPUs.

## Coding Style & Naming Conventions

Use four-space indentation and UTF-8. Use `snake_case` for functions, variables, and modules; `PascalCase` for classes; and `UPPER_SNAKE_CASE` for constants. Type new reusable helpers and document non-obvious behavior. Keep CLI parsing in entry scripts and reusable logic inside `easyeditor/`. Name hyperparameter files after the model, for example `hparams/CrispEdit/llama3-8b.yaml`. Run `ruff check <changed-files>` when available.

## Testing Guidelines

Tests use the standard-library `unittest` framework and follow `test_*.py` naming. Run the current focused suite with:

```bash
python -m unittest skills/analyze-results/test_collect_runs.py -v
python -m py_compile path/to/changed_file.py
```

Add deterministic CPU tests for parsing, aggregation, and utilities. For GPU changes, document the command, model, dataset, seed, and devices used. Do not claim unrun benchmarks as verification. No coverage threshold is enforced.

## Commit & Pull Request Guidelines

Recent commits use short, imperative subjects such as `Fix the data loading function` and `Add Qwen configuration file`. Keep commits focused and separate generated outputs from implementation. Pull requests should explain the change, list affected hyperparameters, include validation commands and metric deltas, and link relevant issues. Do not commit secrets, cache paths, large checkpoints, or raw `logs/` unless required for review.

## Configuration & Secrets

Copy `.env.example` for local configuration. Keep API tokens, tracking credentials, and machine-specific Hugging Face cache paths out of version control. Verify YAML changes against the corresponding hyperparameter dataclass before launching costly experiments.
