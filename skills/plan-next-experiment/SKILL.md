---
name: plan-next-experiment
description: "Use when the next experiment needs to be chosen, documented, or turned into a runnable experiment script."
---

# Plan Next Experiment

## Workflow

1. Read `docs/experiments/AGENTS.md`, `TEMPLATE.md`, `overview.md`, recent experiment docs, and relevant existing scripts/configs.
2. Choose a clear paper-facing question: changed variables, controls, sweep dimensions, and the rule for selecting the best run.
3. Create `docs/experiments/YY-MM-DD-name.md` from the template.
4. Create `scripts/experiments/YY-MM-DD-name.sh` by adapting existing command patterns.
5. Verify the doc and script reference each other correctly.

## Rules

- Keep benchmark metrics, table shapes, and run-specific details in the experiment doc, not this skill.
- Use **only** the sections in `TEMPLATE.md`.
- `Motivation` is one short paragraph stating the question.
- `Notes` is short, factual, operational only (task names, prerequisites, reuse anchors, code-change requirements).
- Reuse repo script structure; factor repeated boilerplate into utilities instead of growing prompt text.
