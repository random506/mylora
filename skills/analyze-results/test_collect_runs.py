import importlib.util
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("collect_runs.py")
SPEC = importlib.util.spec_from_file_location("collect_runs", MODULE_PATH)
collect_runs = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(collect_runs)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def capability(values: list[float]) -> dict:
    return {
        "results": {
            "ifeval": {"prompt_level_strict_acc,none": values[0]},
            "truthfulqa_mc2": {"acc,none": values[1]},
            "mmlu": {"acc,none": values[2]},
            "gsm8k_cot": {"exact_match,flexible-extract": values[3]},
            "arc_challenge": {"acc_norm,none": values[4]},
        }
    }


def edit_metrics(rewrite: float, rephrase: float) -> dict:
    return {"post": {"rewrite_acc": rewrite, "rephrase_acc": rephrase}}


class CollectRunsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.logs = Path(self.temporary_directory.name) / "logs"
        self.logs.mkdir()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def add_complete_run(
        self,
        name: str,
        capability_values: list[float],
        no_context: tuple[float, float],
        qa_inst: tuple[float, float],
    ) -> None:
        write_json(self.logs / name / "capability.json", capability(capability_values))
        write_json(
            self.logs
            / f"{name}{collect_runs.EVAL_SUFFIXES['no_context']}"
            / "mean_metrics.json",
            edit_metrics(*no_context),
        )
        write_json(
            self.logs
            / f"{name}{collect_runs.EVAL_SUFFIXES['qa_inst']}"
            / "mean_metrics.json",
            edit_metrics(*qa_inst),
        )

    def test_groups_artifacts_and_calculates_complete_means(self) -> None:
        self.add_complete_run(
            "run-a",
            [0.1, 0.2, 0.3, 0.4, 0.5],
            (0.6, 0.8),
            (0.7, 0.9),
        )

        names = collect_runs.discover_experiment_names(self.logs)
        row = collect_runs.collect_experiment(self.logs, "run-a")

        self.assertEqual(names, {"run-a"})
        self.assertAlmostEqual(row["capability_mean"], 0.3)
        self.assertAlmostEqual(row["no_context.mean"], 0.7)
        self.assertAlmostEqual(row["qa_inst.mean"], 0.8)
        self.assertEqual(row["status"], "complete")

    def test_keeps_orphan_evaluation_and_never_calculates_partial_mean(self) -> None:
        name = "orphan"
        write_json(
            self.logs
            / f"{name}{collect_runs.EVAL_SUFFIXES['qa_inst']}"
            / "mean_metrics.json",
            {"post": {"rewrite_acc": 0.5}},
        )

        names = collect_runs.discover_experiment_names(self.logs)
        row = collect_runs.collect_experiment(self.logs, name)

        self.assertEqual(names, {name})
        self.assertIsNone(row["capability_mean"])
        self.assertIsNone(row["qa_inst.mean"])
        self.assertEqual(row["status"], "missing")
        self.assertTrue(any("capability.json" in item for item in row["missing_artifacts"]))

    def test_exact_and_pattern_selection_are_unioned_and_sorted(self) -> None:
        discovered = {"beta", "alpha", "gamma"}
        selected = collect_runs.select_experiment_names(
            discovered, ["missing"], ["a*"]
        )
        self.assertEqual(selected, ["alpha", "missing"])

    def test_report_is_one_table_and_bolds_all_tied_group_maxima(self) -> None:
        self.add_complete_run("run-a", [0.5] * 5, (0.2, 0.4), (0.8, 0.8))
        self.add_complete_run("run-b", [0.5] * 5, (0.7, 0.9), (0.4, 0.6))
        rows = [
            collect_runs.collect_experiment(self.logs, name)
            for name in ("run-a", "run-b")
        ]

        report = collect_runs.render_report(
            rows,
            self.logs,
            datetime(2026, 8, 7, 12, 34, tzinfo=collect_runs.SHANGHAI_TZ),
        )

        self.assertEqual(report.count("| Run |"), 1)
        self.assertEqual(report.count("**50.00%**"), 2)
        self.assertIn("**80.00%**", report)
        self.assertLess(report.index("| run-a |"), report.index("| run-b |"))


if __name__ == "__main__":
    unittest.main()
