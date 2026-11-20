"""B2 summary aggregation tests: no model fits or historical artifact writes."""
from __future__ import annotations

import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def summarizer():
    spec = importlib.util.spec_from_file_location(
        "b2_summary_under_test", ROOT / "scripts/76_summarize_b2_full_fair.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class B2SummaryTests(unittest.TestCase):
    def populate(self, root, *, bad_metric=False):
        metrics = {
            "accuracy": 0.6, "balanced_accuracy": 0.59,
            "roc_auc": 0.65, "log_loss": 0.64,
            "brier": 0.23, "ece_15": 0.04, "mce_15": 0.08,
        }
        for seed in (42, 43, 44):
            for model in ("rf", "xgb", "deep"):
                if model == "deep":
                    path = root / f"deep_finalist/real_v3_1_fixed/seed_{seed}/metrics_summary.json"
                    report = {
                        "seed": seed, "model": "deep_final",
                        "n_features": 64, "n_rows_test": 20,
                        "best_candidate": "chosen_on_validation",
                        "test_metrics": {
                            key: (value + (seed - 42) / 100 if key == "accuracy" else value)
                            for key, value in metrics.items()
                            if key not in ("brier", "ece_15", "mce_15")
                        },
                    }
                else:
                    path = root / f"strict_calibration/real_v3_1_fixed/{model}/seed_{seed}/metrics_summary.json"
                    report = {
                        "seed": seed, "model": model, "n_features": 64,
                        "partition_rows": {"test": 20},
                        "scores": {
                            method: {key: (value + (seed - 42) / 100
                                           if key == "accuracy" else value)
                                     for key, value in metrics.items()}
                            for method in ("uncalibrated", "platt", "isotonic")
                        },
                    }
                    if bad_metric and model == "xgb" and seed == 44:
                        report["scores"]["platt"]["log_loss"] = float("nan")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(report), encoding="utf-8")

    def test_collect_aggregate_and_markdown(self):
        summarizer_module = summarizer()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.populate(root)
            rows, inputs = summarizer_module.collect(root)
            self.assertEqual(len(rows), 21)
            self.assertEqual(len(inputs), 9)
            groups = summarizer_module.aggregate(rows)
            self.assertEqual(len(groups), 7)
            rf = next(group for group in groups if group["model"] == "rf"
                      and group["variant"] == "uncalibrated")
            self.assertAlmostEqual(rf["accuracy_mean"], 0.61)
            self.assertAlmostEqual(rf["accuracy_sd"], 0.01)
            deep = next(group for group in groups if group["model"] == "deep")
            self.assertIsNone(deep["brier_mean"])
            self.assertIsNone(deep["ece_15_sd"])
            report = summarizer_module.render_markdown(groups)
            self.assertIn("historically", report)
            self.assertIn("0.610000 ± 0.010000", report)
            self.assertIn("not a confidence interval", report)

    def test_missing_run_and_nonfinite_metric_fail_closed(self):
        summarizer_module = summarizer()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.populate(root, bad_metric=True)
            with self.assertRaisesRegex(ValueError, "nonfinite log_loss"):
                summarizer_module.collect(root)
            path = root / "strict_calibration/real_v3_1_fixed/xgb/seed_44/metrics_summary.json"
            path.unlink()
            with self.assertRaisesRegex(ValueError, "Missing verified B2 report"):
                summarizer_module.collect(root)


if __name__ == "__main__":
    unittest.main()
