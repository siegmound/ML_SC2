"""Stdlib-only tests for conservative B3 residual descriptive profiling."""
from __future__ import annotations

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/83_profile_b3_counter_residuals.py"


def auditor():
    spec = importlib.util.spec_from_file_location("b3_counter_residual", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CounterResidualProfileTests(unittest.TestCase):
    def setUp(self):
        self.tool = auditor()

    def make_evidence(self, home):
        p = Path(home)
        inventory = p / "inventory.csv"
        with inventory.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "replay_id", "time_sec", "player", "army_names_and_counts",
                "upgrade_events_completed", "raw_base_power_before_counters_upgrades",
            ])
            writer.writeheader()
            for who in ("p1", "p2"):
                writer.writerow({
                    "replay_id": "match.SC2Replay", "time_sec": "135", "player": who,
                    "army_names_and_counts": "Marine:2",
                    "upgrade_events_completed": "TerranInfantryWeaponsLevel1",
                    "raw_base_power_before_counters_upgrades": ".04",
                })
        hypothesis = p / "hypothesis.csv"
        with hypothesis.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "replay_id", "time_sec", "player", "kind", "feature", "frozen",
                "as_archived", "counter_relation_reversed",
                "archived_formula_reproduced_from_captured_inventory",
            ])
            writer.writeheader()
            for player, kind, feature, frozen, archived, reverse, included in [
                ("p1", "counter_advantage", "counter_advantage_p1", ".5", ".1", ".48", "True"),
                ("p2", "counter_advantage", "counter_advantage_p2", "0", ".2", "0", "True"),
                ("p1", "combat", "p1_combat", ".4", ".39", ".35", "True"),
                ("p2", "combat", "p2_combat", ".1", ".1", ".1", "False"),
            ]:
                writer.writerow({
                    "replay_id": "match.SC2Replay", "time_sec": "135",
                    "player": player, "kind": kind, "feature": feature,
                    "frozen": frozen, "as_archived": archived,
                    "counter_relation_reversed": reverse,
                    "archived_formula_reproduced_from_captured_inventory": included,
                })
        return hypothesis, inventory

    def test_descriptive_counter_profile_and_exclusion(self):
        with tempfile.TemporaryDirectory() as folder:
            data, source = self.make_evidence(folder)
            report, worst = self.tool.profile(data, source, expected_count=2)
            self.assertEqual(report["counter_summary"]["n"], 2)
            self.assertEqual(report["counter_summary"]["mismatches"], 1)
            self.assertEqual(report["counter_summary"]["mismatch_frozen_gt_reversed"], 1)
            self.assertEqual(report["counter_summary"]["mismatch_frozen_lt_reversed"], 0)
            self.assertEqual(report["counter_summary"]["mismatch_category_counts"],
                             {"both_nonzero_same_sign": 1})
            self.assertAlmostEqual(report["counter_summary"]["nonzero_frozen_to_reversed_ratios"]["median"],
                                   .5/.48)
            self.assertEqual(report["combat_included_summary"]["n"], 1)
            self.assertEqual(report["combat_excluded_incomplete_evidence"], 1)
            self.assertEqual(len(worst), 1)
            self.assertEqual(worst[0]["player"], "p1")

    def test_duplicate_hypothesis_or_missing_inventory_fails_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            data, source = self.make_evidence(folder)
            original = data.read_text()
            self.assertRaisesRegex(
                ValueError, "Required input file absent",
                self.tool.profile, data, Path(folder) / "unavailable.csv",
                expected_count=2,
            )
            data.write_text(original + original.splitlines()[-1] + "\n")
            with self.assertRaisesRegex(ValueError, "Duplicate hypothesis comparison"):
                self.tool.profile(data, source, expected_count=2)

    def test_zero_categories_sign_and_percentiles(self):
        t = self.tool
        self.assertEqual(t.counter_category(0, 0), "both_zero")
        self.assertEqual(t.counter_category(0, .2), "frozen_zero_predicted_nonzero")
        self.assertEqual(t.counter_category(.2, 0), "predicted_zero_frozen_nonzero")
        self.assertEqual(t.counter_category(-.2, .2), "opposite_sign")
        self.assertEqual(t.counter_category(.2, .3), "both_nonzero_same_sign")
        x = t.numeric_stats([1, 2, 3, 4, 5])
        self.assertEqual(x["median"], 3)
        self.assertEqual(x["p10"], 1.4)
        self.assertEqual(x["p90"], 4.6)
        with self.assertRaisesRegex(ValueError, "Invalid summary values"):
            t.numeric_stats([])


if __name__ == "__main__":
    unittest.main()
