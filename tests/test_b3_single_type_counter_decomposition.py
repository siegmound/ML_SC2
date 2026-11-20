"""Stdlib-only tests for conditional single-type counter decomposition."""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/84_decompose_b3_single_type_counter.py"


def load():
    spec = importlib.util.spec_from_file_location("b3_counter_single_type", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class SingleTypeCounterTests(unittest.TestCase):
    def setUp(self):
        self.a = load()
        self.stats = {
            "Queen": {"type": "army", "power": 8},
            "Baneling": {"type": "army", "power": 2},
            "Marine": {"type": "army", "power": 2},
        }
        self.counters = {"Queen": [], "Baneling": ["Queen"], "Marine": []}

    def test_queen_only_implied_weight_consistent_across_enemy_compositions(self):
        frozen_weight, archived_weight = .745, .85
        inventories = {}
        evidence = []
        for sec, enemy in (
            (210, {"Baneling": 3, "Marine": 3}),
            (360, {"Baneling": 6, "Marine": 1}),
        ):
            fraction = enemy["Baneling"] / sum(enemy.values())
            inventories[("r.SC2Replay", sec)] = {
                "p1": {"army": {"Queen": 4}, "upgrades": set()},
                "p2": {"army": enemy, "upgrades": set()},
            }
            evidence.append({
                "replay_id": "r.SC2Replay", "time_sec": str(sec),
                "player": "p1", "feature": "counter_advantage_p1",
                "kind": "counter_advantage",
                "frozen": str(frozen_weight*fraction),
                "counter_relation_reversed": str(archived_weight*fraction),
                "as_archived": "0",
            })
            # Two types in the opposing army: no single unit weight
            # should be inferred for the opposing player.
            evidence.append({
                "replay_id": "r.SC2Replay", "time_sec": str(sec),
                "player": "p2", "feature": "counter_advantage_p2",
                "kind": "counter_advantage",
                "frozen": "0", "counter_relation_reversed": "0",
                "as_archived": "0",
            })
        report, rows = self.a.single_type_evidence(
            evidence, inventories, self.stats, self.counters
        )
        self.assertEqual(report["counter_rows"], 4)
        self.assertEqual(report["non_single_type"], 2)
        self.assertEqual(report["single_type_nonzero_counter_fraction"], 2)
        self.assertEqual(report["single_type_mismatches"], 2)
        self.assertEqual(report["queen_only"]["n_distinct_enemy_army_compositions"], 2)
        self.assertEqual(report["queen_only"]["n_mismatches"], 2)
        self.assertAlmostEqual(report["queen_only"]["archived_unit_weight"], .85)
        self.assertAlmostEqual(report["queen_only"]["implied_weight_median"], .745)
        self.assertAlmostEqual(report["queen_only"]["implied_weight_min"], .745)
        self.assertAlmostEqual(
            report["queen_only"]["frozen_to_reverse_ratio_median"], .745/.85
        )
        self.assertAlmostEqual(rows[0]["enemy_countered_fraction"], .5)
        self.assertAlmostEqual(rows[0]["frozen_implied_effective_unit_weight"], .745)

    def test_rejects_nonreproducible_reversed_predictions(self):
        inventories = {("r.SC2Replay", 210): {
            "p1": {"army": {"Queen": 2}, "upgrades": set()},
            "p2": {"army": {"Baneling": 2}, "upgrades": set()},
        }}
        row = {
            "replay_id": "r.SC2Replay", "time_sec": "210",
            "player": "p1", "feature": "counter_advantage_p1",
            "kind": "counter_advantage", "frozen": ".745",
            "counter_relation_reversed": "999", "as_archived": "0",
        }
        with self.assertRaisesRegex(ValueError, "not reproducible"):
            self.a.single_type_evidence([row], inventories, self.stats, self.counters)

    def test_no_counter_fraction_does_not_invent_effective_weight(self):
        inventories = {("r.SC2Replay", 150): {
            "p1": {"army": {"Queen": 2}, "upgrades": set()},
            "p2": {"army": {"Marine": 2}, "upgrades": set()},
        }}
        row = {
            "replay_id": "r.SC2Replay", "time_sec": "150",
            "player": "p1", "feature": "counter_advantage_p1",
            "kind": "counter_advantage", "frozen": ".2",
            "counter_relation_reversed": "0", "as_archived": "0",
        }
        report, rows = self.a.single_type_evidence(
            [row], inventories, self.stats, self.counters
        )
        self.assertEqual(report["zero_counter_fraction"], 1)
        self.assertEqual(report["single_type_nonzero_counter_fraction"], 0)
        self.assertEqual(rows, [])

    def test_refuse_duplicate_snapshot_and_bad_feature_mapping(self):
        inventories = {("r.SC2Replay", 150): {
            "p1": {"army": {"Queen": 1}, "upgrades": set()},
            "p2": {"army": {"Baneling": 2}, "upgrades": set()},
        }}
        row = {
            "replay_id": "r.SC2Replay", "time_sec": "150",
            "player": "p1", "feature": "counter_advantage_p1",
            "kind": "counter_advantage", "frozen": ".745",
            "counter_relation_reversed": ".85", "as_archived": "0",
        }
        with self.assertRaisesRegex(ValueError, "Repeated counter observation"):
            self.a.single_type_evidence([row, row], inventories, self.stats, self.counters)
        wrong = {**row, "feature": "counter_advantage_p2"}
        with self.assertRaisesRegex(ValueError, "wrong player"):
            self.a.single_type_evidence([wrong], inventories, self.stats, self.counters)


if __name__ == "__main__":
    unittest.main()
