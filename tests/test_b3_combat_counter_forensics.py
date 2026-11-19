"""Synthetic regression tests for the B3 read-only combat/counter forensics."""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def auditor():
    spec = importlib.util.spec_from_file_location(
        "b3_combat_forensics", ROOT / "scripts/81_forensic_b3_combat_counter.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class CombatCounterForensicsTests(unittest.TestCase):
    def test_player_inventory_evidence_does_not_guess_tech_or_upgrades(self):
        mod = auditor()
        stats = {
            "Marine": {"type": "army", "power": 2, "supply": 1},
            "Marauder": {"type": "army", "power": 5, "supply": 2},
            "SCV": {"type": "worker", "power": 1, "supply": 1},
        }
        inventory = {1: {"Marine": 10, "Marauder": 2, "SCV": 12}, 2: {}}
        unit_counters = {"Marine": ["Baneling", "Hellion"], "Marauder": ["Immortal"]}
        upgrade = {1: {"TerranInfantryWeaponsLevel1"}}
        result = mod.summarize_player(inventory, 1, stats, unit_counters, upgrade)
        self.assertEqual(result["army_units"], 12)
        self.assertEqual(result["army_types"], 2)
        self.assertAlmostEqual(result["raw_base_power_before_counters_upgrades"], 0.30)
        self.assertEqual(result["units_missing_tech_definition"], 12)
        self.assertEqual(result["units_missing_upgrade_category"], 12)
        self.assertEqual(result["counter_rules_for_army_types"], 3)
        self.assertEqual(result["upgrade_events_completed"], "TerranInfantryWeaponsLevel1")
        self.assertEqual(result["army_names_and_counts"], "Marauder:2;Marine:10")

    def test_numeric_signed_residuals_and_identity_checks(self):
        mod = auditor()
        data = {
            "replay_id": ["match.SC2Replay", "match.SC2Replay"],
            "time_sec": [135, 150], "p1_wins": [1, 1],
            "p1_combat": [0.2, 0.4], "p2_combat": [0.1, 0.3],
            "counter_advantage_p1": [0.5, 0.6],
            "counter_advantage_p2": [0.7, 0.8],
            "p1_tech_level": [2, 2], "p2_tech_level": [1, 1],
        }
        frozen = pd.DataFrame(data)
        generated = frozen.copy()
        generated.loc[1, "p1_combat"] = 0.25
        generated["p1_tech_level"] = [0, 0]
        residuals, summary = mod.compare_primitives(frozen, generated)
        self.assertEqual(summary["comparisons"], 12)
        self.assertEqual(summary["mismatches"], 3)
        changed = [r for r in residuals if r["mismatch"]]
        self.assertEqual(len(changed), 3)
        self.assertAlmostEqual(
            [x for x in changed if x["feature"] == "p1_combat"][0]
            ["signed_frozen_minus_reconstructed"], 0.15,
        )
        self.assertEqual(
            next(row for row in summary["by_primitive"] if row["feature"] == "p1_tech_level")
            ["mismatches"], 2
        )
        wrong = generated[generated.time_sec != 135]
        with self.assertRaisesRegex(ValueError, "Replay/time keys differ"):
            mod.compare_primitives(frozen, wrong)
        dup = pd.concat([generated, generated.iloc[:1]])
        with self.assertRaisesRegex(ValueError, "duplicate replay/time"):
            mod.compare_primitives(frozen, dup)

    def test_missing_fields_and_label_mismatch_fail_closed(self):
        mod = auditor()
        row = {
            "replay_id": "r", "time_sec": 135, "p1_wins": 1,
            "p1_combat": 0, "p2_combat": 0,
            "counter_advantage_p1": 0, "counter_advantage_p2": 0,
            "p1_tech_level": 0, "p2_tech_level": 0,
        }
        frozen = pd.DataFrame([row])
        missing = frozen.drop(columns=["p1_combat"])
        with self.assertRaisesRegex(ValueError, "Missing primitive"):
            mod.compare_primitives(frozen, missing)
        other = frozen.copy()
        other.loc[0, "p1_wins"] = 0
        with self.assertRaisesRegex(ValueError, "winner differs"):
            mod.compare_primitives(frozen, other)


if __name__ == "__main__":
    unittest.main()
