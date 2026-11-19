"""B3 stdlib-only controlled counter-orientation hypothesis tests."""
from __future__ import annotations

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/82_test_b3_counter_orientation.py"


def auditor():
    spec = importlib.util.spec_from_file_location("b3_counter_orientation", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CounterOrientationTests(unittest.TestCase):
    def setUp(self):
        self.a = auditor()
        self.stats = {
            "Marine": {"type": "army", "power": 2},
            "Baneling": {"type": "army", "power": 2},
        }
        # The keyed unit is vulnerable to the listed units (Marine -> Baneling).
        self.counters = {"Marine": ["Baneling"], "Baneling": []}

    def test_counter_direction_is_not_semantically_symmetric(self):
        f = self.a.counter_score
        own, enemy = {"Marine": 2}, {"Baneling": 3}
        self.assertAlmostEqual(f(own, enemy, self.stats, self.counters,
                                 reverse=False), 0.775)
        self.assertAlmostEqual(f(own, enemy, self.stats, self.counters,
                                 reverse=True), 0.0)
        self.assertAlmostEqual(f(enemy, own, self.stats, self.counters,
                                 reverse=False), 0.0)
        self.assertAlmostEqual(f(enemy, own, self.stats, self.counters,
                                 reverse=True), 0.775)

    def test_combat_direction_changes_penalty_only(self):
        f = self.a.combat_score
        own, enemy = {"Marine": 2}, {"Baneling": 3}
        self.assertAlmostEqual(f(own, enemy, set(), self.stats, self.counters,
                                 {}, reverse=False), 0.034)
        self.assertAlmostEqual(f(own, enemy, set(), self.stats, self.counters,
                                 {}, reverse=True), 0.04)
        self.assertAlmostEqual(f(enemy, own, set(), self.stats, self.counters,
                                 {}, reverse=False), 0.06)
        self.assertAlmostEqual(f(enemy, own, set(), self.stats, self.counters,
                                 {}, reverse=True), 0.051)

    def test_evaluate_hypothesis_and_validate_original_formula(self):
        own, enemy = {"Marine": 2}, {"Baneling": 3}
        players = {
            "p1": {"army": own, "upgrades": set()},
            "p2": {"army": enemy, "upgrades": set()},
        }
        definitions = {
            "UNIT_STATS": self.stats, "UNIT_COUNTERS": self.counters,
            "UNIT_TECH_MULTIPLIERS": {},
        }
        evidence = {}
        for who, other in (("p1", "p2"), ("p2", "p1")):
            a = players[who]["army"]
            b = players[other]["army"]
            key = ("r.SC2Replay", 135)
            for kind, field in (
                ("combat", f"{who}_combat"),
                ("counter", f"counter_advantage_{who}"),
            ):
                if kind == "combat":
                    before = self.a.combat_score(a, b, set(), self.stats,
                                                 self.counters, {}, reverse=False)
                    reverse = self.a.combat_score(a, b, set(), self.stats,
                                                  self.counters, {}, reverse=True)
                else:
                    before = self.a.counter_score(a, b, self.stats,
                                                  self.counters, reverse=False)
                    reverse = self.a.counter_score(a, b, self.stats,
                                                   self.counters, reverse=True)
                evidence[(key[0], key[1], field)] = (reverse, before)
        result, rows = self.a.evaluate({("r.SC2Replay", 135): players},
                                       evidence, definitions)
        self.assertEqual(len(rows), 4)
        self.assertEqual(result["combat"]["as_archived"]["mismatches"], 2)
        self.assertEqual(result["combat"]["counter_relation_reversed"]["mismatches"], 0)
        self.assertEqual(result["counter_advantage"]["as_archived"]["mismatches"], 2)
        self.assertEqual(result["counter_advantage"]["counter_relation_reversed"]["mismatches"], 0)
        faulty = dict(evidence)
        faulty[("r.SC2Replay", 135, "p1_combat")] = (0.04, 999)
        with self.assertRaisesRegex(ValueError, "differs from script-81"):
            self.a.evaluate({("r.SC2Replay", 135): players}, faulty, definitions)

    def test_parse_inventories_and_reject_duplicate_player(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "players.csv"
            with path.open("w", newline="") as source:
                writer = csv.DictWriter(source, fieldnames=[
                    "replay_id", "time_sec", "player", "army_names_and_counts",
                    "raw_base_power_before_counters_upgrades", "upgrade_events_completed"
                ])
                writer.writeheader()
                writer.writerow({
                    "replay_id": "r.SC2Replay", "time_sec": "135",
                    "player": "p1", "army_names_and_counts": "Marine:2",
                    "raw_base_power_before_counters_upgrades": ".04",
                    "upgrade_events_completed": "",
                })
                writer.writerow({
                    "replay_id": "r.SC2Replay", "time_sec": "135",
                    "player": "p2", "army_names_and_counts": "Baneling:3",
                    "raw_base_power_before_counters_upgrades": ".06",
                    "upgrade_events_completed": "",
                })
            parsed = self.a.parse_inventory(path, self.stats)
            self.assertEqual(parsed[("r.SC2Replay", 135)]["p1"]["army"], {"Marine": 2})
            with path.open("a") as source:
                source.write("r.SC2Replay,135,p2,Baneling:3,.06,\n")
            with self.assertRaisesRegex(ValueError, "Missing/duplicate player"):
                self.a.parse_inventory(path, self.stats)


if __name__ == "__main__":
    unittest.main()
