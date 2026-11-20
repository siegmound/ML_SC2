"""v4 synthetic source-event and content-identity regression checks.

No historical frozen CSV, original test metrics or actual game replay
binaries are used to choose the logic or expected results here.
"""
from __future__ import annotations

import io
import json
import sys
import tempfile
import types
import unittest
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
try:
    import sc2reader  # type: ignore
except ImportError:
    sys.modules["sc2reader"] = types.ModuleType("sc2reader")

from sc2proj.bridge import ParseResult
from sc2proj.v4 import V4ReplayBridge, V4SnapshotParser, own_units_counter_enemy
from sc2proj.v4_dataset import build_dataset, validate_replay_frame


def event(name, second, **extras):
    return types.SimpleNamespace(name=name, second=second, **extras)


class SyntheticProcessor:
    def process_replay(self, path):
        name = Path(path).name
        y = int("first" in name or "clone" in name)
        values = pd.DataFrame({
            "replay_id": [name, name],
            "time_sec": [135.0, 150.0],
            "p1_wins": [y, y],
            "diff_workers": [1.0, 2.0],
        })
        return ParseResult(name, "ok", "", values, {"parse_status": "ok"})


class V4CleanRebuildTests(unittest.TestCase):
    def test_counter_direction_is_defined_before_training(self):
        self.assertTrue(own_units_counter_enemy("Baneling", "Marine"))
        self.assertFalse(own_units_counter_enemy("Marine", "Baneling"))
        parser = V4SnapshotParser()
        self.assertGreater(
            parser._counter_advantage({"Baneling": 3}, {"Marine": 6}), 0.0
        )
        self.assertEqual(
            parser._counter_advantage({"Marine": 6}, {"Baneling": 3}), 0.0
        )
        # A Baneling is a counter *against* Marine, so it penalizes
        # Marine's combat strength but not vice versa.
        self.assertLess(
            parser._combat_score({"Marine": 3}, {"Baneling": 2}, set()),
            parser._combat_score({"Marine": 3}, {}, set()),
        )

    def test_lifecycle_birth_init_dedup_morph_and_died(self):
        bridge = V4ReplayBridge()
        events = [
            event("UnitInitEvent", 1, unit_id=11, control_pid=1,
                  unit_type_name="Marine"),
            event("UnitBornEvent", 2, unit_id=11, control_pid=1,
                  unit_type_name="Marine"),
            event("UnitTypeChangeEvent", 18, unit_id=11, control_pid=1,
                  unit_type_name="Marauder"),
            event("UnitBornEvent", 20, unit_id=12, control_pid=2,
                  unit_type_name="Zergling"),
            event("UnitDiedEvent", 29, unit_id=11),
        ]
        inv, metadata, loss = bridge._inventories_until(events, 1, 2, 20)
        self.assertEqual(inv[1], {"Marauder": 1})
        self.assertEqual(inv[2], {"Zergling": 1})
        self.assertEqual(metadata["tracked_unique_unit_ids"], 2)
        inv, metadata, loss = bridge._inventories_until(events, 1, 2, 30)
        self.assertEqual(inv[1], {})
        self.assertEqual(inv[2], {"Zergling": 1})
        self.assertEqual(loss[1], 1)
        self.assertEqual(metadata["tracked_unique_unit_ids"], 1)

    def test_source_hash_identity_dedup_and_invalid_replay_ledger(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "raw"
            root.mkdir()
            (root / "first.SC2Replay").write_bytes(b"MPQ\x1b replay AAA")
            (root / "clone.SC2Replay").write_bytes(b"MPQ\x1b replay AAA")
            (root / "second.SC2Replay").write_bytes(b"MPQ\x1b replay BBB")
            (root / "._notgame.SC2Replay").write_bytes(
                b"\x00\x05\x16\x07" + b"not a game"
            )
            out = Path(folder) / "new_v4"
            summary = build_dataset(root, out, bridge=SyntheticProcessor())
            self.assertEqual(summary["raw_file_paths_scanned"], 4)
            self.assertEqual(summary["raw_unique_hashes_selected"], 3)
            self.assertEqual(summary["raw_duplicate_paths"], 1)
            self.assertEqual(summary["included_replays"], 2)
            self.assertEqual(summary["exclusion_reason_counts"]["appledouble_metadata"], 1)
            self.assertEqual(summary["n_snapshots"], 4)
            self.assertTrue(summary["historical_frozen_data_used_for_build"] is False)
            evidence = pd.read_csv(out / "per_replay_source_audit.csv")
            self.assertEqual(len(evidence), 4)
            self.assertEqual(set(evidence.status), {"included", "excluded"})
            self.assertEqual(sum(evidence.reason == "duplicate_content"), 1)
            with zipfile.ZipFile(out / "sc2_v4_clean.zip") as archive:
                df = pd.read_csv(io.BytesIO(archive.read("sc2_v4_clean.csv")))
            self.assertEqual(df.replay_id.nunique(), 2)
            self.assertFalse(df.duplicated(["replay_id", "time_sec"]).any())
            self.assertTrue(all(name.endswith(".SC2Replay") for name in df.replay_id))
            with self.assertRaises(FileExistsError):
                build_dataset(root, out, bridge=SyntheticProcessor())

    def test_nonfinite_and_contradictory_replay_is_refused(self):
        good = pd.DataFrame({
            "replay_id": ["x", "x"], "time_sec": [135, 150],
            "p1_wins": [0, 0], "feature": [1.0, float("nan")]
        })
        self.assertEqual(validate_replay_frame(good, "x", 120, 15),
                         "nonfinite_features")
        good.loc[1, "feature"] = 0.0
        good.loc[1, "p1_wins"] = 1
        self.assertEqual(validate_replay_frame(good, "x", 120, 15),
                         "contradictory_winner")
        good.loc[1, "p1_wins"] = 0
        good.loc[1, "time_sec"] = 135
        self.assertEqual(validate_replay_frame(good, "x", 120, 15),
                         "duplicate_or_unordered_times")

    def test_no_output_is_generated_for_no_valid_replays(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "raw"
            root.mkdir()
            (root / "._fake.SC2Replay").write_bytes(b"\x00\x05\x16\x07bad")
            output = Path(folder) / "new"
            with self.assertRaisesRegex(ValueError, "No valid replay snapshots"):
                build_dataset(root, output, bridge=SyntheticProcessor())
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
