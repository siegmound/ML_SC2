"""Validation of replay-disjoint v4 test protocol and training plan."""
from __future__ import annotations

import importlib.util
import io
import json
import sys
import tempfile
import types
import unittest
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
try:
    import sc2reader
except ImportError:
    sys.modules["sc2reader"] = types.ModuleType("sc2reader")


def suite():
    spec = importlib.util.spec_from_file_location(
        "sc2_v4_suite", ROOT / "scripts/91_v4_run_suite.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class V4ExperimentSuiteTests(unittest.TestCase):
    def make_dataset(self, n=60):
        return pd.DataFrame([
            {"replay_id": f"{i:064x}.SC2Replay", "time_sec": sec,
             "p1_wins": i % 2, "diff_workers": float(i % 4)}
            for i in range(n) for sec in (135, 150)
        ])

    def test_prespecified_core_profile_avoids_uncalibrated_heuristics(self):
        tool = suite()
        df = self.make_dataset()
        df["counter_advantage_diff"] = 0.5
        df["p1_combat"] = 0.2
        df["delta_combat_p1"] = 1.0
        df["p1_tech_level"] = 1.0
        core = tool.feature_profile_columns(df, "core")
        full = tool.feature_profile_columns(df, "full")
        self.assertIn("p1_tech_level", core)
        self.assertIn("counter_advantage_diff", full)
        self.assertNotIn("counter_advantage_diff", core)
        self.assertNotIn("p1_combat", core)
        self.assertNotIn("delta_combat_p1", core)
        self.assertEqual(core[:3], ["replay_id", "time_sec", "p1_wins"])
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "test.zip"
            digest1 = tool.deterministic_profile_zip(df, "core", out)
            self.assertEqual(digest1, tool.digest(out))
            with zipfile.ZipFile(out) as z:
                columns = pd.read_csv(z.open("sc2_v4_clean.csv")).columns
            self.assertEqual(list(columns), core)
            with self.assertRaises(FileExistsError):
                tool.deterministic_profile_zip(df, "core", out)

    def test_group_split_covers_each_replay_once(self):
        tool = suite()
        df = self.make_dataset()
        plans = tool.validate_and_plan_splits(df, [42, 43, 44])
        self.assertEqual(set(plans), {42, 43, 44})
        for split in plans.values():
            a, b, c = [set(split[k]) for k in (
                "train_groups", "val_groups", "test_groups"
            )]
            self.assertFalse(a & b or a & c or b & c)
            self.assertEqual(len(a | b | c), 60)

    def test_non_hash_identity_and_conflicting_labels_refused(self):
        tool = suite()
        df = self.make_dataset()
        df.loc[0, "replay_id"] = "ambiguous.SC2Replay"
        with self.assertRaisesRegex(ValueError, "content-SHA"):
            tool.validate_and_plan_splits(df, [42])
        df = self.make_dataset()
        df.loc[1, "p1_wins"] = 1 - int(df.loc[1, "p1_wins"])
        with self.assertRaisesRegex(ValueError, "labels are not constant"):
            tool.validate_and_plan_splits(df, [42])

    def test_plan_has_no_test_based_search_or_output_collisions(self):
        tool = suite()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cmds = tool.commands(
                root / "data.zip", root / "splits", root / "experiments",
                models=("rf", "xgb", "deep"), seeds=(42, 43, 44),
                stage="smoke", device="cpu", n_jobs=2,
            )
            self.assertEqual(len(cmds), 9)
            self.assertEqual(len({str(row[2]) for row in cmds}), 9)
            for seed, model, output, argv in cmds:
                self.assertTrue(str(output).startswith(str(root / "experiments")))
                self.assertIn("--split-json", argv)
                self.assertIn(f"v4_clean_split_seed_{seed}.json", " ".join(argv))
                self.assertNotIn("starcraft_full_dataset_v3_1_fixed.zip", argv)
                if model == "deep":
                    self.assertTrue(any("64_run_deep_finalist.py" in x for x in argv))
                else:
                    self.assertIn("--model", argv)
                    self.assertTrue(any("67_run_strict_calibration.py" in x for x in argv))


if __name__ == "__main__":
    unittest.main()
