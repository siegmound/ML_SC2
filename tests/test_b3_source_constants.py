"""Tests for B3 historical constants vs frozen-tech incompatibility audit."""
from __future__ import annotations

import csv
import importlib.util
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def auditor():
    spec = importlib.util.spec_from_file_location(
        "b3_source_constants_under_test", ROOT / "scripts/80_audit_b3_source_constants.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class B3SourceConstantsTests(unittest.TestCase):
    def test_report_groups_root_and_derived_mismatch_families(self):
        tool = auditor()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = root / "per_replay.csv"
            summary.write_text("replay_id,status\nr.SC2Replay,RECONSTRUCTION_MISMATCH\n")
            self.assertEqual(tool.selected_ids(summary), {"r.SC2Replay"})
            features = root / "per_feature.csv"
            with features.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "replay_id", "feature", "comparisons", "mismatches"
                ])
                writer.writeheader()
                for feature, bad in [
                    ("p1_tech_level", 2), ("p2_tech_level", 1),
                    ("tech_diff", 1), ("p1_combat", 2),
                    ("counter_advantage_p1", 2),
                    ("diff_combat_60s", 1), ("p1_workers", 0),
                ]:
                    writer.writerow({
                        "replay_id": "r.SC2Replay", "feature": feature,
                        "comparisons": 3, "mismatches": bad,
                    })
            report = tool.mismatches_by_family(features, {"r.SC2Replay"})
            self.assertEqual(report["total_mismatches"], 9)
            self.assertEqual(report["mismatch_counts_by_family"]["tech_levels"], 4)
            self.assertEqual(report["mismatch_counts_by_family"]["combat_primitive"], 2)
            self.assertEqual(report["mismatch_counts_by_family"]["counter_primitive"], 2)
            self.assertEqual(report["mismatch_counts_by_family"]["combat_derived"], 1)
            self.assertEqual(report["mismatch_counts_by_family"].get("unaffected_other", 0), 0)

    def test_constants_source_hash_rejects_unattested_bytes(self):
        tool = auditor()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fake.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr(
                    "sc2_constants.py",
                    "UNIT_STATS = {'Marine': {'power': 2, 'type': 'army'}}\n",
                )
            with self.assertRaisesRegex(ValueError, "hash does not match"):
                tool.constants_from_historical_zip(path)

    def test_refuse_duplicate_replay_and_invalid_metric_count(self):
        tool = auditor()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "per_replay.csv"
            path.write_text("replay_id\nr.SC2Replay\nr.SC2Replay\n")
            with self.assertRaisesRegex(ValueError, "Duplicate or blank"):
                tool.selected_ids(path)
            path2 = Path(directory) / "per_feature.csv"
            path2.write_text(
                "replay_id,feature,comparisons,mismatches\n"
                "r.SC2Replay,p1_combat,3,4\n"
            )
            with self.assertRaisesRegex(ValueError, "Bad feature comparison totals"):
                tool.mismatches_by_family(path2, {"r.SC2Replay"})


if __name__ == "__main__":
    unittest.main()
