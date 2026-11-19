"""B3 original raw-source inventory tests (no real replay downloads required)."""
from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/77_inventory_b3_raw_sources.py"


class B3RawInventoryTests(unittest.TestCase):
    def set_up(self, folder: str):
        base = Path(folder)
        raw = base / "raw"
        raw.mkdir()
        with zipfile.ZipFile(raw / "packA.zip", "w") as zf:
            zf.writestr("event/r1.SC2Replay", b"MPQ fake one")
            zf.writestr("event/r2.SC2Replay", b"MPQ fake two")
            zf.writestr("nested_more.zip", b"not inspected")
        (raw / "r1.SC2Replay").write_bytes(b"MPQ fake one")
        frozen = base / "frozen.zip"
        with zipfile.ZipFile(frozen, "w") as zf:
            zf.writestr("frozen.csv",
                        "replay_id,time_sec,p1_wins\\n"
                        "r1.SC2Replay,135,1\\n"
                        "r3.SC2Replay,135,0\\n")
        catalogue = base / "catalogue.csv"
        catalogue.write_text(
            "replaypack_filename\\npackA.zip\\npackB.zip\\n",
            encoding="utf-8",
        )
        return raw, frozen, catalogue

    def command(self, raw, frozen, catalogue, out):
        return [
            sys.executable, str(RUNNER),
            "--raw-root", str(raw),
            "--processed-zip", str(frozen),
            "--pack-catalog", str(catalogue),
            "--output-dir", str(out),
            "--documented-upstream", "3",
        ]

    def test_inventory_tracks_duplicates_and_unknown_exclusions(self):
        with tempfile.TemporaryDirectory() as folder:
            raw, frozen, catalog = self.set_up(folder)
            out = Path(folder) / "analysis"
            proc = subprocess.run(
                self.command(raw, frozen, catalog, out),
                capture_output=True, text=True
            )
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            report = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(report["source_replay_entries"], 3)
            self.assertEqual(report["source_unique_replay_basenames"], 2)
            self.assertEqual(report["source_basename_collisions"], 1)
            self.assertEqual(report["unique_source_ids_also_frozen"], 1)
            self.assertEqual(report["unique_source_ids_absent_frozen_unattributed"], 1)
            self.assertEqual(report["frozen_ids_missing_from_scanned_sources"], 1)
            self.assertEqual(report["catalogued_packs_with_raw_entries"], 1)
            self.assertEqual(report["catalogued_packs_not_observed"], ["packB.zip"])
            self.assertEqual(report["zip_files_with_uninspected_nested_archives"],
                             ["packA.zip"])
            self.assertFalse(report["historical_raw_to_feature_certified"])
            self.assertFalse(report["exclusion_reasons_certified"])
            with (out / "unresolved_ids.csv").open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual({row["replay_id"] for row in rows},
                             {"r2.SC2Replay", "r3.SC2Replay"})
            self.assertTrue(all("REASON_UNKNOWN" in row["classification"]
                                or "MISSING_FROM" in row["classification"]
                                for row in rows))
            second = subprocess.run(self.command(raw, frozen, catalog, out),
                                    capture_output=True, text=True)
            self.assertNotEqual(second.returncode, 0)
            self.assertIn("Refusing to overwrite", second.stderr + second.stdout)

    def test_absent_raw_fails_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            raw, frozen, catalog = self.set_up(folder)
            (raw / "r1.SC2Replay").unlink()
            (raw / "packA.zip").unlink()
            proc = subprocess.run(
                self.command(raw, frozen, catalog, Path(folder) / "output"),
                capture_output=True, text=True
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("No actual .SC2Replay", proc.stderr + proc.stdout)

    def test_corrupt_zip_fails_closed_without_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            raw, frozen, catalog = self.set_up(folder)
            (raw / "broken.zip").write_text("Not a valid archive")
            out = Path(folder) / "output"
            proc = subprocess.run(self.command(raw, frozen, catalog, out),
                                  capture_output=True, text=True)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("Unreadable ZIP source", proc.stderr + proc.stdout)
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
