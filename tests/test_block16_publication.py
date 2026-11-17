"""Fast regression tests for the frozen Block 16 publication table (no GPU)."""

from __future__ import annotations

import csv
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / "scripts/58_verify_final_table.py"
TABLE = ROOT / "tables/final_full_multiseed_summary.csv"
DETAILED = ROOT / "results/block16_final/block16_full_detailed.csv"
SUMMARY = ROOT / "results/block16_final/block16_full_summary.csv"


def run_checker(root: Path, mode: str = "--check") -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CHECKER), "--repo-root", str(root), mode],
        capture_output=True,
        text=True,
        check=False,
    )


class Block16PublicationTests(unittest.TestCase):
    def test_checked_in_table_matches_saved_runs(self) -> None:
        result = run_checker(ROOT)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_xgb_combatfix_population_standard_deviations(self) -> None:
        with TABLE.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 4)
        matching = [
            row for row in rows
            if row["dataset"] == "v3_2_combatfix" and row["model"] == "xgb"
        ]
        self.assertEqual(len(matching), 1)
        row = matching[0]
        self.assertEqual(
            (row["acc_std"], row["balacc_std"], row["auc_std"], row["logloss_std"]),
            ("0.0056", "0.0051", "0.0048", "0.0024"),
        )

    def test_tampered_publication_fails_then_can_be_regenerated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "results/block16_final").mkdir(parents=True)
            (root / "tables").mkdir(parents=True)
            shutil.copy2(DETAILED, root / "results/block16_final/block16_full_detailed.csv")
            shutil.copy2(SUMMARY, root / "results/block16_final/block16_full_summary.csv")
            content = TABLE.read_text(encoding="utf-8")
            self.assertIn("v3_2_combatfix,xgb,full,0.6128,0.0056", content)
            corrupted = content.replace(
                "v3_2_combatfix,xgb,full,0.6128,0.0056",
                "v3_2_combatfix,xgb,full,0.6128,0.0069",
            )
            output = root / "tables/final_full_multiseed_summary.csv"
            output.write_text(corrupted, encoding="utf-8")

            bad = run_checker(root)
            self.assertNotEqual(bad.returncode, 0)
            self.assertIn("0.0056", bad.stdout)

            synced = run_checker(root, "--write")
            self.assertEqual(synced.returncode, 0, synced.stdout + synced.stderr)
            self.assertEqual(output.read_text(encoding="utf-8"), content)
            good = run_checker(root)
            self.assertEqual(good.returncode, 0, good.stdout + good.stderr)


if __name__ == "__main__":
    unittest.main()
