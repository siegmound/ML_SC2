"""B3 historical-build discovery tests: local temporary files only."""
from __future__ import annotations

import importlib.util
import tempfile
import unittest
import zipfile
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/86_discover_b3_generation_sources.py"


def discoverer():
    spec = importlib.util.spec_from_file_location("b3_historical_discovery", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class B3HistoricalDiscoveryTests(unittest.TestCase):
    def test_find_real_replay_magic_archives_and_builder_log(self):
        mod = discoverer()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "old_project"
            raw = source / "replays"
            raw.mkdir(parents=True)
            (raw / "valid.SC2Replay").write_bytes(b"MPQ\x1b" + b"sample bytes")
            (raw / "metadata.SC2Replay").write_bytes(
                b"\x00\x05\x16\x07" + b"AppleDouble metadata"
            )
            (source / "original_build.log").write_text(
                "Avvio estrazione massiva su 24057 replay...\n"
                "Snapshot totali: 892048\n", encoding="utf-8"
            )
            (source / "buildf_v3_counter120.py").write_text("MIN_DURATION_SEC=180\n")
            with zipfile.ZipFile(root / "cup.zip", "w") as z:
                z.writestr("event/replay1.SC2Replay", b"MPQ\x1bR1")
                z.writestr("event/replay2.SC2Replay", b"MPQ\x1bR2")
                z.writestr("nested.zip", b"not opened")
            report, details = mod.scan([root], {"cup.zip"})
            self.assertEqual(report["loose_sc2replay_files"], 2)
            self.assertEqual(report["loose_mpq_candidates"], 1)
            self.assertEqual(report["loose_appledouble"], 1)
            self.assertEqual(report["direct_zip_replay_entries_not_deduplicated"], 2)
            self.assertEqual(report["catalogued_zenodo_packs_found"], 1)
            self.assertEqual(report["candidate_raw_replay_folders"][0]["n_extension_files"], 2)
            self.assertTrue(any(
                item["filename"] == "buildf_v3_counter120.py"
                for item in details["source_candidates"]
            ))
            self.assertTrue(any(x["match"] == "24057" for x in details["text_witnesses"]))
            self.assertFalse(report["historical_build_execution_attested"])
            self.assertFalse(report["historical_816_exclusions_explained"])

    def test_symlinks_hardlinks_and_git_are_not_recounted(self):
        mod = discoverer()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "a.SC2Replay").write_bytes(b"MPQ\x1a" + b"data")
            (root / "same.SC2Replay").hardlink_to(root / "a.SC2Replay")
            (root / "link.SC2Replay").symlink_to(root / "a.SC2Replay")
            (root / ".git").mkdir()
            (root / ".git" / "fake.SC2Replay").write_bytes(b"MPQ\x1a")
            report, _ = mod.scan([root], set())
            self.assertEqual(report["loose_sc2replay_files"], 1)

    def test_partial_file_scan_preserves_evidence_and_does_not_assert_missing_packs(self):
        mod = discoverer()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            first = root / "first"
            second = root / "second"
            first.mkdir()
            second.mkdir()
            (first / "alpha.SC2Replay").write_bytes(b"MPQ\x1b" + b"actual data")
            (first / "beta.SC2Replay").write_bytes(b"MPQ\x1b" + b"second data")
            (second / "cup.zip").write_bytes(b"not inspected")
            result, details = mod.scan(
                [first, second], {"cup.zip"}, max_files=1, on_limit="partial"
            )
            self.assertEqual(result["scan_status"], "PARTIAL_INCOMPLETE_LIMIT_REACHED")
            self.assertFalse(result["scan_complete"])
            self.assertEqual(result["limit_reached"], "max_files")
            self.assertEqual(result["n_visited_files"], 1)
            self.assertEqual(result["loose_mpq_candidates"], 1)
            self.assertEqual(len(details["raw_files"]), 1)
            self.assertIsNone(result["catalogued_zenodo_packs_missing"])
            self.assertEqual(result["roots_not_started_due_to_limit"], [str(second)])

    def test_partial_zip_limit_is_explicit_and_non_exhaustive(self):
        mod = discoverer()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ("a.zip", "b.zip"):
                with zipfile.ZipFile(root / name, "w") as z:
                    z.writestr("example.SC2Replay", b"MPQ\x1b")
            report, _ = mod.scan(
                [root], {"a.zip", "b.zip"},
                max_archives=1, on_limit="partial",
            )
            self.assertEqual(report["zip_files_examined"], 1)
            self.assertEqual(report["catalogued_zenodo_packs_found"], 1)
            self.assertEqual(report["limit_reached"], "max_archives")
            self.assertFalse(report["scan_complete"])
            self.assertIsNone(report["catalogued_zenodo_packs_missing"])

    def test_invalid_catalogue_and_limits_rejected(self):
        mod = discoverer()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            catalog = root / "packs.csv"
            catalog.write_text("replaypack_filename\ncup.zip\ncup.zip\n")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                mod.replaypack_catalog(catalog)
            (root / "x.txt").write_text("abc")
            with self.assertRaisesRegex(ValueError, "cap"):
                mod.scan([root], set(), max_files=1)


if __name__ == "__main__":
    unittest.main()
