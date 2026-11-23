"""V4 original outer archive safe extraction: entirely synthetic fixtures."""
from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/94_v4_extract_original_replays.py"


def importer():
    spec = importlib.util.spec_from_file_location("v4_outer_zip_importer", SOURCE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class V4RawArchiveImportTests(unittest.TestCase):
    def make_zip(self, target: Path):
        with zipfile.ZipFile(target, "w") as archive:
            archive.writestr("replays/alpha.SC2Replay", b"MPQ\x1b replay alpha")
            archive.writestr("replays/beta.SC2Replay", b"MPQ\x1b replay alpha")
            archive.writestr("replays/gamma.SC2Replay", b"MPQ\x1b replay gamma")
            archive.writestr("replays/a_tournament.zip", b"nested ZIP ignored")
            archive.writestr("replays/source.json", '{"original":true}')

    def test_plan_then_crc_checked_unique_path_extraction(self):
        mod = importer()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = root / "source.zip"
            self.make_zip(source)
            output = root / "extracted"
            hash_value = mod.digest_file(source)
            plan = mod.extract_replays(
                source, output, expected_sha=hash_value,
                expected_count=3, execute=False,
            )
            self.assertEqual(plan["status"], "PLAN_ONLY_NO_OUTPUTS_CREATED")
            self.assertFalse(output.exists())
            result = mod.extract_replays(
                source, output, expected_sha=hash_value,
                expected_count=3, execute=True,
            )
            self.assertEqual(result["selected_direct_raw_replay_members"], 3)
            self.assertEqual(result["extracted_replay_paths"], 3)
            self.assertEqual(result["byte_unique_replay_contents"], 2)
            self.assertEqual(result["duplicate_binary_content_count"], 1)
            self.assertEqual(result["embedded_tournament_zip_members_extracted"], 0)
            self.assertEqual(result["header_type_counts"], {"MPQ_CANDIDATE": 3})
            self.assertEqual(
                (output / "replays/alpha.SC2Replay").read_bytes(),
                b"MPQ\x1b replay alpha",
            )
            self.assertFalse((output / "replays/a_tournament.zip").exists())
            with (output / "original_replay_source_manifest.csv").open(newline="") as f:
                record = list(csv.DictReader(f))
            self.assertEqual(len(record), 3)
            self.assertEqual(
                json.loads((output / "manifest.json").read_text())["status"],
                "EXTRACTED_AND_CRC_VERIFIED",
            )
            with self.assertRaisesRegex(ValueError, "Refusing overwrite"):
                mod.extract_replays(
                    source, output, expected_sha=hash_value,
                    expected_count=3, execute=True,
                )

    def test_rejects_archive_mismatch_and_unexpected_file_count(self):
        mod = importer()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            path = root / "source.zip"
            self.make_zip(path)
            out = root / "not_created"
            with self.assertRaisesRegex(ValueError, "SHA-256 changed"):
                mod.extract_replays(
                    path, out, expected_sha="0" * 64,
                    expected_count=3, execute=True,
                )
            self.assertFalse(out.exists())
            with self.assertRaisesRegex(ValueError, "Expected 7"):
                mod.extract_replays(
                    path, out, expected_sha=mod.digest_file(path),
                    expected_count=7, execute=True,
                )
            self.assertFalse(out.exists())

    def test_rejects_bad_path_duplicate_basename_and_oversize(self):
        mod = importer()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            bad = root / "bad.zip"
            with zipfile.ZipFile(bad, "w") as z:
                z.writestr("../escape.SC2Replay", b"MPQ\x1b hello")
            with zipfile.ZipFile(bad) as z:
                with self.assertRaisesRegex(ValueError, "Unsafe"):
                    mod.source_members(z, max_member_bytes=1000, max_total_bytes=1000)
            dup = root / "duplicate.zip"
            with zipfile.ZipFile(dup, "w") as z:
                z.writestr("a/same.SC2Replay", b"MPQ\x1b aaa")
                z.writestr("b/same.SC2Replay", b"MPQ\x1b bbb")
            with zipfile.ZipFile(dup) as z:
                with self.assertRaisesRegex(ValueError, "Ambiguous basename"):
                    mod.source_members(z, max_member_bytes=1000, max_total_bytes=1000)
            good = root / "good.zip"
            self.make_zip(good)
            with zipfile.ZipFile(good) as z:
                with self.assertRaisesRegex(ValueError, "per-file safety"):
                    mod.source_members(z, max_member_bytes=5, max_total_bytes=1000)


if __name__ == "__main__":
    unittest.main()
