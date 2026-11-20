"""Fast stdlib tests for immutable Git v1.2 controlled source experiment."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/85_compare_b3_git_v12_source_parity.py"


def auditor():
    spec = importlib.util.spec_from_file_location("b3_git_v12", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class B3GitV12ParityTests(unittest.TestCase):
    def test_git_blob_hash_matches_known_git_fixture(self):
        x = auditor()
        self.assertEqual(
            x.git_blob_sha(b"hello\n"),
            "ce013625030ba8dba906f756967f9e9ca394464a",
        )
        self.assertNotEqual(x.git_blob_sha(b"hello"), x.git_blob_sha(b"hello\n"))
        self.assertEqual(len(x.GIT_SOURCE_BLOBS), 4)
        self.assertEqual(x.V12_COMMIT,
                         "bd64a32b305cc7203d592c7fc9015a01f962d6f9")

    def test_paired_baseline_checks_replay_hash_and_sample_identity(self):
        x = auditor()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            a = root / "r1.SC2Replay"
            a.write_bytes(b"MPQ\x1b first raw replay")
            b = root / "r2.SC2Replay"
            b.write_bytes(b"MPQ\x1b second raw replay")
            sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
            original = {
                "schema": "ml_sc2.b3.historical_raw_to_v31_sample.v1",
                "frozen_v31_sha256": "z" * 64,
                "sampled_replay_ids": [a.name, b.name],
                "sampled_raw_file_sha256": [
                    {"replay_id": a.name, "raw_sha256": sha(a)},
                    {"replay_id": b.name, "raw_sha256": sha(b)},
                ],
                "sampled_parity_count": 0,
                "total_feature_mismatches": 1632,
            }
            path = root / "summary.json"
            path.write_text(json.dumps(original))
            result = x.verify_baseline(
                path, [(a.name, a), (b.name, b)], "z" * 64, sha
            )
            self.assertEqual(result["baseline_original_feature_mismatches"], 1632)
            self.assertTrue(result["same_raw_source_bytes_as_baseline"])
            with self.assertRaisesRegex(ValueError, "identical baseline replay"):
                x.verify_baseline(path, [(b.name, b), (a.name, a)], "z" * 64, sha)
            b.write_bytes(b"modified")
            with self.assertRaisesRegex(ValueError, "Raw replay SHA differs"):
                x.verify_baseline(path, [(a.name, a), (b.name, b)], "z" * 64, sha)

    def test_comparison_aggregate_and_replay_failure_are_not_hidden(self):
        x = auditor()
        rows = [
            {"replay_id": "r1", "status": "SAMPLED_PARITY",
             "feature_comparisons": 64, "feature_mismatches": 0,
             "label_mismatches": 0, "time_keys_only_frozen": 0,
             "time_keys_only_reconstructed": 0},
            {"replay_id": "r2", "status": "RECONSTRUCTION_MISMATCH",
             "feature_comparisons": 64, "feature_mismatches": 3,
             "label_mismatches": 0, "time_keys_only_frozen": 0,
             "time_keys_only_reconstructed": 0},
            {"replay_id": "r3", "status": "GIT_V12_RECONSTRUCTION_EXCEPTION",
             "error_type": "ValueError"},
        ]
        fields = [
            {"feature": "p1_tech_level", "comparisons": 2, "mismatches": 2},
            {"feature": "p1_combat", "comparisons": 2, "mismatches": 1},
        ]
        s = x.summarize_comparisons(rows, fields)
        self.assertEqual(s["number_of_replays"], 3)
        self.assertEqual(s["full_replay_parity_count"], 1)
        self.assertEqual(s["replays_with_mismatches_or_errors"], 2)
        self.assertEqual(s["total_feature_mismatches"], 3)
        self.assertEqual(s["feature_columns_with_mismatches"], 2)

    def test_rejects_mutable_or_wrong_git_revision_before_loading(self):
        x = auditor()
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "Only independently identified"):
                x.load_git_v12_sources(
                    Path(folder), Path(folder),
                    commit="main",
                )


if __name__ == "__main__":
    unittest.main()
