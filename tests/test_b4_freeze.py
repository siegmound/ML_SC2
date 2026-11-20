"""B4 Git-object and Git LFS checks are testable without network or GPU."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("b4_freeze", ROOT / "scripts/70_verify_freeze.py")
assert SPEC and SPEC.loader
freeze = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(freeze)


class FreezeVerificationTests(unittest.TestCase):
    def git(self, root: Path, *args):
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout.strip()

    def make_repo(self, root: Path):
        self.git(root, "init", "-q")
        self.git(root, "config", "user.name", "B4 Test")
        self.git(root, "config", "user.email", "b4@example.invalid")
        stored = root / "results/final/metrics_summary.json"
        stored.parent.mkdir(parents=True)
        stored.write_text('{"accuracy": 0.61}\n')
        data = root / "data/processed/starcraft_full_dataset_v3_1_fixed.zip"
        data.parent.mkdir(parents=True)
        real_zip = root / "actual_payload.zip"
        with zipfile.ZipFile(real_zip, "w") as zf:
            zf.writestr("toy.csv", "replay_id,time_sec,feature_a,p1_wins\nr1,135,0.4,1\n")
        payload = real_zip.read_bytes()
        sha = hashlib.sha256(payload).hexdigest()
        data.write_text("version https://git-lfs.github.com/spec/v1\n"
                        f"oid sha256:{sha}\nsize {len(payload)}\n")
        config = root / freeze.FEATURE_CONFIGS["v3_1_fixed"]
        config.parent.mkdir(parents=True)
        config.write_text(json.dumps({"features": ["feature_a"]}))
        self.git(root, "add", ".")
        self.git(root, "commit", "-qm", "test baseline")
        metric_sha = self.git(root, "rev-parse", "HEAD:results/final/metrics_summary.json")
        pointer_sha = self.git(root, "rev-parse", "HEAD:data/processed/starcraft_full_dataset_v3_1_fixed.zip")
        manifest = {
            "tracked_artifact_git_blob_sha1": {
                "results/final/metrics_summary.json": metric_sha,
                "data/processed/starcraft_full_dataset_v3_1_fixed.zip": pointer_sha,
            },
            "datasets": {"data/processed/starcraft_full_dataset_v3_1_fixed.zip": {
                "lfs_sha256": sha, "size_bytes": len(payload), "features": 1,
            }},
        }
        manifest_path = root / "baseline.json"
        manifest_path.write_text(json.dumps(manifest))
        return data, real_zip, manifest_path

    def test_pointer_and_payload_pass_with_verified_git_objects(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset, payload, manifest = self.make_repo(root)
            result = freeze.verify(root, manifest_path=manifest.name)
            self.assertEqual(result["protected_artifacts"], 2)
            self.assertEqual(result["git_object_integrity"], "PASS")
            dataset.write_bytes(payload.read_bytes())
            result = freeze.verify(root, "payload", manifest.name)
            self.assertEqual(result["lfs_payloads"]["data/processed/starcraft_full_dataset_v3_1_fixed.zip"]["features"], 1)

    def test_missing_lfs_payload_is_not_mistaken_for_verified_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, _, manifest = self.make_repo(root)
            with self.assertRaisesRegex(ValueError, "LFS payload size/hash mismatch"):
                freeze.verify(root, "payload", manifest.name)

    def test_mutated_committed_results_are_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, _, manifest = self.make_repo(root)
            source = root / "results/final/metrics_summary.json"
            source.write_text('{"accuracy": 1.0}\n')
            self.git(root, "add", str(source))
            self.git(root, "commit", "-qm", "tamper")
            with self.assertRaisesRegex(ValueError, "Frozen Git object changed"):
                freeze.verify(root, manifest_path=manifest.name)

    def test_mutated_committed_lfs_pointer_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset, _, manifest = self.make_repo(root)
            dataset.write_text("version https://git-lfs.github.com/spec/v1\n"
                               "oid sha256:" + "0" * 64 + "\nsize 1\n")
            self.git(root, "add", str(dataset))
            self.git(root, "commit", "-qm", "tamper pointer")
            with self.assertRaisesRegex(ValueError, "Frozen Git object changed"):
                freeze.verify(root, manifest_path=manifest.name)

    def test_exported_source_directory_not_valid_as_git_checkout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "baseline.json").write_text(json.dumps({}))
            with self.assertRaisesRegex(ValueError, "Git checkout required"):
                freeze.verify(root, manifest_path="baseline.json")


if __name__ == "__main__":
    unittest.main()
