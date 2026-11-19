"""B2 FAIR guard regressions: strict outputs, metrics and Deep CUDA behavior."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import types
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def deep_module():
    spec = importlib.util.spec_from_file_location("b2_deep", ROOT / "scripts/64_run_deep_finalist.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class B2FullFAIRTests(unittest.TestCase):
    def test_deep_refuses_existing_output_or_silent_cpu_fallback(self):
        deep = deep_module()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            split = root / "split.json"
            split.write_text(json.dumps({"seed": 42}), encoding="utf-8")
            args = types.SimpleNamespace(
                dataset_zip=str(root / "unused.zip"), split_json=str(split),
                dataset_name="toy", random_state=42,
                output_root=str(root / "deep"), device="cuda",
            )
            output = root / "deep/toy/seed_42"
            output.mkdir(parents=True)
            with patch.object(deep, "parse_args", return_value=args):
                with self.assertRaisesRegex(FileExistsError, "refusing overwrite"):
                    deep.run()
            output.rmdir()
            with patch.object(deep, "parse_args", return_value=args):
                with patch.object(deep.torch.cuda, "is_available", return_value=False):
                    with self.assertRaisesRegex(RuntimeError, "CUDA requested"):
                        deep.run()

    def test_verifier_rechecks_real_predictions_not_just_file_existence(self):
        rng = np.random.default_rng(123)
        rows = []
        for i in range(80):
            for j in range(3):
                rows.append({
                    "replay_id": f"R{i}", "time_sec": 135 + j * 15,
                    "p1_wins": i % 2, "econ": float(i % 2) + rng.normal(),
                    "army": rng.normal(), "counter_power": rng.normal(),
                })
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            archive = root / "toy.zip"
            with zipfile.ZipFile(archive, "w") as zf:
                zf.writestr("toy.csv", pd.DataFrame(rows).to_csv(index=False))
            split = root / "data/processed/splits/real_v3_1_fixed_split_seed_43.json"
            split.parent.mkdir(parents=True)
            split.write_text(json.dumps({
                "seed": 43,
                "train_groups": [f"R{i}" for i in range(40)],
                "val_groups": [f"R{i}" for i in range(40, 60)],
                "test_groups": [f"R{i}" for i in range(60, 80)],
            }), encoding="utf-8")
            out = root / "results/b2_fullscale_audit_TEST"
            train = subprocess.run([
                sys.executable, str(ROOT / "scripts/67_run_strict_calibration.py"),
                "--dataset-zip", str(archive), "--split-json", str(split),
                "--dataset-name", "real_v3_1_fixed", "--model", "rf",
                "--output-root", str(out / "strict_calibration"),
                "--n-estimators", "15", "--n-jobs", "1",
            ], text=True, capture_output=True)
            self.assertEqual(train.returncode, 0, train.stdout + train.stderr)
            cmd = [
                sys.executable, str(ROOT / "scripts/75_verify_b2_full_fair.py"),
                "--repo-root", str(root), "--output-base", str(out),
                "--seed", "43", "--model", "rf",
            ]
            verified = subprocess.run(cmd, text=True, capture_output=True)
            self.assertEqual(verified.returncode, 0, verified.stdout + verified.stderr)
            pred = out / "strict_calibration/real_v3_1_fixed/rf/seed_43/test_predictions_platt.csv"
            frame = pd.read_csv(pred)
            frame.loc[0, "y_prob"] = 1.5
            frame.to_csv(pred, index=False)
            invalid = subprocess.run(cmd, text=True, capture_output=True)
            self.assertNotEqual(invalid.returncode, 0)
            self.assertIn("Invalid probabilities", invalid.stderr + invalid.stdout)


if __name__ == "__main__":
    unittest.main()
