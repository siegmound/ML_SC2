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

            # XGB originally reports log_loss from float32 raw booster outputs;
            # the CSV reader normally promotes them to float64.
            xgb = subprocess.run([
                sys.executable, str(ROOT / "scripts/67_run_strict_calibration.py"),
                "--dataset-zip", str(archive), "--split-json", str(split),
                "--dataset-name", "real_v3_1_fixed", "--model", "xgb",
                "--device", "cpu",
                "--output-root", str(out / "strict_calibration"),
                "--num-boost-round", "25", "--early-stopping-rounds", "5",
                "--n-jobs", "1",
            ], text=True, capture_output=True)
            self.assertEqual(xgb.returncode, 0, xgb.stdout + xgb.stderr)
            xgb_cmd = cmd[:-1] + ["xgb"]
            xgb_verified = subprocess.run(xgb_cmd, text=True, capture_output=True)
            self.assertEqual(xgb_verified.returncode, 0,
                             xgb_verified.stdout + xgb_verified.stderr)
            # A plausible-looking but altered in-range probability must still
            # fail the 1e-6 float32 CSV metric allowance.
            xgb_pred = (out / "strict_calibration/real_v3_1_fixed"
                        / "xgb/seed_43/test_predictions_uncalibrated.csv")
            xgb_frame = pd.read_csv(xgb_pred)
            xgb_original = xgb_frame.copy(deep=True)
            xgb_frame.loc[0, "y_prob"] = 0.99 if xgb_frame.loc[0, "y_prob"] < 0.5 else 0.01
            xgb_frame.loc[0, "y_pred"] = int(xgb_frame.loc[0, "y_prob"] >= 0.5)
            xgb_frame.to_csv(xgb_pred, index=False)
            rejected = subprocess.run(xgb_cmd, text=True, capture_output=True)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("Metric", rejected.stderr + rejected.stdout)
            xgb_original.to_csv(xgb_pred, index=False)
            # Isotonic probabilities inherit the same float32 round-trip
            # allowance. A realistic large modification still fails.
            isotonic_path = (out / "strict_calibration/real_v3_1_fixed"
                             / "xgb/seed_43/test_predictions_isotonic.csv")
            iso_frame = pd.read_csv(isotonic_path)
            iso_frame.loc[0, "y_prob"] = 0.99 if iso_frame.loc[0, "y_prob"] < 0.5 else 0.01
            iso_frame.loc[0, "y_pred"] = int(iso_frame.loc[0, "y_prob"] >= 0.5)
            iso_frame.to_csv(isotonic_path, index=False)
            iso_rejected = subprocess.run(xgb_cmd, text=True, capture_output=True)
            self.assertNotEqual(iso_rejected.returncode, 0)
            self.assertIn("Metric", iso_rejected.stderr + iso_rejected.stdout)


    def test_xgb_float32_metric_csv_roundtrip(self):
        """XGB raw probabilities must be audited in their native float32 precision."""
        from io import StringIO

        def module(filename, module_name):
            spec = importlib.util.spec_from_file_location(module_name, ROOT / "scripts" / filename)
            assert spec and spec.loader
            value = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(value)
            return value

        strict = module("67_run_strict_calibration.py", "b2_strict_for_precision_test")
        verifier = module("75_verify_b2_full_fair.py", "b2_verify_for_precision_test")

        labels = np.array([0, 1, 0, 1, 1, 0, 1, 0], dtype=np.int64)
        probabilities = np.array([
            0.10123456, 0.74234569, 0.22654322, 0.54876542,
            0.63898766, 0.43345678, 0.88877654, 0.31987655,
        ], dtype=np.float32)
        frame = pd.DataFrame({
            "replay_id": [f"X{i}" for i in range(len(labels))],
            "time_sec": [120] * len(labels),
            "y_true": labels, "y_prob": probabilities,
            "y_pred": (probabilities >= 0.5).astype(int),
        })
        csv_buffer = StringIO()
        frame.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        restored = pd.read_csv(csv_buffer, dtype={"replay_id": str})
        np.testing.assert_array_equal(
            restored["y_prob"].to_numpy(dtype=np.float32), probabilities
        )
        original_metrics = strict.metrics(labels, probabilities)
        reproduced = verifier.scores(restored, True, probability_dtype=np.float32)
        verifier.check_scores(reproduced, original_metrics, Path("float32_csv"),
                              float32_csv_roundtrip=True)

        # Native float32 CSV serialization may shift aggregate metrics by a
        # few 1e-7, but the verifier must not accept genuine 1e-4 tampering.
        near = dict(original_metrics)
        near["log_loss"] += 5e-7
        verifier.check_scores(reproduced, near, Path("float32_csv"),
                              float32_csv_roundtrip=True)
        corrupted = dict(original_metrics)
        corrupted["log_loss"] += 1e-4
        with self.assertRaisesRegex(ValueError, "Metric log_loss mismatch"):
            verifier.check_scores(reproduced, corrupted, Path("float32_csv"),
                                  float32_csv_roundtrip=True)
        with self.assertRaisesRegex(ValueError, "Metric log_loss mismatch"):
            verifier.check_scores(reproduced, near, Path("float32_csv"),
                                  float32_csv_roundtrip=False)


if __name__ == "__main__":
    unittest.main()
