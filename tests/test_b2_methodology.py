"""B2 regression checks: isolated RF fit, deep test gating, calibration leakage."""

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


def load_script(name: str):
    path = ROOT / 'scripts' / name
    spec = importlib.util.spec_from_file_location(name.replace('.py', ''), path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def toy_frame() -> pd.DataFrame:
    rng = np.random.default_rng(123)
    rows = []
    for i in range(24):
        label = i % 2
        for j in range(3):
            rows.append({
                'replay_id': f'r{i:03}', 'time_sec': 135 + 15 * j,
                'p1_wins': label, 'feature_a': float(label) + rng.normal(0, .4),
                'feature_b': rng.normal(), 'feature_counter': rng.normal(),
            })
    return pd.DataFrame(rows)


def toy_split() -> dict:
    return {
        'seed': 43,
        'train_groups': [f'r{i:03}' for i in range(12)],
        'val_groups': [f'r{i:03}' for i in range(12, 18)],
        'test_groups': [f'r{i:03}' for i in range(18, 24)],
    }


class B2MethodologyTests(unittest.TestCase):
    def test_rf_train_only_is_isolated_and_uses_fewer_fit_rows(self):
        data = toy_frame()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            zippath = root / 'tiny.zip'
            with zipfile.ZipFile(zippath, 'w') as archive:
                archive.writestr('tiny.csv', data.to_csv(index=False))
            splitpath = root / 'split.json'
            splitpath.write_text(json.dumps(toy_split()), encoding='utf-8')
            results = {}
            for mode in ('train', 'train_val'):
                outroot = root / mode
                proc = subprocess.run([
                    sys.executable, str(ROOT / 'scripts/54_run_rf_full.py'),
                    '--dataset-zip', str(zippath), '--split-json', str(splitpath),
                    '--dataset-name', 'tiny', '--fit-partition', mode,
                    '--output-root', str(outroot), '--n-estimators', '12',
                    '--n-jobs', '1', '--profile', 'no_counter',
                ], text=True, capture_output=True, check=False)
                self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
                results[mode] = json.loads((outroot / 'tiny/seed_43/metrics_summary.json').read_text())
            self.assertEqual(results['train']['n_rows_fit'], 36)
            self.assertEqual(results['train_val']['n_rows_fit'], 54)
            self.assertEqual(results['train']['fit_partition'], 'train')
            self.assertEqual(results['train_val']['fit_partition'], 'train_val')
            self.assertEqual(results['train']['n_rows_test'], 18)

    def test_deep_test_is_only_scored_after_candidate_selection(self):
        deep = load_script('64_run_deep_finalist.py')
        df = toy_frame()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            splitpath = root / 'split.json'
            splitpath.write_text(json.dumps(toy_split()))
            args = types.SimpleNamespace(
                dataset_zip='unused.zip', split_json=str(splitpath),
                dataset_name='toy', device='cpu', candidate_set='strict',
                selection_metric='neg_log_loss', output_root=str(root / 'deep'),
                random_state=42,
            )
            base = {
                'name': 'candidate_a', 'profile': 'full', 'scaler': 'standard',
                'hidden_dims': [8, 4], 'dropout': .1, 'activation': 'relu',
                'batch_norm': True, 'lr': .005, 'weight_decay': 0.0,
                'batch_size': 8, 'max_epochs': 3, 'patience': 2,
                'scheduler': False, 'use_pos_weight': False,
            }
            candidates = [base, {**base, 'name': 'candidate_b', 'hidden_dims': [6, 3]}]
            original_predict = deep.predict_prob
            predicted_sizes = []

            def counted_predict(model, x, device, batch_size=4096):
                predicted_sizes.append(len(x))
                return original_predict(model, x, device, batch_size)

            with patch.object(deep, 'parse_args', return_value=args), \
                 patch.object(deep, 'load_zip_csv', return_value=df), \
                 patch.object(deep, 'get_candidates', return_value=candidates), \
                 patch.object(deep, 'predict_prob', side_effect=counted_predict):
                deep.run()

            self.assertEqual(predicted_sizes, [18, 18, 18])
            output = root / 'deep/toy/seed_43'
            search = pd.read_csv(output / 'validation_search_results.csv')
            self.assertFalse(any(k.startswith('test_') for k in search.columns))
            self.assertEqual(len(pd.read_csv(output / 'test_predictions.csv')), 18)
            summary = json.loads((output / 'metrics_summary.json').read_text())
            self.assertEqual(summary['test_evaluation_policy'], 'selected_candidate_only')

    def test_calibration_rejects_leakage_and_invalid_probabilities(self):
        calibrate = load_script('63_run_rigorous_calibration.py')
        frame = pd.DataFrame({
            'replay_id': ['a', 'b', 'c', 'd'], 'time_sec': [135] * 4,
            'y_true': [0, 1, 0, 1], 'y_prob': [.1, .8, .2, .9],
        })
        test = frame.copy()
        test['replay_id'] = ['e', 'f', 'g', 'h']
        calibrate.verify_prediction_partitions(frame, test)
        with self.assertRaisesRegex(ValueError, 'leakage'):
            calibrate.verify_prediction_partitions(frame, frame)
        bad = test.copy()
        bad.loc[0, 'y_prob'] = 1.2
        with self.assertRaisesRegex(ValueError, 'probabilities'):
            calibrate.verify_prediction_partitions(frame, bad)
        duplicated = test.copy()
        duplicated.loc[1, 'replay_id'] = 'e'
        with self.assertRaisesRegex(ValueError, 'duplicated'):
            calibrate.verify_prediction_partitions(frame, duplicated)


    def test_four_way_split_is_deterministic_and_disjoint(self):
        strict = load_script('67_run_strict_calibration.py')
        plan = strict.split_four_way(toy_split())
        self.assertEqual(plan, strict.split_four_way(toy_split()))
        parts = [set(plan[name + '_groups']) for name in ('train', 'tune', 'calibration', 'test')]
        for i, left in enumerate(parts):
            for right in parts[i + 1:]:
                self.assertFalse(left & right)
        self.assertEqual(len(parts[1]) + len(parts[2]), len(toy_split()['val_groups']))
        malformed = toy_split()
        malformed['test_groups'] = malformed['val_groups']
        with self.assertRaisesRegex(ValueError, 'Overlapping'):
            strict.split_four_way(malformed)

    def test_four_way_strict_calibration_runs_both_models(self):
        rng = np.random.default_rng(99)
        rows = []
        for i in range(80):
            for j in range(3):
                rows.append({
                    'replay_id': f'R{i}', 'time_sec': 135 + 15 * j,
                    'p1_wins': i % 2, 'econ': float(i % 2) + rng.normal(),
                    'army': rng.normal(), 'counter_power': rng.normal(),
                })
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / 'toy.zip'
            with zipfile.ZipFile(archive, 'w') as zf:
                zf.writestr('toy.csv', pd.DataFrame(rows).to_csv(index=False))
            split = root / 'split.json'
            split.write_text(json.dumps({
                'seed': 43, 'train_groups': [f'R{i}' for i in range(40)],
                'val_groups': [f'R{i}' for i in range(40, 60)],
                'test_groups': [f'R{i}' for i in range(60, 80)],
            }))
            for model in ('rf', 'xgb'):
                cmd = [
                    sys.executable, str(ROOT / 'scripts/67_run_strict_calibration.py'),
                    '--dataset-zip', str(archive), '--split-json', str(split),
                    '--dataset-name', 'toy', '--model', model,
                    '--output-root', str(root / 'b2'), '--n-estimators', '15',
                    '--num-boost-round', '25', '--early-stopping-rounds', '5',
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, check=False)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                output = root / 'b2/toy' / model / 'seed_43'
                manifest = json.loads((output / 'split_manifest.json').read_text())
                counts = json.loads((output / 'metrics_summary.json').read_text())
                self.assertEqual(counts['partition_rows'], {
                    'train': 120, 'tune': 30, 'calibration': 30, 'test': 60,
                })
                self.assertFalse(set(manifest['tune_groups']) & set(manifest['calibration_groups']))
                self.assertEqual(len(pd.read_csv(output / 'test_predictions_platt.csv')), 60)
                # No silently overwriting a prior experiment.
                second = subprocess.run(cmd, capture_output=True, text=True, check=False)
                self.assertNotEqual(second.returncode, 0)


if __name__ == '__main__':
    unittest.main()
