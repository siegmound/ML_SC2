"""Check that the two trainable finalist families can save their fitted state."""
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

import joblib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]


def module(script: str):
    spec = importlib.util.spec_from_file_location(script.replace('.py', ''), ROOT / 'scripts' / script)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sample():
    rows = []
    for i in range(24):
        for t in (135, 150, 165):
            rows.append({'replay_id': f'r{i}', 'time_sec': t, 'p1_wins': i % 2,
                         'feature_a': i % 2 + t * .0005, 'feature_b': i * .04,
                         'counter_signal': i * .012})
    return pd.DataFrame(rows)


def split():
    return {'seed': 42, 'train_groups': [f'r{i}' for i in range(12)],
            'val_groups': [f'r{i}' for i in range(12, 18)],
            'test_groups': [f'r{i}' for i in range(18, 24)]}


class CheckpointTests(unittest.TestCase):
    def test_rf_opt_in_checkpoint_matches_saved_probabilities(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = sample()
            archive = root / 'tiny.zip'
            with zipfile.ZipFile(archive, 'w') as zf:
                zf.writestr('tiny.csv', data.to_csv(index=False))
            sp = root / 'split.json'
            sp.write_text(json.dumps(split()))
            output = root / 'models'
            cmd = [sys.executable, str(ROOT / 'scripts/54_run_rf_full.py'),
                   '--dataset-zip', str(archive), '--split-json', str(sp),
                   '--dataset-name', 'small', '--profile', 'no_counter',
                   '--fit-partition', 'train', '--output-root', str(output),
                   '--n-estimators', '12', '--n-jobs', '1', '--save-model']
            run = subprocess.run(cmd, capture_output=True, text=True, check=False)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            dest = output / 'small/seed_42'
            clf = joblib.load(dest / 'model.joblib')  # only locally produced trusted checkpoint
            test_data = data[data.replay_id.isin(split()['test_groups'])]
            probs = clf.predict_proba(test_data[['feature_a', 'feature_b']].astype(np.float32))[:, 1]
            actual = pd.read_csv(dest / 'predictions.csv').y_prob.to_numpy()
            np.testing.assert_allclose(probs, actual, atol=1e-12)
            manifest = json.loads((dest / 'artifacts_manifest.json').read_text())
            self.assertIn('model.joblib', manifest['files'])

    def test_deep_selected_checkpoint_and_scaler_present(self):
        deep = module('64_run_deep_finalist.py')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sp = root / 'split.json'
            sp.write_text(json.dumps(split()))
            args = types.SimpleNamespace(dataset_zip='unused.zip', split_json=str(sp),
                                         dataset_name='small', device='cpu',
                                         candidate_set='strict', selection_metric='neg_log_loss',
                                         output_root=str(root / 'deep'), random_state=42)
            candidate = {'name': 'smoke_relu', 'profile': 'full', 'scaler': 'standard',
                         'hidden_dims': [8, 4], 'dropout': .1, 'activation': 'relu',
                         'batch_norm': True, 'lr': .005, 'weight_decay': 0.,
                         'batch_size': 8, 'max_epochs': 2, 'patience': 2,
                         'scheduler': False, 'use_pos_weight': False}
            with patch.object(deep, 'parse_args', return_value=args), \
                 patch.object(deep, 'load_zip_csv', return_value=sample()), \
                 patch.object(deep, 'get_candidates', return_value=[candidate]):
                deep.run()
            dest = root / 'deep/small/seed_42'
            weights = torch.load(dest / 'selected_model_state.pt',
                                 map_location='cpu', weights_only=True)
            self.assertTrue(weights)
            self.assertTrue(all(tensor.device.type == 'cpu' for tensor in weights.values()))
            scaler = joblib.load(dest / 'selected_scaler.joblib')
            self.assertEqual(scaler.n_features_in_, 3)
            artifacts = json.loads((dest / 'artifacts_manifest.json').read_text())
            self.assertTrue((dest / 'selected_scaler.joblib').exists())
            self.assertIn('selected_model_state', artifacts)
            self.assertIn('selected_scaler', artifacts)


if __name__ == '__main__':
    unittest.main()
