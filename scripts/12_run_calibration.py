from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sc2proj.config import deep_update, load_yaml
from sc2proj.experiment_runner import run_model_target
from sc2proj.metrics import classification_summary
from sc2proj.modeling import make_numeric_preprocessor, threshold_predictions
from sc2proj.training_io import load_split_manifest, select_split_frames
from sc2proj.utils import ensure_dir, load_dataframe_from_zip


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ids = np.digitize(y_prob, bins) - 1
    ece = 0.0
    for b in range(n_bins):
        mask = ids == b
        if not np.any(mask):
            continue
        acc = float(np.mean(y_true[mask] == (y_prob[mask] >= 0.5).astype(int)))
        conf = float(np.mean(y_prob[mask]))
        ece += (np.sum(mask) / len(y_true)) * abs(acc - conf)
    return float(ece)


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((y_prob - y_true) ** 2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-zip', type=Path, required=True)
    parser.add_argument('--split-json', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=PROJECT_ROOT / 'experiments' / 'calibration')
    parser.add_argument('--config-yaml', type=Path, default=PROJECT_ROOT / 'configs' / 'experiments' / 'calibration.yaml')
    parser.add_argument('--model-target', choices=['logreg', 'rf', 'xgb', 'mlp'], default=None)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config_yaml)
    overrides = {k: v for k, v in {'model_target': args.model_target, 'device': args.device}.items() if v is not None}
    cfg = deep_update(cfg, overrides)

    df = load_dataframe_from_zip(args.dataset_zip)
    split = load_split_manifest(args.split_json)
    loaded = select_split_frames(df, split)
    seed = int(split['seed'])
    run_dir = args.output_dir / cfg['model_target'] / f'seed_{seed}'
    ensure_dir(run_dir)

    if cfg['model_target'] == 'logreg':
        base = Pipeline([('prep', make_numeric_preprocessor('standard')), ('clf', LogisticRegression(max_iter=1000, random_state=seed))])
        base.fit(loaded.X_train, loaded.y_train)
        methods = {'uncalibrated': base, 'platt': CalibratedClassifierCV(base, method='sigmoid', cv='prefit'), 'isotonic': CalibratedClassifierCV(base, method='isotonic', cv='prefit')}
        rows = []
        curves = []
        for name, model in methods.items():
            if name != 'uncalibrated':
                model.fit(loaded.X_val, loaded.y_val)
            y_prob = model.predict_proba(loaded.X_test)[:, 1]
            y_pred = threshold_predictions(y_prob)
            metrics = classification_summary(loaded.y_test, y_pred, y_prob)
            metrics.update({'method': name, 'brier_score': brier_score(loaded.y_test.to_numpy(), y_prob), 'ece_10': expected_calibration_error(loaded.y_test.to_numpy(), y_prob, 10)})
            rows.append(metrics)
            bin_edges = np.linspace(0.0, 1.0, 11)
            bin_ids = np.digitize(y_prob, bin_edges) - 1
            for b in range(10):
                mask = bin_ids == b
                if not np.any(mask):
                    continue
                curves.append({'method': name, 'bin_index': b, 'mean_prob': float(np.mean(y_prob[mask])), 'empirical_rate': float(np.mean(loaded.y_test.to_numpy()[mask])), 'count': int(np.sum(mask))})
        pd.DataFrame(rows).to_csv(run_dir / 'calibration_summary.csv', index=False)
        pd.DataFrame(curves).to_csv(run_dir / 'reliability_curve.csv', index=False)
        return

    result = run_model_target(cfg['model_target'], loaded, seed=seed, cv_scoring=cfg.get('cv_scoring', 'neg_log_loss'), selection_metric=cfg.get('selection_metric', 'neg_log_loss'), device=cfg.get('device', 'cpu'))
    y_prob = result.y_prob
    y_pred = result.y_pred
    metrics = classification_summary(loaded.y_test, y_pred, y_prob)
    metrics.update({'method': 'uncalibrated_only', 'brier_score': brier_score(loaded.y_test.to_numpy(), y_prob), 'ece_10': expected_calibration_error(loaded.y_test.to_numpy(), y_prob, 10), 'model_target': cfg['model_target']})
    pd.DataFrame([metrics]).to_csv(run_dir / 'calibration_summary.csv', index=False)


if __name__ == '__main__':
    main()
