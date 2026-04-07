from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sc2proj.config import deep_update, load_yaml
from sc2proj.experiment_runner import run_model_target
from sc2proj.metrics import classification_summary
from sc2proj.training_io import load_split_manifest, select_split_frames
from sc2proj.utils import ensure_dir, load_dataframe_from_zip, write_json


def _clone_loaded_with_filtered_frames(loaded, train_df, val_df, test_df):
    feature_columns = loaded.feature_columns
    return type(loaded)(
        feature_columns=feature_columns,
        X_train=train_df[feature_columns].copy(),
        y_train=train_df['p1_wins'].astype(int).copy(),
        groups_train=train_df['replay_id'].astype(str).copy(),
        X_val=val_df[feature_columns].copy(),
        y_val=val_df['p1_wins'].astype(int).copy(),
        groups_val=val_df['replay_id'].astype(str).copy(),
        X_test=test_df[feature_columns].copy(),
        y_test=test_df['p1_wins'].astype(int).copy(),
        groups_test=test_df['replay_id'].astype(str).copy(),
        train_df=train_df.copy(),
        val_df=val_df.copy(),
        test_df=test_df.copy(),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-zip', type=Path, required=True)
    parser.add_argument('--split-json', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=PROJECT_ROOT / 'experiments' / 'temporal')
    parser.add_argument('--config-yaml', type=Path, default=PROJECT_ROOT / 'configs' / 'experiments' / 'temporal.yaml')
    parser.add_argument('--model-target', choices=['logreg', 'rf', 'xgb', 'mlp', 'mlp_torch'], default=None)
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

    horizons = list(cfg.get('horizons_sec', [180, 300, 420, 600, 900]))
    horizon_rows = []
    for horizon in horizons:
        train_df = loaded.train_df[loaded.train_df['time_sec'] <= horizon].copy()
        val_df = loaded.val_df[loaded.val_df['time_sec'] <= horizon].copy()
        test_df = loaded.test_df[loaded.test_df['time_sec'] <= horizon].copy()
        if min(len(train_df), len(val_df), len(test_df)) == 0:
            continue
        sub_loaded = _clone_loaded_with_filtered_frames(loaded, train_df, val_df, test_df)
        result = run_model_target(cfg['model_target'], sub_loaded, seed=seed, cv_scoring=cfg.get('cv_scoring', 'neg_log_loss'), selection_metric=cfg.get('selection_metric', 'neg_log_loss'), device=cfg.get('device', 'cpu'))
        horizon_rows.append({'horizon_sec': horizon, 'n_train_rows': len(train_df), 'n_test_rows': len(test_df), **result.metrics})

    pred_result = run_model_target(cfg['model_target'], loaded, seed=seed, cv_scoring=cfg.get('cv_scoring', 'neg_log_loss'), selection_metric=cfg.get('selection_metric', 'neg_log_loss'), device=cfg.get('device', 'cpu'))
    pred_df = pd.DataFrame({'replay_id': loaded.test_df['replay_id'].astype(str), 'time_sec': loaded.test_df['time_sec'], 'y_true': loaded.y_test, 'y_pred': pred_result.y_pred, 'y_prob': pred_result.y_prob})
    durations = pred_df.groupby('replay_id')['time_sec'].max().rename('match_length_sec')
    pred_df = pred_df.join(durations, on='replay_id')

    bins = list(cfg.get('length_bins_sec', [300, 600, 900, 1200]))
    labels = []
    start = 0
    cut_bins = []
    for b in bins:
        cut_bins.append(b)
        labels.append(f'{start}-{b}s')
        start = b
    cut_bins.append(np.inf)
    labels.append(f'{bins[-1]}+s')
    pred_df['length_bin'] = pd.cut(pred_df['match_length_sec'], bins=[0, *bins, np.inf], labels=labels, right=True, include_lowest=True)

    length_rows = []
    for length_bin, part in pred_df.groupby('length_bin', observed=False):
        if part.empty:
            continue
        metrics = classification_summary(part['y_true'], part['y_pred'], part['y_prob'])
        metrics.update({'length_bin': str(length_bin), 'n_rows': int(len(part)), 'n_replays': int(part['replay_id'].nunique())})
        length_rows.append(metrics)

    pd.DataFrame(horizon_rows).to_csv(run_dir / 'temporal_horizon_results.csv', index=False)
    pd.DataFrame(length_rows).to_csv(run_dir / 'match_length_results.csv', index=False)
    write_json({'seed': seed, 'model_target': cfg['model_target'], 'horizons_sec': horizons, 'length_bins_sec': bins}, run_dir / 'temporal_config.json')


if __name__ == '__main__':
    main()
