from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sc2proj.config import deep_update, load_yaml
from sc2proj.error_analysis import build_prediction_frame, probability_flip_summary, replay_level_error_summary
from sc2proj.experiment_runner import run_model_target
from sc2proj.training_io import load_split_manifest, select_split_frames
from sc2proj.utils import ensure_dir, load_dataframe_from_zip, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-zip', type=Path, required=True)
    parser.add_argument('--split-json', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=PROJECT_ROOT / 'experiments' / 'error_analysis')
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

    result = run_model_target(cfg['model_target'], loaded, seed=seed, cv_scoring=cfg.get('cv_scoring', 'neg_log_loss'), selection_metric=cfg.get('selection_metric', 'neg_log_loss'), device=cfg.get('device', 'cpu'))
    pred_df = build_prediction_frame(loaded.test_df, result.y_pred, result.y_prob)
    replay_df = replay_level_error_summary(pred_df)
    flip_df = probability_flip_summary(pred_df)

    pred_df.to_csv(run_dir / 'prediction_frame.csv', index=False)
    replay_df.to_csv(run_dir / 'replay_error_summary.csv', index=False)
    flip_df.to_csv(run_dir / 'probability_flip_summary.csv', index=False)

    top_fp = pred_df[(pred_df['y_true'] == 0) & (pred_df['y_pred'] == 1)].sort_values('confidence', ascending=False).head(50)
    top_fn = pred_df[(pred_df['y_true'] == 1) & (pred_df['y_pred'] == 0)].sort_values('confidence', ascending=False).head(50)
    uncertain = pred_df.sort_values('uncertainty', ascending=True).head(50) if 'uncertainty' in pred_df.columns else pred_df.head(50)
    top_fp.to_csv(run_dir / 'top_false_positives.csv', index=False)
    top_fn.to_csv(run_dir / 'top_false_negatives.csv', index=False)
    uncertain.to_csv(run_dir / 'most_uncertain_cases.csv', index=False)

    write_json({'seed': seed, 'model_target': cfg['model_target'], 'n_predictions': int(len(pred_df)), 'n_replay_rows': int(len(replay_df))}, run_dir / 'error_analysis_summary.json')


if __name__ == '__main__':
    main()
