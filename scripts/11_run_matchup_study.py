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
from sc2proj.experiment_runner import run_model_target
from sc2proj.metrics import classification_summary
from sc2proj.training_io import load_split_manifest, select_split_frames
from sc2proj.utils import ensure_dir, load_dataframe_from_zip, write_json


MATCHUP_CANDIDATES = ['matchup', 'race_matchup', 'race_pair']


def infer_matchup_column(df: pd.DataFrame) -> str | None:
    for col in MATCHUP_CANDIDATES:
        if col in df.columns:
            return col
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-zip', type=Path, required=True)
    parser.add_argument('--split-json', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=PROJECT_ROOT / 'experiments' / 'matchup')
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

    matchup_col = infer_matchup_column(df)
    if matchup_col is None:
        write_json({'status': 'skipped', 'reason': 'no matchup column found'}, run_dir / 'matchup_status.json')
        return

    result = run_model_target(cfg['model_target'], loaded, seed=seed, cv_scoring=cfg.get('cv_scoring', 'neg_log_loss'), selection_metric=cfg.get('selection_metric', 'neg_log_loss'), device=cfg.get('device', 'cpu'))
    pred_df = pd.DataFrame({'replay_id': loaded.test_df['replay_id'].astype(str), 'y_true': loaded.y_test, 'y_pred': result.y_pred, 'y_prob': result.y_prob, matchup_col: loaded.test_df[matchup_col].values})

    rows = []
    for matchup, part in pred_df.groupby(matchup_col):
        if len(part) < 50:
            continue
        metrics = classification_summary(part['y_true'], part['y_pred'], part['y_prob'])
        metrics.update({'matchup': matchup, 'n_rows': int(len(part)), 'n_replays': int(part['replay_id'].nunique())})
        rows.append(metrics)
    pd.DataFrame(rows).to_csv(run_dir / 'matchup_results.csv', index=False)
    write_json({'matchup_column': matchup_col, 'seed': seed, 'model_target': cfg['model_target']}, run_dir / 'matchup_config.json')


if __name__ == '__main__':
    main()
