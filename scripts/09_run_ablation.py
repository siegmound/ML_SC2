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
from sc2proj.feature_registry import build_feature_family_map, select_features_by_family
from sc2proj.training_io import load_split_manifest, select_split_frames
from sc2proj.utils import ensure_dir, load_dataframe_from_zip, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-zip', type=Path, required=True)
    parser.add_argument('--split-json', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=PROJECT_ROOT / 'experiments' / 'ablations')
    parser.add_argument('--config-yaml', type=Path, default=PROJECT_ROOT / 'configs' / 'experiments' / 'ablation.yaml')
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

    family_map = build_feature_family_map(loaded.feature_columns)
    rows = []
    search_rows = []

    full_result = run_model_target(cfg['model_target'], loaded, seed=seed, cv_scoring=cfg.get('cv_scoring', 'neg_log_loss'), selection_metric=cfg.get('selection_metric', 'neg_log_loss'), device=cfg.get('device', 'cpu'))
    rows.append({'study_mode': 'full', 'family': 'ALL', 'n_features': len(loaded.feature_columns), **full_result.metrics})
    for r in full_result.search_rows:
        search_rows.append({'study_mode': 'full', 'family': 'ALL', **r})

    cumulative_order = [fam for fam in family_map.keys() if fam != 'metadata']
    selected_so_far: list[str] = []
    for fam in cumulative_order:
        selected_so_far.extend(family_map[fam])
        result = run_model_target(cfg['model_target'], loaded, seed=seed, cv_scoring=cfg.get('cv_scoring', 'neg_log_loss'), selection_metric=cfg.get('selection_metric', 'neg_log_loss'), device=cfg.get('device', 'cpu'), feature_columns=selected_so_far)
        rows.append({'study_mode': 'cumulative_add', 'family': fam, 'n_features': len(selected_so_far), **result.metrics})
        for r in result.search_rows:
            search_rows.append({'study_mode': 'cumulative_add', 'family': fam, **r})

    for fam in cumulative_order:
        feature_subset = select_features_by_family(loaded.feature_columns, exclude_families=[fam])
        result = run_model_target(cfg['model_target'], loaded, seed=seed, cv_scoring=cfg.get('cv_scoring', 'neg_log_loss'), selection_metric=cfg.get('selection_metric', 'neg_log_loss'), device=cfg.get('device', 'cpu'), feature_columns=feature_subset)
        rows.append({'study_mode': 'leave_one_family_out', 'family': fam, 'n_features': len(feature_subset), **result.metrics})
        for r in result.search_rows:
            search_rows.append({'study_mode': 'leave_one_family_out', 'family': fam, **r})

    pd.DataFrame(rows).to_csv(run_dir / 'ablation_results.csv', index=False)
    pd.DataFrame(search_rows).to_csv(run_dir / 'ablation_search_details.csv', index=False)
    write_json({'seed': seed, 'model_target': cfg['model_target'], 'cv_scoring': cfg.get('cv_scoring'), 'selection_metric': cfg.get('selection_metric'), 'device': cfg.get('device', 'cpu'), 'feature_families': family_map}, run_dir / 'ablation_config.json')


if __name__ == '__main__':
    main()
