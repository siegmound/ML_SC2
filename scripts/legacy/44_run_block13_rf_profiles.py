from __future__ import annotations

import argparse
from pathlib import Path
import sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sc2proj.feature_registry import build_feature_family_map
from sc2proj.metrics import classification_summary
from sc2proj.training_io import load_split_manifest, select_split_frames, downsample_loaded_split
from sc2proj.utils import ensure_dir, load_dataframe_from_zip, write_json

PROFILE_DEFS = {
    'full': {'include': None, 'exclude': []},
    'no_counter': {'include': None, 'exclude': ['counter']},
    'no_losses': {'include': None, 'exclude': ['losses']},
    'no_counter_no_losses': {'include': None, 'exclude': ['counter', 'losses']},
    'economy_only': {'include': ['economy'], 'exclude': []},
    'economy_scouting': {'include': ['economy', 'scouting'], 'exclude': []},
    'economy_scouting_combat': {'include': ['economy', 'scouting', 'combat'], 'exclude': []},
    'economy_scouting_composition': {'include': ['economy', 'scouting', 'composition'], 'exclude': []},
}


def choose_features(feature_columns: list[str], profile: str) -> tuple[list[str], dict[str, list[str]]]:
    family_map = build_feature_family_map(feature_columns)
    spec = PROFILE_DEFS[profile]
    if spec['include']:
        selected = []
        for fam in spec['include']:
            selected.extend(family_map.get(fam, []))
    else:
        selected = list(feature_columns)
    if spec['exclude']:
        excluded = set()
        for fam in spec['exclude']:
            excluded.update(family_map.get(fam, []))
        selected = [c for c in selected if c not in excluded]
    return selected, family_map


def fit_eval(loaded, feature_columns: list[str], seed: int, params: dict, n_jobs: int):
    clf = RandomForestClassifier(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        min_samples_split=params['min_samples_split'],
        min_samples_leaf=params['min_samples_leaf'],
        max_features=params['max_features'],
        class_weight=params['class_weight'],
        random_state=seed,
        n_jobs=n_jobs,
    )
    clf.fit(loaded.X_train[feature_columns], loaded.y_train)
    y_prob = clf.predict_proba(loaded.X_test[feature_columns])[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = classification_summary(loaded.y_test, y_pred, y_prob)
    metrics['n_features'] = len(feature_columns)
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset-zip', type=Path, required=True)
    ap.add_argument('--split-json', type=Path, required=True)
    ap.add_argument('--dataset-name', required=True)
    ap.add_argument('--profiles', nargs='+', default=['full', 'no_counter', 'no_losses', 'no_counter_no_losses', 'economy_only', 'economy_scouting', 'economy_scouting_combat'])
    ap.add_argument('--output-dir', type=Path, default=PROJECT_ROOT / 'results' / 'block13_rf_profiles')
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--n-estimators', type=int, default=300)
    ap.add_argument('--max-depth', type=int, default=20)
    ap.add_argument('--min-samples-split', type=int, default=5)
    ap.add_argument('--min-samples-leaf', type=int, default=2)
    ap.add_argument('--max-features', default='sqrt')
    ap.add_argument('--class-weight', default='none', choices=['none', 'balanced', 'balanced_subsample'])
    ap.add_argument('--max-train-rows', type=int, default=30000)
    ap.add_argument('--max-val-rows', type=int, default=12000)
    ap.add_argument('--max-test-rows', type=int, default=None)
    ap.add_argument('--n-jobs', type=int, default=-1)
    args = ap.parse_args()

    df = load_dataframe_from_zip(args.dataset_zip)
    split = load_split_manifest(args.split_json)
    seed = int(args.seed if args.seed is not None else split['seed'])
    loaded = select_split_frames(df, split)
    loaded = downsample_loaded_split(loaded, seed=seed, max_train_rows=args.max_train_rows, max_val_rows=args.max_val_rows, max_test_rows=args.max_test_rows)
    class_weight = None if args.class_weight == 'none' else args.class_weight
    params = {
        'n_estimators': args.n_estimators,
        'max_depth': args.max_depth,
        'min_samples_split': args.min_samples_split,
        'min_samples_leaf': args.min_samples_leaf,
        'max_features': args.max_features,
        'class_weight': class_weight,
    }

    run_dir = args.output_dir / args.dataset_name / f'seed_{seed}'
    ensure_dir(run_dir)
    rows = []
    profile_feature_counts = {}
    family_map_ref = None
    full_metrics = None
    for profile in args.profiles:
        feats, family_map = choose_features(loaded.feature_columns, profile)
        family_map_ref = family_map
        profile_feature_counts[profile] = len(feats)
        metrics = fit_eval(loaded, feats, seed, params, args.n_jobs)
        row = {'profile': profile, **metrics}
        if profile == 'full':
            full_metrics = metrics
        rows.append(row)
        print(f"profile={profile} acc={metrics['accuracy']:.4f} auc={metrics['roc_auc']:.4f} ll={metrics['log_loss']:.4f} n_features={len(feats)}")

    out = pd.DataFrame(rows)
    if full_metrics is not None:
        for metric_name in ['accuracy', 'balanced_accuracy', 'roc_auc', 'log_loss']:
            base = float(full_metrics[metric_name])
            delta_col = f'delta_{metric_name}_vs_full'
            if metric_name == 'log_loss':
                out[delta_col] = out[metric_name] - base
            else:
                out[delta_col] = out[metric_name] - base
    out.to_csv(run_dir / 'rf_profile_comparison.csv', index=False)
    out.sort_values(['log_loss', 'roc_auc'], ascending=[True, False]).to_csv(run_dir / 'rf_profile_ranking.csv', index=False)
    write_json({
        'dataset_name': args.dataset_name,
        'seed': seed,
        'rf_params': params,
        'profile_feature_counts': profile_feature_counts,
        'family_map_counts': {k: len(v) for k, v in (family_map_ref or {}).items()},
        'downsample': {'train': args.max_train_rows, 'val': args.max_val_rows, 'test': args.max_test_rows},
    }, run_dir / 'rf_profile_config.json')


if __name__ == '__main__':
    main()
