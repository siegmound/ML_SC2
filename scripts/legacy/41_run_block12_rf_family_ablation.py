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
from sc2proj.training_io import load_split_manifest, select_split_frames
from sc2proj.utils import ensure_dir, load_dataframe_from_zip, write_json


def fit_eval(loaded, feature_columns: list[str], seed: int, n_estimators: int, max_depth: int, min_samples_split: int, min_samples_leaf: int, max_features: str, class_weight: str | None, n_jobs: int):
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight=class_weight,
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-zip', type=Path, required=True)
    parser.add_argument('--split-json', type=Path, required=True)
    parser.add_argument('--dataset-name', required=True)
    parser.add_argument('--output-dir', type=Path, default=PROJECT_ROOT / 'experiments' / 'block12_rf_ablation')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--n-estimators', type=int, default=30)
    parser.add_argument('--max-depth', type=int, default=20)
    parser.add_argument('--min-samples-split', type=int, default=5)
    parser.add_argument('--min-samples-leaf', type=int, default=2)
    parser.add_argument('--max-features', default='sqrt')
    parser.add_argument('--class-weight', default='none', choices=['none', 'balanced', 'balanced_subsample'])
    parser.add_argument('--n-jobs', type=int, default=-1)
    args = parser.parse_args()

    df = load_dataframe_from_zip(args.dataset_zip)
    split = load_split_manifest(args.split_json)
    seed = int(args.seed if args.seed is not None else split['seed'])
    loaded = select_split_frames(df, split)
    family_map = build_feature_family_map(loaded.feature_columns)
    family_order = [f for f in sorted(family_map) if f != 'metadata']
    class_weight = None if args.class_weight == 'none' else args.class_weight

    run_dir = args.output_dir / args.dataset_name / f'seed_{seed}'
    ensure_dir(run_dir)
    rows = []

    full_metrics = fit_eval(loaded, loaded.feature_columns, seed, args.n_estimators, args.max_depth, args.min_samples_split, args.min_samples_leaf, args.max_features, class_weight, args.n_jobs)
    rows.append({'study_mode': 'full', 'family': 'ALL', **full_metrics})

    for fam in family_order:
        feature_subset = [c for c in loaded.feature_columns if c not in set(family_map.get(fam, []))]
        metrics = fit_eval(loaded, feature_subset, seed, args.n_estimators, args.max_depth, args.min_samples_split, args.min_samples_leaf, args.max_features, class_weight, args.n_jobs)
        delta_acc = metrics['accuracy'] - full_metrics['accuracy']
        delta_auc = metrics['roc_auc'] - full_metrics['roc_auc']
        delta_ll = metrics['log_loss'] - full_metrics['log_loss']
        rows.append({'study_mode': 'leave_one_family_out', 'family': fam, 'left_out_features': len(family_map.get(fam, [])), 'delta_accuracy_vs_full': delta_acc, 'delta_roc_auc_vs_full': delta_auc, 'delta_log_loss_vs_full': delta_ll, **metrics})

    out = pd.DataFrame(rows)
    out.to_csv(run_dir / 'rf_family_ablation.csv', index=False)
    ranking = out[out['study_mode'] == 'leave_one_family_out'].sort_values('delta_log_loss_vs_full', ascending=False)
    ranking.to_csv(run_dir / 'rf_family_ablation_ranked.csv', index=False)
    write_json({'dataset_name': args.dataset_name, 'seed': seed, 'rf_params': {'n_estimators': args.n_estimators, 'max_depth': args.max_depth, 'min_samples_split': args.min_samples_split, 'min_samples_leaf': args.min_samples_leaf, 'max_features': args.max_features, 'class_weight': class_weight}, 'family_map_counts': {k: len(v) for k, v in family_map.items()}}, run_dir / 'rf_family_ablation_config.json')


if __name__ == '__main__':
    main()
