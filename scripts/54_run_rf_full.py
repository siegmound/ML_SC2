#!/usr/bin/env python3
import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, log_loss, confusion_matrix


def load_dataset_from_zip(zip_path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path, 'r') as zf:
        names = [n for n in zf.namelist() if n.lower().endswith('.csv')]
        if not names:
            raise FileNotFoundError(f'No CSV found inside {zip_path}')
        with zf.open(names[0]) as f:
            df = pd.read_csv(f)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df


def infer_profile_columns(columns, profile):
    base_drop = {'p1_wins', 'replay_id', 'time_sec'}
    feats = [c for c in columns if c not in base_drop]
    if profile == 'full':
        return feats
    if profile == 'no_counter':
        return [c for c in feats if 'counter' not in c.lower()]
    if profile == 'no_counter_no_losses':
        out = []
        for c in feats:
            cl = c.lower()
            if 'counter' in cl:
                continue
            if 'loss' in cl or 'recently_lost' in cl:
                continue
            out.append(c)
        return out
    raise ValueError(f'Unknown profile: {profile}')


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2)


def run():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset-zip', required=True)
    ap.add_argument('--split-json', required=True)
    ap.add_argument('--dataset-name', required=True)
    ap.add_argument('--profile', choices=['full', 'no_counter', 'no_counter_no_losses'], default='full')
    ap.add_argument('--n-estimators', type=int, default=700)
    ap.add_argument('--max-depth', type=int, default=24)
    ap.add_argument('--min-samples-split', type=int, default=4)
    ap.add_argument('--min-samples-leaf', type=int, default=2)
    ap.add_argument('--max-features', default='sqrt')
    ap.add_argument('--class-weight', choices=['none', 'balanced', 'balanced_subsample'], default='balanced_subsample')
    ap.add_argument('--n-jobs', type=int, default=-1)
    ap.add_argument('--random-state', type=int, default=42)
    args = ap.parse_args()

    dataset_zip = Path(args.dataset_zip)
    split_json = Path(args.split_json)
    out_dir = Path('results') / 'rf_full' / args.dataset_name

    print(f'Loading dataset from {dataset_zip} ...', flush=True)
    df = load_dataset_from_zip(dataset_zip)
    print(f'Loaded rows={len(df)} unique_replays={df["replay_id"].nunique()}', flush=True)

    with open(split_json, 'r', encoding='utf-8') as f:
        split = json.load(f)

    seed = int(split.get('seed', args.random_state))
    train_ids = set(split['train_groups'])
    val_ids = set(split['val_groups'])
    test_ids = set(split['test_groups'])

    train_df = df[df['replay_id'].isin(train_ids)].copy()
    val_df = df[df['replay_id'].isin(val_ids)].copy()
    test_df = df[df['replay_id'].isin(test_ids)].copy()

    print(f'Split rows train={len(train_df)} val={len(val_df)} test={len(test_df)}', flush=True)

    feature_cols = infer_profile_columns(df.columns, args.profile)
    print(f'Profile={args.profile} n_features={len(feature_cols)}', flush=True)

    X_train = train_df[feature_cols].astype(np.float32)
    y_train = train_df['p1_wins'].astype(np.int32)
    X_val = val_df[feature_cols].astype(np.float32)
    y_val = val_df['p1_wins'].astype(np.int32)
    X_test = test_df[feature_cols].astype(np.float32)
    y_test = test_df['p1_wins'].astype(np.int32)

    class_weight = None if args.class_weight == 'none' else args.class_weight

    # Train on train+val for final full-data seed evaluation, using chosen hyperparams.
    X_train_full = pd.concat([X_train, X_val], axis=0)
    y_train_full = pd.concat([y_train, y_val], axis=0)

    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        class_weight=class_weight,
        n_jobs=args.n_jobs,
        random_state=seed,
    )

    clf.fit(X_train_full, y_train_full)

    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(np.int32)

    metrics = {
        'dataset_name': args.dataset_name,
        'model_name': 'rf_full',
        'profile': args.profile,
        'seed': seed,
        'n_rows_train': int(len(X_train)),
        'n_rows_val': int(len(X_val)),
        'n_rows_test': int(len(X_test)),
        'n_rows_train_full': int(len(X_train_full)),
        'n_features': int(len(feature_cols)),
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_test, y_pred)),
        'roc_auc': float(roc_auc_score(y_test, y_prob)),
        'log_loss': float(log_loss(y_test, y_prob)),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'params': {
            'n_estimators': args.n_estimators,
            'max_depth': args.max_depth,
            'min_samples_split': args.min_samples_split,
            'min_samples_leaf': args.min_samples_leaf,
            'max_features': args.max_features,
            'class_weight': class_weight,
            'n_jobs': args.n_jobs,
        },
    }

    run_dir = out_dir / f'seed_{seed}'
    run_dir.mkdir(parents=True, exist_ok=True)

    pred_df = pd.DataFrame({
        'replay_id': test_df['replay_id'].values,
        'time_sec': test_df['time_sec'].values if 'time_sec' in test_df.columns else np.nan,
        'y_true': y_test.values,
        'y_prob': y_prob,
        'y_pred': y_pred,
    })
    pred_df.to_csv(run_dir / 'predictions.csv', index=False)

    fi = pd.DataFrame({
        'feature': feature_cols,
        'importance': clf.feature_importances_,
    }).sort_values('importance', ascending=False)
    fi.to_csv(run_dir / 'feature_importance.csv', index=False)

    save_json(run_dir / 'metrics_summary.json', metrics)
    save_json(run_dir / 'config_used.json', {
        'dataset_zip': str(dataset_zip),
        'split_json': str(split_json),
        'dataset_name': args.dataset_name,
        'profile': args.profile,
        'seed': seed,
        **metrics['params'],
    })
    save_json(run_dir / 'artifacts_manifest.json', {
        'files': [
            'metrics_summary.json',
            'config_used.json',
            'artifacts_manifest.json',
            'predictions.csv',
            'feature_importance.csv',
        ]
    })

    print(json.dumps({
        'seed': seed,
        'profile': args.profile,
        'accuracy': metrics['accuracy'],
        'balanced_accuracy': metrics['balanced_accuracy'],
        'roc_auc': metrics['roc_auc'],
        'log_loss': metrics['log_loss'],
        'output_dir': str(run_dir),
    }, indent=2), flush=True)


if __name__ == '__main__':
    run()
