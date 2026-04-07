import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, log_loss, confusion_matrix

TARGET = 'p1_wins'
GROUP = 'replay_id'
TIME = 'time_sec'
DROP_ALWAYS = {TARGET, GROUP, TIME}
COUNTER_TOKENS = ('counter',)
LOSSES_TOKENS = ('loss', 'killed', 'destroyed', 'recent_loss')


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset-zip', required=True)
    ap.add_argument('--split-json', required=True)
    ap.add_argument('--dataset-name', required=True)
    ap.add_argument('--profile', choices=['full', 'no_counter', 'no_counter_no_losses'], default='no_counter')
    ap.add_argument('--output-root', default='results/rf_calibration')
    ap.add_argument('--random-state', type=int, default=42)
    ap.add_argument('--n-estimators', type=int, default=700)
    ap.add_argument('--max-depth', type=int, default=24)
    ap.add_argument('--min-samples-split', type=int, default=4)
    ap.add_argument('--min-samples-leaf', type=int, default=2)
    ap.add_argument('--max-features', default='sqrt')
    ap.add_argument('--class-weight', choices=['none', 'balanced', 'balanced_subsample'], default='balanced_subsample')
    ap.add_argument('--n-jobs', type=int, default=-1)
    return ap.parse_args()


def load_zip_csv(zip_path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path, 'r') as zf:
        names = [n for n in zf.namelist() if n.lower().endswith('.csv')]
        if not names:
            raise FileNotFoundError(f'No CSV found in {zip_path}')
        with zf.open(names[0]) as f:
            return pd.read_csv(f)


def select_features(columns, profile):
    feats = [c for c in columns if c not in DROP_ALWAYS]
    if profile == 'full':
        return feats
    out = []
    for c in feats:
        cl = c.lower()
        if profile in ('no_counter', 'no_counter_no_losses') and any(tok in cl for tok in COUNTER_TOKENS):
            continue
        if profile == 'no_counter_no_losses' and any(tok in cl for tok in LOSSES_TOKENS):
            continue
        out.append(c)
    return out


def make_frame(df, y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(np.int32)
    return pd.DataFrame({
        GROUP: df[GROUP].values,
        TIME: df[TIME].values,
        'y_true': y_true.values,
        'y_prob': y_prob,
        'y_pred': y_pred,
    })


def metrics_from_probs(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(np.int32)
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'roc_auc': float(roc_auc_score(y_true, y_prob)),
        'log_loss': float(log_loss(y_true, y_prob)),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
    }


def run():
    args = parse_args()
    dataset_zip = Path(args.dataset_zip)
    split_json = Path(args.split_json)

    with open(split_json, 'r', encoding='utf-8') as f:
        split = json.load(f)
    seed = split.get('seed', args.random_state)

    out_dir = Path(args.output_root) / args.dataset_name / f'seed_{seed}'
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'Loading dataset from {dataset_zip} ...')
    df = load_zip_csv(dataset_zip)
    df = df.replace([np.inf, -np.inf], np.nan).dropna().copy()
    print(f'Loaded rows={len(df)} unique_replays={df[GROUP].nunique()}')

    train_groups = set(split['train_groups'])
    val_groups = set(split['val_groups'])
    test_groups = set(split['test_groups'])

    train_df = df[df[GROUP].isin(train_groups)].copy()
    val_df = df[df[GROUP].isin(val_groups)].copy()
    test_df = df[df[GROUP].isin(test_groups)].copy()
    print(f'Split rows train={len(train_df)} val={len(val_df)} test={len(test_df)}')

    features = select_features(df.columns, args.profile)
    print(f'Profile={args.profile} n_features={len(features)}')

    X_train = train_df[features].astype(np.float32)
    y_train = train_df[TARGET].astype(np.int32)
    X_val = val_df[features].astype(np.float32)
    y_val = val_df[TARGET].astype(np.int32)
    X_test = test_df[features].astype(np.float32)
    y_test = test_df[TARGET].astype(np.int32)

    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=None if args.max_depth in ('none', None) else int(args.max_depth),
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        class_weight=None if args.class_weight == 'none' else args.class_weight,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
    )
    clf.fit(X_train, y_train)

    val_prob = clf.predict_proba(X_val)[:, 1]
    test_prob = clf.predict_proba(X_test)[:, 1]

    val_preds = make_frame(val_df, y_val, val_prob)
    test_preds = make_frame(test_df, y_test, test_prob)

    val_metrics = metrics_from_probs(y_val, val_prob)
    test_metrics = metrics_from_probs(y_test, test_prob)

    fi = pd.DataFrame({'feature': features, 'importance': clf.feature_importances_})
    fi = fi.sort_values('importance', ascending=False).reset_index(drop=True)

    summary = {
        'dataset_name': args.dataset_name,
        'seed': seed,
        'model': 'rf',
        'profile': args.profile,
        'n_rows_train': int(len(train_df)),
        'n_rows_val': int(len(val_df)),
        'n_rows_test': int(len(test_df)),
        'n_features': int(len(features)),
        'validation_metrics': val_metrics,
        'test_metrics': test_metrics,
        'rf_params': {
            'n_estimators': args.n_estimators,
            'max_depth': args.max_depth,
            'min_samples_split': args.min_samples_split,
            'min_samples_leaf': args.min_samples_leaf,
            'max_features': args.max_features,
            'class_weight': args.class_weight,
        }
    }

    with open(out_dir / 'metrics_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    val_preds.to_csv(out_dir / 'validation_predictions.csv', index=False)
    test_preds.to_csv(out_dir / 'test_predictions.csv', index=False)
    fi.to_csv(out_dir / 'feature_importance.csv', index=False)
    with open(out_dir / 'config_used.json', 'w', encoding='utf-8') as f:
        json.dump({'profile': args.profile, 'features': features}, f, indent=2)
    with open(out_dir / 'artifacts_manifest.json', 'w', encoding='utf-8') as f:
        json.dump({
            'metrics_summary': str(out_dir / 'metrics_summary.json'),
            'validation_predictions': str(out_dir / 'validation_predictions.csv'),
            'test_predictions': str(out_dir / 'test_predictions.csv'),
            'feature_importance': str(out_dir / 'feature_importance.csv'),
            'config_used': str(out_dir / 'config_used.json')
        }, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f'Saved outputs to {out_dir}')


if __name__ == '__main__':
    run()
