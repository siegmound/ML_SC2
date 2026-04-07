
import argparse
import json
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
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
    ap.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    ap.add_argument('--profile', choices=['full','no_counter','no_counter_no_losses'], default='full')
    ap.add_argument('--output-root', default='results/xgb_full')
    ap.add_argument('--random-state', type=int, default=42)
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


def run():
    args = parse_args()
    dataset_zip = Path(args.dataset_zip)
    split_json = Path(args.split_json)
    # Read split first so outputs are written to the correct seed directory.
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
    print(f'Split rows train={len(train_df)} val={len(val_df)} test={len(test_df)} device={args.device}')

    features = select_features(df.columns, args.profile)
    print(f'Profile={args.profile} n_features={len(features)}')

    X_train = train_df[features].astype(np.float32)
    y_train = train_df[TARGET].astype(np.int32)
    X_val = val_df[features].astype(np.float32)
    y_val = val_df[TARGET].astype(np.int32)
    X_test = test_df[features].astype(np.float32)
    y_test = test_df[TARGET].astype(np.int32)

    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'random_state': args.random_state,
        'max_depth': 5,
        'learning_rate': 0.03,
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_lambda': 1.0,
        'tree_method': 'hist',
        'device': args.device,
    }

    if args.device == 'cuda':
        dtrain = xgb.QuantileDMatrix(X_train, y_train)
        dval = xgb.QuantileDMatrix(X_val, y_val, ref=dtrain)
        dtest = xgb.QuantileDMatrix(X_test, y_test, ref=dtrain)
    else:
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        dtest = xgb.DMatrix(X_test, label=y_test)

    evals_result = {}
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=2000,
        evals=[(dval, 'validation')],
        early_stopping_rounds=50,
        evals_result=evals_result,
        verbose_eval=False,
    )

    y_prob = booster.predict(dtest, iteration_range=(0, booster.best_iteration + 1 if booster.best_iteration is not None else 0))
    y_pred = (y_prob >= 0.5).astype(np.int32)

    metrics = {
        'dataset_name': args.dataset_name,
        'seed': seed,
        'model': 'xgb_full',
        'profile': args.profile,
        'device': args.device,
        'n_rows_train': int(len(train_df)),
        'n_rows_val': int(len(val_df)),
        'n_rows_test': int(len(test_df)),
        'n_features': int(len(features)),
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_test, y_pred)),
        'roc_auc': float(roc_auc_score(y_test, y_prob)),
        'log_loss': float(log_loss(y_test, y_prob)),
        'best_iteration': int(booster.best_iteration if booster.best_iteration is not None else booster.num_boosted_rounds()-1),
        'best_score': float(booster.best_score) if getattr(booster, 'best_score', None) is not None else None,
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
    }

    preds = pd.DataFrame({
        GROUP: test_df[GROUP].values,
        TIME: test_df[TIME].values,
        'y_true': y_test.values,
        'y_prob': y_prob,
        'y_pred': y_pred,
    })
    raw_gain = booster.get_score(importance_type='gain')
    fi = pd.DataFrame({
        'feature': list(raw_gain.keys()),
        'importance': [float(v) for v in raw_gain.values()],
    })
    if not fi.empty:
        fi = fi.groupby('feature', as_index=False)['importance'].sum()
    existing = set(fi['feature']) if not fi.empty else set()
    missing = [f for f in features if f not in existing]
    if missing:
        fi = pd.concat([fi, pd.DataFrame({'feature': missing, 'importance': [0.0]*len(missing)})], ignore_index=True)
    fi = fi.sort_values('importance', ascending=False).reset_index(drop=True)

    with open(out_dir/'metrics_summary.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    preds.to_csv(out_dir/'predictions.csv', index=False)
    fi.to_csv(out_dir/'feature_importance.csv', index=False)
    with open(out_dir/'config_used.json', 'w', encoding='utf-8') as f:
        json.dump({'params': params, 'profile': args.profile, 'features': features}, f, indent=2)
    with open(out_dir/'artifacts_manifest.json', 'w', encoding='utf-8') as f:
        json.dump({
            'metrics_summary': str(out_dir/'metrics_summary.json'),
            'predictions': str(out_dir/'predictions.csv'),
            'feature_importance': str(out_dir/'feature_importance.csv'),
            'config_used': str(out_dir/'config_used.json'),
            'model': str(out_dir/'model.json')
        }, f, indent=2)
    booster.save_model(str(out_dir/'model.json'))

    print(json.dumps(metrics, indent=2))
    print(f'Saved outputs to {out_dir}')


if __name__ == '__main__':
    run()
