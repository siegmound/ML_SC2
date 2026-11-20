"""B2 strict four-way RF/XGB calibration runner (new outputs only).

Replay-disjoint train / tuning / calibration / test partitions. The original
validation replay groups are split deterministically into tuning and calibration
subsets. XGBoost early stopping uses tuning only; calibration fits calibration
only; test probabilities are generated after calibration models are frozen.

Historical test sets have already been inspected in the project, so results from
this runner remain a revised analysis, NOT a pristine never-seen test result.
"""

from __future__ import annotations

import argparse
import json
import random
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, brier_score_loss,
    log_loss, roc_auc_score,
)


def split_four_way(original: dict, fraction: float = 0.5) -> dict:
    if not (0.05 <= fraction <= 0.95):
        raise ValueError('Calibration fraction must be between 0.05 and 0.95')
    names = ('train_groups', 'val_groups', 'test_groups')
    groups = {name: list(map(str, original[name])) for name in names}
    for name, ids in groups.items():
        if len(ids) != len(set(ids)) or not ids:
            raise ValueError(f'Empty/duplicated {name}')
    a, b, c = [set(groups[name]) for name in names]
    if a & b or a & c or b & c:
        raise ValueError('Overlapping original train/validation/test replay IDs')
    rng = random.Random(int(original['seed']))
    val = sorted(b)
    rng.shuffle(val)
    n_cal = max(1, min(len(val) - 1, round(len(val) * fraction)))
    if len(val) < 4:
        raise ValueError('At least four validation replay IDs required')
    calibration = sorted(val[:n_cal])
    tune = sorted(val[n_cal:])
    return {
        'seed': int(original['seed']),
        'train_groups': sorted(a), 'tune_groups': tune,
        'calibration_groups': calibration, 'test_groups': sorted(c),
        'calibration_fraction': fraction,
        'groups_disjoint': True,
    }


def load_data(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        candidates = [name for name in archive.namelist() if name.lower().endswith('.csv')]
        if len(candidates) != 1:
            raise ValueError('Exactly one CSV is required in the dataset archive')
        with archive.open(candidates[0]) as source:
            frame = pd.read_csv(source)
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    return frame


def partition(df: pd.DataFrame, plan: dict, model: str):
    parts = {key: df[df['replay_id'].astype(str).isin(plan[key + '_groups'])].copy()
             for key in ('train', 'tune', 'calibration', 'test')}
    for key, subset in parts.items():
        if subset.empty or subset['p1_wins'].nunique() != 2:
            raise ValueError(f'{model}: empty or single-class {key} split')
    # The filtered table must be completely covered by the original split IDs.
    if sum(map(len, parts.values())) != len(df):
        raise ValueError('Four-way split does not cover every filtered snapshot row')
    return parts


def calibration_errors(y, prob, n_bins: int = 15):
    bin_ids = np.digitize(prob, np.linspace(0, 1, n_bins + 1)[1:-1], right=True)
    ece, mce = 0.0, 0.0
    for index in range(n_bins):
        mask = bin_ids == index
        if not np.any(mask):
            continue
        gap = abs(float(np.mean(prob[mask])) - float(np.mean(y[mask])))
        ece += float(np.mean(mask)) * gap
        mce = max(mce, gap)
    return float(ece), float(mce)


def metrics(y, prob):
    pred = (prob >= 0.5).astype(int)
    ece, mce = calibration_errors(y, prob)
    return {
        'accuracy': float(accuracy_score(y, pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y, pred)),
        'roc_auc': float(roc_auc_score(y, prob)),
        'log_loss': float(log_loss(y, prob)),
        'brier': float(brier_score_loss(y, prob)),
        'ece_15': ece, 'mce_15': mce,
    }


def fit_predictors(parts, model: str, seed: int, args):
    drop = {'replay_id', 'time_sec', 'p1_wins'}
    columns = [c for c in parts['train'].columns if c not in drop]
    if model == 'rf':
        columns = [c for c in columns if 'counter' not in c.lower()]
    arrays = {name: part[columns].to_numpy(dtype=np.float32)
              for name, part in parts.items()}
    labels = {name: part['p1_wins'].to_numpy(dtype=int)
              for name, part in parts.items()}
    if model == 'rf':
        estimator = RandomForestClassifier(
            n_estimators=args.n_estimators, max_depth=24,
            min_samples_split=4, min_samples_leaf=2, max_features='sqrt',
            class_weight='balanced_subsample', random_state=seed,
            n_jobs=args.n_jobs,
        )
        estimator.fit(arrays['train'], labels['train'])
        def predict(name):
            return estimator.predict_proba(arrays[name])[:, 1]
        training = {'fit_groups': 'train', 'selection_groups': None,
                    'calibration_groups': 'calibration', 'n_estimators': args.n_estimators}
    else:
        import xgboost as xgb
        params = {
            'objective': 'binary:logistic', 'eval_metric': 'logloss',
            'random_state': seed, 'max_depth': 5, 'learning_rate': 0.03,
            'min_child_weight': 1, 'subsample': 0.8,
            'colsample_bytree': 0.8, 'reg_lambda': 1.0,
            'tree_method': 'hist', 'device': args.device, 'nthread': args.n_jobs,
        }
        dtrain = xgb.DMatrix(arrays['train'], label=labels['train'])
        dtune = xgb.DMatrix(arrays['tune'], label=labels['tune'])
        booster = xgb.train(
            params, dtrain, num_boost_round=args.num_boost_round,
            evals=[(dtune, 'tuning')],
            early_stopping_rounds=args.early_stopping_rounds,
            verbose_eval=False,
        )
        end = booster.best_iteration + 1
        def predict(name):
            return booster.predict(xgb.DMatrix(arrays[name]), iteration_range=(0, end))
        training = {'fit_groups': 'train', 'selection_groups': 'tune',
                    'calibration_groups': 'calibration', 'best_iteration': int(booster.best_iteration),
                    'num_boost_round': args.num_boost_round}
    return predict, labels, columns, training


def run(args):
    with Path(args.split_json).open(encoding='utf-8') as handle:
        original = json.load(handle)
    plan = split_four_way(original, args.calibration_fraction)
    seed = plan['seed']
    output = Path(args.output_root) / args.dataset_name / args.model / f'seed_{seed}'
    if output.exists():
        raise FileExistsError(f'Output already exists (refusing overwrite): {output}')
    data = load_data(Path(args.dataset_zip))
    parts = partition(data, plan, args.model)
    predictor, labels, columns, training = fit_predictors(parts, args.model, seed, args)

    # Calibrator fitting precedes any test inference, and tune IDs are excluded.
    p_cal = predictor('calibration')
    platt = LogisticRegression(solver='lbfgs', max_iter=1000)
    platt.fit(p_cal.reshape(-1, 1), labels['calibration'])
    isotonic = IsotonicRegression(out_of_bounds='clip')
    isotonic.fit(p_cal, labels['calibration'])

    p_test = predictor('test')  # A single test model inference after selection/fitting.
    probabilities = {
        'uncalibrated': p_test,
        'platt': platt.predict_proba(p_test.reshape(-1, 1))[:, 1],
        'isotonic': isotonic.predict(p_test),
    }
    output.mkdir(parents=True)
    scores = {name: metrics(labels['test'], p) for name, p in probabilities.items()}
    for name, p in probabilities.items():
        pd.DataFrame({
            'replay_id': parts['test']['replay_id'].to_numpy(),
            'time_sec': parts['test']['time_sec'].to_numpy(),
            'y_true': labels['test'], 'y_prob': p,
            'y_pred': (p >= 0.5).astype(int),
        }).to_csv(output / f'test_predictions_{name}.csv', index=False)

    report = {
        'protocol': 'B2_four_way_replay_disjoint', 'historical_test_previously_inspected': True,
        'seed': seed, 'model': args.model, 'dataset_name': args.dataset_name,
        'training': training,
        'partition_rows': {key: len(value) for key, value in parts.items()},
        'partition_replay_counts': {key: len(plan[key + '_groups']) for key in parts},
        'n_features': len(columns), 'scores': scores,
        'calibration_fitting': 'calibration replay IDs only; independent of XGB tune set',
    }
    (output / 'metrics_summary.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    (output / 'split_manifest.json').write_text(json.dumps(plan, indent=2) + '\n', encoding='utf-8')
    (output / 'feature_names.json').write_text(json.dumps(columns, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-zip', required=True)
    parser.add_argument('--split-json', required=True)
    parser.add_argument('--dataset-name', required=True)
    parser.add_argument('--model', choices=['rf', 'xgb'], required=True)
    parser.add_argument('--output-root', default='results/b2_strict_calibration')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--calibration-fraction', type=float, default=0.5)
    parser.add_argument('--n-estimators', type=int, default=700)
    parser.add_argument('--n-jobs', type=int, default=1)
    parser.add_argument('--num-boost-round', type=int, default=2000)
    parser.add_argument('--early-stopping-rounds', type=int, default=50)
    return parser.parse_args()


if __name__ == '__main__':
    run(parse_args())
