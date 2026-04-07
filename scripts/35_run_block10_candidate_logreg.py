from __future__ import annotations
import argparse, json, zipfile
from pathlib import Path
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, log_loss
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def load_df(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as zf:
        name = [n for n in zf.namelist() if n.endswith('.csv')][0]
        with zf.open(name) as fh:
            return pd.read_csv(fh)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset-zip', type=Path, required=True)
    ap.add_argument('--dataset-name', required=True)
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 43])
    ap.add_argument('--output-dir', type=Path, required=True)
    args = ap.parse_args()

    df = load_df(args.dataset_zip)
    X = df.drop(columns=['p1_wins', 'replay_id', 'time_sec'])
    y = df['p1_wins'].astype(int)
    g = df['replay_id'].astype(str)
    rows = []
    for seed in args.seeds:
        outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        tv_idx, test_idx = next(outer.split(X, y, groups=g))
        X_tv, y_tv, g_tv = X.iloc[tv_idx], y.iloc[tv_idx], g.iloc[tv_idx]
        X_test, y_test, g_test = X.iloc[test_idx], y.iloc[test_idx], g.iloc[test_idx]
        inner = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        tr_idx, val_idx = next(inner.split(X_tv, y_tv, groups=g_tv))
        X_train = pd.concat([X_tv.iloc[tr_idx], X_tv.iloc[val_idx]])
        y_train = pd.concat([y_tv.iloc[tr_idx], y_tv.iloc[val_idx]])
        model = Pipeline([('sc', StandardScaler()), ('clf', LogisticRegression(max_iter=200, class_weight='balanced', random_state=seed))])
        model.fit(X_train, y_train)
        prob = model.predict_proba(X_test)[:, 1]
        pred = (prob >= 0.5).astype(int)
        rows.append({
            'dataset_name': args.dataset_name,
            'seed': seed,
            'model_name': 'logreg',
            'accuracy': float(accuracy_score(y_test, pred)),
            'balanced_accuracy': float(balanced_accuracy_score(y_test, pred)),
            'roc_auc': float(roc_auc_score(y_test, prob)),
            'log_loss': float(log_loss(y_test, prob, labels=[0, 1])),
            'n_rows': int(len(df)),
            'n_replays': int(g.nunique()),
            'n_train_rows': int(len(X_train)),
            'n_test_rows': int(len(X_test)),
            'n_train_replays': int(pd.concat([g_tv.iloc[tr_idx], g_tv.iloc[val_idx]]).nunique()),
            'n_test_replays': int(g_test.nunique()),
        })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail = pd.DataFrame(rows)
    detail.to_csv(args.output_dir / 'block10_logreg_multiseed.csv', index=False)
    leader = pd.DataFrame([{
        'model_name': 'logreg',
        'runs': int(len(detail)),
        'accuracy_mean': float(detail['accuracy'].mean()),
        'accuracy_std': float(detail['accuracy'].std(ddof=1)) if len(detail) > 1 else None,
        'balanced_accuracy_mean': float(detail['balanced_accuracy'].mean()),
        'balanced_accuracy_std': float(detail['balanced_accuracy'].std(ddof=1)) if len(detail) > 1 else None,
        'roc_auc_mean': float(detail['roc_auc'].mean()),
        'roc_auc_std': float(detail['roc_auc'].std(ddof=1)) if len(detail) > 1 else None,
        'log_loss_mean': float(detail['log_loss'].mean()),
        'log_loss_std': float(detail['log_loss'].std(ddof=1)) if len(detail) > 1 else None,
    }])
    leader.to_csv(args.output_dir / 'block10_candidate_leaderboard.csv', index=False)
    (args.output_dir.parent / 'block10_status_report.json').write_text(json.dumps({
        'dataset_name': args.dataset_name,
        'completed_models': ['logreg'],
        'attempted_models': ['logreg', 'rf', 'xgb'],
        'notes': [
            'RF and XGB candidate-final attempts were not stable enough in this session to claim completed runs.',
            'This block therefore freezes the first candidate-final benchmark around a smoke3000 LogReg multi-seed run.'
        ],
        'detailed_csv': str(args.output_dir / 'block10_logreg_multiseed.csv'),
        'leaderboard_csv': str(args.output_dir / 'block10_candidate_leaderboard.csv'),
    }, indent=2), encoding='utf-8')
    print(detail)
    print(leader)


if __name__ == '__main__':
    main()
