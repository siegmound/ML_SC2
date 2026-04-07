import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, log_loss, roc_auc_score


def ece_mce(y_true, y_prob, n_bins=15):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins[1:-1], right=True)
    ece = 0.0
    mce = 0.0
    details = []
    for b in range(n_bins):
        mask = bin_ids == b
        if not np.any(mask):
            details.append({'bin': b, 'count': 0, 'mean_prob': None, 'empirical_rate': None, 'gap': None})
            continue
        yp = y_prob[mask]
        yt = y_true[mask]
        mean_prob = float(np.mean(yp))
        empirical = float(np.mean(yt))
        gap = abs(mean_prob - empirical)
        weight = float(np.mean(mask))
        ece += weight * gap
        mce = max(mce, gap)
        details.append({'bin': b, 'count': int(np.sum(mask)), 'mean_prob': mean_prob, 'empirical_rate': empirical, 'gap': float(gap)})
    return float(ece), float(mce), details


def metrics(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    ece, mce, bins = ece_mce(y_true, y_prob, n_bins=15)
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'roc_auc': float(roc_auc_score(y_true, y_prob)),
        'log_loss': float(log_loss(y_true, y_prob)),
        'brier_score': float(brier_score_loss(y_true, y_prob)),
        'ece_15': float(ece),
        'mce_15': float(mce),
        'bins': bins,
    }


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--validation-predictions', required=True)
    ap.add_argument('--test-predictions', required=True)
    ap.add_argument('--model-name', required=True)
    ap.add_argument('--dataset-name', required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--output-root', default='results/final_calibration')
    return ap.parse_args()


def run():
    args = parse_args()
    out_dir = Path(args.output_root) / args.dataset_name / args.model_name / f'seed_{args.seed}'
    out_dir.mkdir(parents=True, exist_ok=True)

    val_df = pd.read_csv(args.validation_predictions)
    test_df = pd.read_csv(args.test_predictions)

    y_val = val_df['y_true'].astype(int).to_numpy()
    p_val = val_df['y_prob'].astype(float).to_numpy()
    y_test = test_df['y_true'].astype(int).to_numpy()
    p_test = test_df['y_prob'].astype(float).to_numpy()

    # Platt scaling on validation
    lr = LogisticRegression(solver='lbfgs', max_iter=1000)
    lr.fit(p_val.reshape(-1, 1), y_val)
    p_test_platt = lr.predict_proba(p_test.reshape(-1, 1))[:, 1]
    p_val_platt = lr.predict_proba(p_val.reshape(-1, 1))[:, 1]

    # Isotonic on validation
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(p_val, y_val)
    p_test_iso = iso.predict(p_test)
    p_val_iso = iso.predict(p_val)

    rows = []
    payload = {}
    for method, probs_test, probs_val in [
        ('uncalibrated', p_test, p_val),
        ('platt', p_test_platt, p_val_platt),
        ('isotonic', p_test_iso, p_val_iso),
    ]:
        m_test = metrics(y_test, probs_test)
        m_val = metrics(y_val, probs_val)
        payload[method] = {'validation': m_val, 'test': m_test}
        rows.append({
            'dataset_name': args.dataset_name,
            'model_name': args.model_name,
            'seed': args.seed,
            'method': method,
            'test_accuracy': m_test['accuracy'],
            'test_balanced_accuracy': m_test['balanced_accuracy'],
            'test_roc_auc': m_test['roc_auc'],
            'test_log_loss': m_test['log_loss'],
            'test_brier_score': m_test['brier_score'],
            'test_ece_15': m_test['ece_15'],
            'test_mce_15': m_test['mce_15'],
            'val_log_loss': m_val['log_loss'],
            'val_brier_score': m_val['brier_score'],
            'val_ece_15': m_val['ece_15'],
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / 'calibration_comparison.csv', index=False)
    with open(out_dir / 'calibration_report.json', 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    print(df.to_string(index=False))
    print(f'Saved calibration outputs to {out_dir}')


if __name__ == '__main__':
    run()
