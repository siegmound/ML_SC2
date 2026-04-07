from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-root', type=Path, default=PROJECT_ROOT / 'results')
    parser.add_argument('--output-json', type=Path, default=PROJECT_ROOT / 'results' / 'block8_status_report.json')
    args = parser.parse_args()

    completed = []
    for model_name in ['logreg', 'rf', 'xgb', 'mlp', 'mlp_torch']:
        model_root = args.results_root / model_name
        if not model_root.exists():
            continue
        for metrics_path in sorted(model_root.glob('*/*/metrics_summary.json')):
            data = load_json(metrics_path)
            completed.append({
                'model': model_name,
                'dataset_name': data.get('dataset_name'),
                'seed': data.get('seed'),
                'accuracy': data.get('accuracy'),
                'balanced_accuracy': data.get('balanced_accuracy'),
                'roc_auc': data.get('roc_auc'),
                'log_loss': data.get('log_loss'),
                'path': str(metrics_path),
            })
    df = pd.DataFrame(completed)
    summary = {
        'n_completed_runs': int(len(df)),
        'models_completed': sorted(df['model'].unique().tolist()) if not df.empty else [],
        'datasets_completed': sorted(df['dataset_name'].dropna().unique().tolist()) if not df.empty else [],
        'best_accuracy_run': None,
        'best_log_loss_run': None,
    }
    if not df.empty:
        best_acc_row = df.sort_values('accuracy', ascending=False).iloc[0].to_dict()
        best_ll_row = df.sort_values('log_loss', ascending=True).iloc[0].to_dict()
        summary['best_accuracy_run'] = best_acc_row
        summary['best_log_loss_run'] = best_ll_row
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(args.output_json)


if __name__ == '__main__':
    main()
