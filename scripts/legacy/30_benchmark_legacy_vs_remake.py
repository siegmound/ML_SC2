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
    parser.add_argument('--output-csv', type=Path, default=PROJECT_ROOT / 'results' / 'summaries' / 'block8_legacy_vs_remake.csv')
    args = parser.parse_args()

    rows = []
    legacy_dir = args.results_root / 'legacy_imports' / 'summaries'
    if legacy_dir.exists():
        for path in sorted(legacy_dir.glob('*.json')):
            data = load_json(path)
            rows.append({'source': 'legacy', 'name': path.stem, 'accuracy': data.get('accuracy'), 'balanced_accuracy': data.get('balanced_accuracy'), 'roc_auc': data.get('roc_auc'), 'log_loss': data.get('log_loss')})
    for model_dir in [args.results_root / 'logreg', args.results_root / 'rf', args.results_root / 'xgb']:
        if not model_dir.exists():
            continue
        for metrics_path in sorted(model_dir.glob('*/*/metrics_summary.json')):
            data = load_json(metrics_path)
            rows.append({'source': 'remake', 'name': str(metrics_path.parent.relative_to(args.results_root)), 'accuracy': data.get('accuracy'), 'balanced_accuracy': data.get('balanced_accuracy'), 'roc_auc': data.get('roc_auc'), 'log_loss': data.get('log_loss')})
    out = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    print(args.output_csv)


if __name__ == '__main__':
    main()
