from __future__ import annotations

import argparse
from pathlib import Path
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sc2proj.utils import ensure_dir, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--imported-base', type=Path, default=PROJECT_ROOT / 'results' / 'block11_imported')
    parser.add_argument('--stability-dir', type=Path, default=PROJECT_ROOT / 'results' / 'block12_feature_stability')
    parser.add_argument('--ablation-dir', type=Path, default=PROJECT_ROOT / 'experiments' / 'block12_rf_ablation' / 'real_v3_1_fixed_smoke3000' / 'seed_42')
    parser.add_argument('--output-dir', type=Path, default=PROJECT_ROOT / 'results' / 'block12_report')
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    report = {}
    imported_metrics = args.imported_base / 'imported_metrics.csv'
    if imported_metrics.exists():
        df = pd.read_csv(imported_metrics)
        summary = df.groupby('model')[['accuracy','balanced_accuracy','roc_auc','log_loss']].agg(['mean','std']).reset_index()
        summary.to_csv(args.output_dir / 'block12_model_summary.csv', index=False)
        report['model_summary_rows'] = int(len(summary))

    for name in ['rf_feature_importance_stability.csv', 'xgb_feature_importance_stability.csv', 'rf_permutation_importance_stability.csv', 'block11_metric_summary_from_imports.csv']:
        p = args.stability_dir / name
        if p.exists():
            target = args.output_dir / name
            target.write_bytes(p.read_bytes())

    ablation_csv = args.ablation_dir / 'rf_family_ablation_ranked.csv'
    if ablation_csv.exists():
        ab = pd.read_csv(ablation_csv)
        ab.to_csv(args.output_dir / 'rf_family_ablation_ranked.csv', index=False)
        report['rf_top_ablation_impacts'] = ab.head(8).to_dict(orient='records')

    write_json(report, args.output_dir / 'block12_report.json')


if __name__ == '__main__':
    main()
