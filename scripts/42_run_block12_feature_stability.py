from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sc2proj.utils import ensure_dir, write_json


def load_run_dir(base_dir: Path, model: str, seed: int) -> Path:
    return base_dir / model / 'real_v3_1_fixed_smoke3000' / f'seed_{seed}'


def aggregate_feature_csv(paths: list[tuple[int, Path]], output_csv: Path, value_col: str = 'importance') -> pd.DataFrame:
    rows = []
    for seed, path in paths:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        feature_col = 'feature' if 'feature' in df.columns else 'Feature'
        val_col = value_col if value_col in df.columns else df.columns[-1]
        for _, row in df.iterrows():
            rows.append({'seed': seed, 'feature': row[feature_col], 'importance': float(row[val_col])})
    out = pd.DataFrame(rows)
    agg = out.groupby('feature')['importance'].agg(['mean','std','count']).reset_index().sort_values('mean', ascending=False)
    agg.to_csv(output_csv, index=False)
    return agg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--imported-base', type=Path, default=PROJECT_ROOT / 'results' / 'block11_imported')
    parser.add_argument('--output-dir', type=Path, default=PROJECT_ROOT / 'results' / 'block12_feature_stability')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42,43,44])
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    manifest = {}
    for model in ['rf','xgb']:
        paths = [(seed, load_run_dir(args.imported_base, model, seed) / 'feature_importance.csv') for seed in args.seeds]
        agg = aggregate_feature_csv(paths, args.output_dir / f'{model}_feature_importance_stability.csv')
        manifest[f'{model}_top10'] = agg.head(10).to_dict(orient='records')
        if model == 'rf':
            perm_paths = [(seed, load_run_dir(args.imported_base, model, seed) / 'permutation_importance.csv') for seed in args.seeds]
            existing = [(s,p) for s,p in perm_paths if p.exists()]
            if existing:
                aggregate_feature_csv(existing, args.output_dir / 'rf_permutation_importance_stability.csv')
    imported_metrics = args.imported_base / 'imported_metrics.csv'
    if imported_metrics.exists():
        df = pd.read_csv(imported_metrics)
        summary = df.groupby('model')[['accuracy','balanced_accuracy','roc_auc','log_loss']].agg(['mean','std']).reset_index()
        summary.to_csv(args.output_dir / 'block11_metric_summary_from_imports.csv', index=False)
    write_json(manifest, args.output_dir / 'feature_stability_manifest.json')


if __name__ == '__main__':
    main()
