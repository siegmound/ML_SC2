from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / 'results'
OUTDIR = RESULTS / 'block13_report'
OUTDIR.mkdir(parents=True, exist_ok=True)

report = {'profiles': {}, 'dataset_comparison_available': False}

base = RESULTS / 'block13_rf_profiles'
for dataset_dir in sorted(base.glob('*')):
    if not dataset_dir.is_dir():
        continue
    seed_dirs = sorted(dataset_dir.glob('seed_*'))
    frames = []
    for sd in seed_dirs:
        p = sd / 'rf_profile_comparison.csv'
        if p.exists():
            df = pd.read_csv(p)
            df['seed_dir'] = sd.name
            frames.append(df)
    if frames:
        all_df = pd.concat(frames, ignore_index=True)
        all_df.to_csv(OUTDIR / f'{dataset_dir.name}_profiles_all.csv', index=False)
        agg = all_df.groupby('profile').agg(
            accuracy_mean=('accuracy', 'mean'),
            accuracy_std=('accuracy', 'std'),
            roc_auc_mean=('roc_auc', 'mean'),
            roc_auc_std=('roc_auc', 'std'),
            log_loss_mean=('log_loss', 'mean'),
            log_loss_std=('log_loss', 'std'),
            n_features_mean=('n_features', 'mean'),
            n_runs=('profile', 'size'),
        ).reset_index().sort_values(['log_loss_mean', 'roc_auc_mean'], ascending=[True, False])
        agg.to_csv(OUTDIR / f'{dataset_dir.name}_profiles_aggregated.csv', index=False)
        report['profiles'][dataset_dir.name] = agg.to_dict(orient='records')

cmp = OUTDIR / 'block13_profile_dataset_comparison.csv'
# allow from script 46 output copied here later
if not cmp.exists():
    alt = RESULTS / 'block13_dataset_compare' / 'block13_profile_dataset_comparison.csv'
    if alt.exists():
        cmp = alt
if cmp.exists():
    report['dataset_comparison_available'] = True
    report['dataset_comparison_csv'] = str(cmp)

(OUTDIR / 'block13_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print('Wrote', OUTDIR / 'block13_report.json')
