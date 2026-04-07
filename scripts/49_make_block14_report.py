from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / 'results'
BASE = RESULTS / 'block13_rf_profiles'
OUT = RESULTS / 'block14_report'
OUT.mkdir(parents=True, exist_ok=True)
TARGETS = ['real_v3_1_fixed_smoke3000', 'real_v3_2_combatfix_smoke3000']
PROFILES = ['full', 'no_counter', 'no_counter_no_losses']

all_frames = []
report = {'datasets': {}, 'cross_dataset_winner': {}}
for dataset in TARGETS:
    frames = []
    for seed_dir in sorted((BASE / dataset).glob('seed_*')):
        p = seed_dir / 'rf_profile_comparison.csv'
        if p.exists():
            df = pd.read_csv(p)
            df = df[df['profile'].isin(PROFILES)].copy()
            df['seed'] = int(seed_dir.name.split('_')[-1])
            df['dataset_name'] = dataset
            frames.append(df)
    if not frames:
        continue
    df = pd.concat(frames, ignore_index=True)
    all_frames.append(df)
    df.to_csv(OUT / f'{dataset}_all_profiles_multiseed.csv', index=False)
    agg = df.groupby('profile').agg(
        accuracy_mean=('accuracy', 'mean'),
        accuracy_std=('accuracy', 'std'),
        balanced_accuracy_mean=('balanced_accuracy', 'mean'),
        balanced_accuracy_std=('balanced_accuracy', 'std'),
        roc_auc_mean=('roc_auc', 'mean'),
        roc_auc_std=('roc_auc', 'std'),
        log_loss_mean=('log_loss', 'mean'),
        log_loss_std=('log_loss', 'std'),
        n_runs=('profile', 'size'),
        n_features_mean=('n_features', 'mean'),
    ).reset_index().sort_values(['log_loss_mean','roc_auc_mean'], ascending=[True, False])
    agg.to_csv(OUT / f'{dataset}_aggregated_profiles_multiseed.csv', index=False)
    best_row = agg.sort_values(['log_loss_mean','roc_auc_mean'], ascending=[True, False]).iloc[0].to_dict()
    report['datasets'][dataset] = {'best_profile': best_row, 'rows': agg.to_dict(orient='records')}

if all_frames:
    all_df = pd.concat(all_frames, ignore_index=True)
    all_df.to_csv(OUT / 'block14_all_profiles_multiseed.csv', index=False)
    pivot_acc = all_df.pivot_table(index='profile', columns='dataset_name', values='accuracy', aggfunc='mean')
    pivot_auc = all_df.pivot_table(index='profile', columns='dataset_name', values='roc_auc', aggfunc='mean')
    pivot_ll = all_df.pivot_table(index='profile', columns='dataset_name', values='log_loss', aggfunc='mean')
    pivot_acc.to_csv(OUT / 'block14_accuracy_pivot.csv')
    pivot_auc.to_csv(OUT / 'block14_auc_pivot.csv')
    pivot_ll.to_csv(OUT / 'block14_logloss_pivot.csv')

    if set(TARGETS).issubset(set(all_df['dataset_name'].unique())):
        cmp = []
        for profile in PROFILES:
            a = all_df[(all_df['dataset_name']==TARGETS[0]) & (all_df['profile']==profile)]
            b = all_df[(all_df['dataset_name']==TARGETS[1]) & (all_df['profile']==profile)]
            if len(a) and len(b):
                cmp.append({
                    'profile': profile,
                    'acc_delta_combatfix_minus_fixed': float(b['accuracy'].mean() - a['accuracy'].mean()),
                    'auc_delta_combatfix_minus_fixed': float(b['roc_auc'].mean() - a['roc_auc'].mean()),
                    'logloss_delta_combatfix_minus_fixed': float(b['log_loss'].mean() - a['log_loss'].mean()),
                })
        cmp_df = pd.DataFrame(cmp).sort_values(['logloss_delta_combatfix_minus_fixed','auc_delta_combatfix_minus_fixed'], ascending=[True, False])
        cmp_df.to_csv(OUT / 'block14_dataset_profile_deltas.csv', index=False)
        report['cross_dataset_winner'] = cmp_df.to_dict(orient='records')

(OUT / 'block14_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print('Wrote', OUT / 'block14_report.json')
