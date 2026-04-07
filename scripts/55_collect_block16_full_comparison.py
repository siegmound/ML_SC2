import argparse
import json
import zipfile
from pathlib import Path
import pandas as pd

METRICS = ["accuracy", "balanced_accuracy", "roc_auc", "log_loss"]


def read_metrics_from_zip(zip_path: Path):
    rows = []
    with zipfile.ZipFile(zip_path, 'r') as zf:
        names = [n for n in zf.namelist() if n.endswith('metrics_summary.json')]
        for n in names:
            data = json.loads(zf.read(n))
            row = {k: data.get(k) for k in METRICS}
            row['seed'] = data.get('seed')
            row['dataset_name'] = data.get('dataset_name')
            row['profile'] = data.get('profile')
            row['model'] = data.get('model')
            row['zip_file'] = zip_path.name
            rows.append(row)
    if not rows:
        raise FileNotFoundError(f'No metrics_summary.json found in {zip_path}')
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame):
    out_rows = []
    group_cols = ['source_dataset', 'model_family', 'profile']
    for keys, g in df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row['n_runs'] = int(len(g))
        for m in METRICS:
            row[f'{m}_mean'] = float(g[m].mean())
            row[f'{m}_std'] = float(g[m].std(ddof=0))
        out_rows.append(row)
    return pd.DataFrame(out_rows).sort_values(['source_dataset', 'model_family', 'profile']).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo-root', default='.')
    ap.add_argument('--output-dir', default='results/block16_final')
    args = ap.parse_args()

    repo = Path(args.repo_root)
    out_dir = repo / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    sources = [
        ('v3_1_fixed', 'xgb', 'full', repo/'results/xgb_full/real_v3_1_fixed_fullgpu_clean/real_v3_1_fixed_fullgpu_clean.zip'),
        ('v3_2_combatfix', 'xgb', 'full', repo/'results/xgb_full/real_v3_2_combatfix_fullgpu_clean/xgbreal_v3_2_combatfix_fullgpu_clean.zip'),
        ('v3_1_fixed', 'rf', 'no_counter', repo/'results/rf_full/real_v3_1_fixed_fullrf_clean/real_v3_1_fixed_fullrf_clean.zip'),
        ('v3_2_combatfix', 'rf', 'no_counter', repo/'results/rf_full/real_v3_2_combatfix_fullrf_clean/rfreal_v3_2_combatfix_fullrf_clean.zip'),
    ]

    detailed = []
    missing = []
    for dataset, model, profile, zip_path in sources:
        if not zip_path.exists():
            missing.append(str(zip_path))
            continue
        df = read_metrics_from_zip(zip_path)
        df['source_dataset'] = dataset
        df['model_family'] = model
        df['target_profile'] = profile
        detailed.append(df)

    if missing:
        raise FileNotFoundError('Missing required result zips:\n' + '\n'.join(missing))

    detailed_df = pd.concat(detailed, ignore_index=True)
    summary_df = summarize(detailed_df)

    xgb_cmp = summary_df[summary_df['model_family']=='xgb'].copy()
    rf_cmp = summary_df[summary_df['model_family']=='rf'].copy()

    # Winner snapshots: classification and probabilistic viewpoints
    def winner_table(summary_df):
        rows = []
        for dataset, g in summary_df.groupby('source_dataset'):
            best_cls = g.sort_values(['accuracy_mean','balanced_accuracy_mean','roc_auc_mean'], ascending=False).iloc[0]
            best_prob = g.sort_values(['log_loss_mean'], ascending=True).iloc[0]
            rows.append({
                'source_dataset': dataset,
                'best_classifier_model': best_cls['model_family'],
                'best_classifier_profile': best_cls['profile'],
                'best_classifier_accuracy_mean': best_cls['accuracy_mean'],
                'best_classifier_roc_auc_mean': best_cls['roc_auc_mean'],
                'best_prob_model': best_prob['model_family'],
                'best_prob_profile': best_prob['profile'],
                'best_prob_log_loss_mean': best_prob['log_loss_mean'],
            })
        return pd.DataFrame(rows)

    winners = winner_table(summary_df)

    detailed_df.to_csv(out_dir/'block16_full_detailed.csv', index=False)
    summary_df.to_csv(out_dir/'block16_full_summary.csv', index=False)
    winners.to_csv(out_dir/'block16_winners.csv', index=False)

    manifest = {
        'sources': [{
            'dataset': d, 'model': m, 'profile': p, 'zip': str(z.relative_to(repo))
        } for d,m,p,z in sources],
        'outputs': [
            'block16_full_detailed.csv',
            'block16_full_summary.csv',
            'block16_winners.csv'
        ]
    }
    with open(out_dir/'block16_collection_manifest.json', 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)

    print(summary_df.to_string(index=False))
    print('\nSaved outputs to', out_dir)


if __name__ == '__main__':
    main()
