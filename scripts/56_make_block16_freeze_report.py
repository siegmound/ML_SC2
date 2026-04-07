import argparse
import json
from pathlib import Path
import pandas as pd


def fmt_pm(mean, std):
    return f"{mean:.4f} ± {std:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo-root', default='.')
    ap.add_argument('--input-dir', default='results/block16_final')
    args = ap.parse_args()

    repo = Path(args.repo_root)
    in_dir = repo / args.input_dir
    summary = pd.read_csv(in_dir / 'block16_full_summary.csv')

    report = {
        'formal_conclusion': {
            'dataset_versioning': 'v3_1_fixed is at least as strong as v3_2_combatfix in full multi-seed evaluation; combatfix does not show a clear final advantage.',
            'model_comparison': 'RF (no_counter) and XGB are effectively tied on full multi-seed evaluation. RF is marginally better on classification-oriented metrics, while XGB is marginally better on log loss.',
            'rf_profile': 'no_counter remains the strongest RF profile among the tested options.'
        },
        'recommended_finalists': {
            'classification_focused': 'RF + v3_1_fixed + no_counter',
            'probability_focused': 'XGB + v3_1_fixed + full'
        },
        'summary_rows': summary.to_dict(orient='records')
    }
    with open(in_dir / 'block16_final_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    md = []
    md.append('# Block 16 — Final Reproducible Freeze Summary\n')
    md.append('## Formalized current results\n')
    md.append('| Dataset | Model | Profile | Accuracy | Balanced Acc. | ROC-AUC | Log Loss |')
    md.append('|---|---|---|---:|---:|---:|---:|')
    for _, r in summary.iterrows():
        md.append(
            f"| {r['source_dataset']} | {r['model_family']} | {r['profile']} | {fmt_pm(r['accuracy_mean'], r['accuracy_std'])} | {fmt_pm(r['balanced_accuracy_mean'], r['balanced_accuracy_std'])} | {fmt_pm(r['roc_auc_mean'], r['roc_auc_std'])} | {fmt_pm(r['log_loss_mean'], r['log_loss_std'])} |"
        )
    md.append('\n## Interpretation\n')
    md.append('- `v3_1_fixed` is not worse than `v3_2_combatfix` in full multi-seed tests; if anything, it is marginally stronger or equivalent across the four reported metrics.')
    md.append('- RF with profile `no_counter` remains the best RF configuration among the tested feature profiles.')
    md.append('- RF and XGB are effectively tied at the project end-state. RF is slightly stronger as a classifier; XGB is slightly stronger in log loss, so it is the better probability-oriented candidate.')
    md.append('\n## Recommended final model selections\n')
    md.append('- **Classification-oriented final choice:** RF + `v3_1_fixed` + `no_counter`')
    md.append('- **Probability-oriented final choice:** XGB + `v3_1_fixed` + full feature set')
    md.append('\n## Remaining work (optional, not required to support the current conclusions)\n')
    md.append('- Final comparative calibration between RF and XGB on full runs.')
    md.append('- Deep-learning challenger as an extra, not as the main project axis.')
    (in_dir / 'block16_final_report.md').write_text('\n'.join(md), encoding='utf-8')

    tex_lines = []
    tex_lines.append(r'\begin{tabular}{lllcccc}')
    tex_lines.append(r'Dataset & Model & Profile & Accuracy & Bal. Acc. & ROC-AUC & Log Loss \\')
    tex_lines.append(r'\hline')
    for _, r in summary.iterrows():
        tex_lines.append(
            f"{r['source_dataset']} & {r['model_family']} & {r['profile']} & {fmt_pm(r['accuracy_mean'], r['accuracy_std'])} & {fmt_pm(r['balanced_accuracy_mean'], r['balanced_accuracy_std'])} & {fmt_pm(r['roc_auc_mean'], r['roc_auc_std'])} & {fmt_pm(r['log_loss_mean'], r['log_loss_std'])} \\\\"
        )
    tex_lines.append(r'\end{tabular}')
    (in_dir / 'block16_final_table.tex').write_text('\n'.join(tex_lines), encoding='utf-8')

    print('Saved final report artifacts to', in_dir)


if __name__ == '__main__':
    main()
