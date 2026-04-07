from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def write_latex_table(df: pd.DataFrame, path: Path) -> None:
    path.write_text(df.to_latex(index=False, float_format=lambda x: f'{x:.4f}'), encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--master-results', type=Path, required=True)
    parser.add_argument('--aggregate-results', type=Path, default=None)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parents[1] / 'results' / 'figures')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.master_results)
    agg_df = pd.read_csv(args.aggregate_results) if args.aggregate_results and args.aggregate_results.exists() else None

    metric_cols = [c for c in ['accuracy', 'balanced_accuracy', 'roc_auc', 'log_loss'] if c in df.columns]
    if metric_cols:
        table_cols = [c for c in ['model_name', 'dataset_name', 'seed', *metric_cols] if c in df.columns]
        model_table = df[table_cols].copy()
        model_table.to_csv(args.output_dir / 'model_summary_table.csv', index=False)
        write_latex_table(model_table, args.output_dir / 'model_summary_table.tex')

    if agg_df is not None and not agg_df.empty:
        agg_df.to_csv(args.output_dir / 'aggregate_model_summary.csv', index=False)
        write_latex_table(agg_df, args.output_dir / 'aggregate_model_summary.tex')

        if 'model_name' in agg_df.columns and 'accuracy_mean' in agg_df.columns:
            plot_df = agg_df.sort_values('accuracy_mean', ascending=False)
            plt.figure(figsize=(8, 5))
            plt.bar(plot_df['model_name'], plot_df['accuracy_mean'])
            plt.ylabel('Mean accuracy')
            plt.xlabel('Model')
            plt.title('Aggregate Mean Accuracy by Model')
            plt.xticks(rotation=30, ha='right')
            plt.tight_layout()
            plt.savefig(args.output_dir / 'aggregate_mean_accuracy_by_model.png')
            plt.close()


if __name__ == '__main__':
    main()
