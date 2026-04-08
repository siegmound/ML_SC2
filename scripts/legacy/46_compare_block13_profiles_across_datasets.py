from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def load_profile_csv(path: Path, dataset_label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['dataset_label'] = dataset_label
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--profile-csv', type=Path, nargs='+', required=True)
    ap.add_argument('--dataset-label', nargs='+', required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    args = ap.parse_args()
    if len(args.profile_csv) != len(args.dataset_label):
        raise SystemExit('profile-csv and dataset-label must have same length')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames = [load_profile_csv(p, label) for p, label in zip(args.profile_csv, args.dataset_label)]
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(args.output_dir / 'block13_profile_dataset_comparison.csv', index=False)

    pivot_acc = combined.pivot_table(index='profile', columns='dataset_label', values='accuracy')
    pivot_auc = combined.pivot_table(index='profile', columns='dataset_label', values='roc_auc')
    pivot_ll = combined.pivot_table(index='profile', columns='dataset_label', values='log_loss')
    pivot_acc.to_csv(args.output_dir / 'block13_profile_accuracy_pivot.csv')
    pivot_auc.to_csv(args.output_dir / 'block13_profile_auc_pivot.csv')
    pivot_ll.to_csv(args.output_dir / 'block13_profile_logloss_pivot.csv')
    print('Wrote dataset comparison outputs to', args.output_dir)


if __name__ == '__main__':
    main()
