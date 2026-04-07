from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROFILES = ['full', 'no_counter', 'no_counter_no_losses']


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset-name', required=True)
    ap.add_argument('--dataset-zip', type=Path, required=True)
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    ap.add_argument('--max-train-rows', type=int, default=10000)
    ap.add_argument('--max-val-rows', type=int, default=4000)
    ap.add_argument('--max-test-rows', type=int, default=5000)
    ap.add_argument('--n-estimators', type=int, default=100)
    ap.add_argument('--max-depth', type=int, default=16)
    ap.add_argument('--min-samples-split', type=int, default=5)
    ap.add_argument('--min-samples-leaf', type=int, default=2)
    ap.add_argument('--max-features', default='sqrt')
    ap.add_argument('--class-weight', default='none', choices=['none', 'balanced', 'balanced_subsample'])
    ap.add_argument('--n-jobs', type=int, default=2)
    args = ap.parse_args()

    for seed in args.seeds:
        split_json = PROJECT_ROOT / 'data' / 'processed' / 'splits' / f'{args.dataset_name}_split_seed_{seed}.json'
        cmd = [
            sys.executable, str(PROJECT_ROOT / 'scripts' / '44_run_block13_rf_profiles.py'),
            '--dataset-zip', str(args.dataset_zip),
            '--split-json', str(split_json),
            '--dataset-name', args.dataset_name,
            '--profiles', *PROFILES,
            '--seed', str(seed),
            '--n-estimators', str(args.n_estimators),
            '--max-depth', str(args.max_depth),
            '--min-samples-split', str(args.min_samples_split),
            '--min-samples-leaf', str(args.min_samples_leaf),
            '--max-features', str(args.max_features),
            '--class-weight', args.class_weight,
            '--max-train-rows', str(args.max_train_rows),
            '--max-val-rows', str(args.max_val_rows),
            '--max-test-rows', str(args.max_test_rows),
            '--n-jobs', str(args.n_jobs),
        ]
        print('RUN', args.dataset_name, 'seed', seed)
        subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
