from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print('RUN', ' '.join(cmd))
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-zip', type=Path, required=True)
    parser.add_argument('--dataset-name', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--xgb-device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--include-rf', action='store_true')
    parser.add_argument('--max-train-rows', type=int, default=12000)
    parser.add_argument('--max-val-rows', type=int, default=4000)
    parser.add_argument('--max-test-rows', type=int, default=4000)
    args = parser.parse_args()

    split_json = PROJECT_ROOT / 'data' / 'processed' / 'splits' / f'{args.dataset_name}_split_seed_{args.seed}.json'
    run([sys.executable, 'scripts/04_make_group_splits.py', '--dataset-zip', str(args.dataset_zip), '--dataset-name', args.dataset_name, '--seeds', str(args.seed)])
    run([sys.executable, 'scripts/03_dataset_quality_report.py', '--dataset-zip', str(args.dataset_zip), '--dataset-name', args.dataset_name])
    run([sys.executable, 'scripts/05_train_logreg.py', '--dataset-zip', str(args.dataset_zip), '--split-json', str(split_json), '--dataset-name', args.dataset_name, '--fast-smoke'])
    if args.include_rf:
        run([sys.executable, 'scripts/06_train_rf.py', '--dataset-zip', str(args.dataset_zip), '--split-json', str(split_json), '--dataset-name', args.dataset_name, '--fast-smoke', '--max-train-rows', str(args.max_train_rows), '--max-val-rows', str(args.max_val_rows), '--max-test-rows', str(args.max_test_rows)])
    run([sys.executable, 'scripts/07_train_xgb.py', '--dataset-zip', str(args.dataset_zip), '--split-json', str(split_json), '--dataset-name', args.dataset_name, '--fast-smoke', '--device', args.xgb_device, '--max-train-rows', str(args.max_train_rows), '--max-val-rows', str(args.max_val_rows), '--max-test-rows', str(args.max_test_rows)])
    run([sys.executable, 'scripts/13_collect_results.py', '--results-root', str(PROJECT_ROOT / 'results')])


if __name__ == '__main__':
    main()
