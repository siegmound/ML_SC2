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
    parser.add_argument('--dataset-name', type=str, default='real_v3_1_fixed')
    parser.add_argument('--dataset-version', type=str, default='v3_1_fixed')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--xgb-device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--skip-xgb', action='store_true')
    args = parser.parse_args()

    split_json = PROJECT_ROOT / 'data' / 'processed' / 'splits' / f'split_seed_{args.seed}.json'
    run([sys.executable, 'scripts/23_register_real_dataset.py', '--dataset-zip', str(args.dataset_zip), '--dataset-name', args.dataset_name, '--dataset-version', args.dataset_version])
    run([sys.executable, 'scripts/03_dataset_quality_report.py', '--dataset-zip', str(args.dataset_zip), '--dataset-name', args.dataset_name])
    run([sys.executable, 'scripts/04_make_group_splits.py', '--dataset-zip', str(args.dataset_zip), '--seeds', str(args.seed)])
    run([sys.executable, 'scripts/05_train_logreg.py', '--dataset-zip', str(args.dataset_zip), '--split-json', str(split_json), '--dataset-name', args.dataset_name])
    if not args.skip_xgb:
        run([sys.executable, 'scripts/07_train_xgb.py', '--dataset-zip', str(args.dataset_zip), '--split-json', str(split_json), '--dataset-name', args.dataset_name, '--device', args.xgb_device])
    run([sys.executable, 'scripts/13_collect_results.py', '--results-root', str(PROJECT_ROOT / 'results')])
    run([sys.executable, 'scripts/26_build_block7_status_report.py', '--dataset-name', args.dataset_name])


if __name__ == '__main__':
    main()
