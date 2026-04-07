from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-zip', type=Path, required=True)
    parser.add_argument('--dataset-name', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-path', type=Path, default=PROJECT_ROOT / 'results' / 'block8_staged_run_plan.json')
    args = parser.parse_args()

    split_json = PROJECT_ROOT / 'data' / 'processed' / 'splits' / f'{args.dataset_name}_split_seed_{args.seed}.json'
    plan = {
        'dataset_zip': str(args.dataset_zip),
        'dataset_name': args.dataset_name,
        'seed': args.seed,
        'split_json': str(split_json),
        'stages': [
            {
                'name': 'smoke_xgb',
                'command': ['python', 'scripts/07_train_xgb.py', '--dataset-zip', str(args.dataset_zip), '--split-json', str(split_json), '--dataset-name', args.dataset_name, '--fast-smoke', '--device', 'cpu', '--max-train-rows', '12000', '--max-val-rows', '4000', '--max-test-rows', '4000'],
            },
            {
                'name': 'smoke_rf',
                'command': ['python', 'scripts/06_train_rf.py', '--dataset-zip', str(args.dataset_zip), '--split-json', str(split_json), '--dataset-name', args.dataset_name, '--fast-smoke', '--max-train-rows', '12000', '--max-val-rows', '4000', '--max-test-rows', '4000'],
            },
            {
                'name': 'standard_xgb',
                'command': ['python', 'scripts/07_train_xgb.py', '--dataset-zip', str(args.dataset_zip), '--split-json', str(split_json), '--dataset-name', args.dataset_name, '--search-stage', 'standard', '--device', 'cpu'],
            },
            {
                'name': 'standard_rf',
                'command': ['python', 'scripts/06_train_rf.py', '--dataset-zip', str(args.dataset_zip), '--split-json', str(split_json), '--dataset-name', args.dataset_name],
            },
        ],
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(plan, indent=2), encoding='utf-8')
    print(args.output_path)


if __name__ == '__main__':
    main()
