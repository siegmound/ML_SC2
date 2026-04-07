from __future__ import annotations

import argparse
from pathlib import Path

REQUIRED_SCRIPT_NAMES = [
    'scripts/00_parser_smoke_test.py',
    'scripts/01_parser_audit.py',
    'scripts/02_build_dataset.py',
    'scripts/03_dataset_quality_report.py',
    'scripts/04_make_group_splits.py',
    'scripts/05_train_logreg.py',
    'scripts/06_train_rf.py',
    'scripts/07_train_xgb.py',
    'scripts/08_train_mlp.py',
    'scripts/08_train_mlp_torch.py',
    'scripts/09_run_ablation.py',
    'scripts/10_run_temporal_study.py',
    'scripts/11_run_matchup_study.py',
    'scripts/12_run_calibration.py',
    'scripts/13_collect_results.py',
    'scripts/14_make_tables_figures.py',
    'scripts/15_verify_reproducibility.py',
    'scripts/16_run_error_analysis.py',
    'scripts/17_internal_audit.py',
]

REQUIRED_CONFIGS = [
    'configs/models/logreg.yaml',
    'configs/models/rf.yaml',
    'configs/models/xgb.yaml',
    'configs/models/mlp.yaml',
    'configs/models/mlp_torch.yaml',
    'configs/experiments/ablation.yaml',
    'configs/experiments/temporal.yaml',
    'configs/experiments/calibration.yaml',
    'configs/experiments/collection.yaml',
]

REQUIRED_DOCS = [
    'README.md',
    'environment.yml',
    'pyproject.toml',
    'docs/methodology.md',
    'docs/dataset_schema.md',
    'docs/experiment_registry.md',
    'docs/known_limitations.md',
    'docs/repo_freeze_checklist.md',
    'templates/dataset_manifest.template.json',
    'templates/experiment_manifest.template.json',
    'examples/commands.sh',
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo-root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()

    lines: list[str] = []
    missing: list[str] = []
    for rel in REQUIRED_SCRIPT_NAMES + REQUIRED_CONFIGS + REQUIRED_DOCS:
        path = args.repo_root / rel
        exists = path.exists()
        status = 'OK' if exists else 'MISSING'
        lines.append(f'{rel}: {status}')
        if not exists:
            missing.append(rel)

    out_path = args.repo_root / 'results' / 'logs' / 'reproducibility_check.txt'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')

    if missing:
        raise SystemExit(f'Missing required files: {missing}')


if __name__ == '__main__':
    main()
