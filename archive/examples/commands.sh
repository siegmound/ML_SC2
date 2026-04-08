#!/usr/bin/env bash
set -euo pipefail

python scripts/00_parser_smoke_test.py --replays-zip data/raw/replay_subsets/parser_smoke_test_replays.zip
python scripts/01_parser_audit.py --replays-dir data/raw/replays --output-dir experiments/parser_audit
python scripts/02_build_dataset.py --replays-dir data/raw/replays --dataset-name canonical_v1
python scripts/03_dataset_quality_report.py --dataset-zip data/processed/canonical_v1.zip
python scripts/04_make_group_splits.py --dataset-zip data/processed/canonical_v1.zip --seeds 42 43 44 45 46
python scripts/07_train_xgb.py --dataset-zip data/processed/canonical_v1.zip --split-json data/processed/splits/split_seed_42.json --dataset-name canonical_v1 --device cpu
python scripts/13_collect_results.py --config configs/experiments/collection.yaml
python scripts/14_make_tables_figures.py --results-root results
python scripts/15_verify_reproducibility.py
python scripts/17_internal_audit.py
