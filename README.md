# ML_SC2

Clean-room remake of the StarCraft II outcome-prediction project.

This repository replaces the older patch-heavy structure with a canonical remake on `main`, while the historical line is preserved separately in branch `v1.2`.

## Repository status

- `main` is the **v2.0-final remake**
- `v1.2` preserves the **older original branch**
- the repository already contains the final processed dataset archives under `data/processed/`
- final multi-seed tables, calibration summaries, and final runner scripts are committed
- replay-group-aware evaluation is the default discipline for final claims

## Final conclusions

- **Classification-oriented recommendation:** RF + `v3_1_fixed` + `no_counter`
- **Probability-oriented recommendation:** XGB + `v3_1_fixed` + full features
- the deep finalist is competitive, but does not beat the two final tabular candidates

Canonical final summary files:

- `tables/final_full_multiseed_summary.csv`
- `tables/deep_final_summary.csv`
- `tables/rigorous_calibration_summary.csv`
- `docs/final_project_status.md`
- `docs/final_report_refined.md`

## Repository layout

```text
ML_SC2/
├── configs/                 # model and experiment YAML configs
├── data/
│   ├── raw/                 # replay subsets, notes, manifests
│   ├── interim/             # parser audits, feature audits
│   └── processed/           # zipped datasets, manifests, split files
├── docs/                    # methodology, schema, freeze and status docs
├── experiments/             # structured outputs from studies
├── results/                 # logs, summaries, tables, figures, predictions
├── scripts/                 # executable runners and audit tools
└── src/sc2proj/             # package code
```

## Core principles

- one canonical remake on `main`
- replay-group-aware evaluation
- explicit reason codes for replay filtering/failures
- traceable saved artifacts for final claims
- synchronized environment definitions across `requirements.txt`, `pyproject.toml`, and `environment.yml`
- no manual edits to final metrics tables

## Environment setup

### Conda

```bash
conda env create -f environment.yml
conda activate ml-sc2-remake
pip install -e .
```

### pip

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

Both installation paths are intended to support the same runtime stack:
NumPy, pandas, SciPy, scikit-learn, matplotlib, PyYAML, tqdm, joblib, XGBoost,
PyTorch, sc2reader, skorch, and pyarrow.

## End-to-end workflow

### 1) Parser validation

```bash
python scripts/00_parser_smoke_test.py --replays-zip data/raw/replay_subsets/parser_smoke_test_replays.zip
python scripts/01_parser_audit.py --replays-dir data/raw/replays --output-dir experiments/parser_audit
```

### 2) Dataset build and quality report

```bash
python scripts/02_build_dataset.py --replays-dir data/raw/replays --dataset-name canonical_v1
python scripts/03_dataset_quality_report.py --dataset-zip data/processed/canonical_v1.zip
```

### 3) Group-aware split generation

```bash
python scripts/04_make_group_splits.py --dataset-zip data/processed/canonical_v1.zip --seeds 42 43 44 45 46
```

### 4) Canonical training

```bash
python scripts/05_train_logreg.py --dataset-zip data/processed/canonical_v1.zip --split-json data/processed/splits/split_seed_42.json --dataset-name canonical_v1
python scripts/06_train_rf.py --dataset-zip data/processed/canonical_v1.zip --split-json data/processed/splits/split_seed_42.json --dataset-name canonical_v1
python scripts/07_train_xgb.py --dataset-zip data/processed/canonical_v1.zip --split-json data/processed/splits/split_seed_42.json --dataset-name canonical_v1 --device cpu
python scripts/08_train_mlp.py --dataset-zip data/processed/canonical_v1.zip --split-json data/processed/splits/split_seed_42.json --dataset-name canonical_v1
python scripts/08_train_mlp_torch.py --dataset-zip data/processed/canonical_v1.zip --split-json data/processed/splits/split_seed_42.json --dataset-name canonical_v1
```

### 5) Studies and reporting

```bash
python scripts/09_run_ablation.py --config configs/experiments/ablation.yaml
python scripts/10_run_temporal_study.py --config configs/experiments/temporal.yaml
python scripts/11_run_matchup_study.py --dataset-zip data/processed/canonical_v1.zip
python scripts/12_run_calibration.py --config configs/experiments/calibration.yaml
python scripts/13_collect_results.py --config configs/experiments/collection.yaml
python scripts/14_make_tables_figures.py --results-root results
python scripts/15_verify_reproducibility.py
python scripts/17_internal_audit.py
```

## Final datasets

The final processed dataset archives are currently committed under `data/processed/`:

- `starcraft_full_dataset_v3_1_fixed.zip`
- `starcraft_full_dataset_v3_2_combatfix.zip`

Smoke and intermediate subsets are also present for staged testing and debugging.

## Final recommended entry points

Use these scripts for the final full-data comparison and rigorous calibration workflow:

- `scripts/53_run_xgb_full_gpu_fixed_v3.py`
- `scripts/54_run_rf_full.py`
- `scripts/64_run_deep_finalist.py`
- `scripts/61_run_xgb_for_calibration.py`
- `scripts/62_run_rf_for_calibration.py`
- `scripts/63_run_rigorous_calibration.py`

## Freeze and audit

- `docs/repo_freeze_checklist.md` documents the freeze-signoff process
- `docs/known_limitations.md` records the remaining public limitations
- `scripts/15_verify_reproducibility.py` checks required artifacts and dependency alignment
- `scripts/17_internal_audit.py` checks Python syntax, empty files, task markers, and style-risk warnings

## Legacy integration

When an older frozen artifact bundle is available, the remake can probe and normalize it without pretending to regenerate missing raw data:

```bash
python scripts/18_probe_legacy_freeze.py --freeze-zip /path/to/sc2_project_freeze_final.zip
python scripts/19_import_legacy_freeze.py --freeze-zip /path/to/sc2_project_freeze_final.zip
python scripts/20_build_real_artifact_manifests.py
python scripts/21_validate_real_schema.py
python scripts/22_run_real_compatibility_check.py --freeze-zip /path/to/sc2_project_freeze_final.zip
```

## Notes for public presentation

Before treating the repository as fully polished for external reviewers, also apply the non-file actions documented in `docs/repo_fix_plan_v2_0.md`.
