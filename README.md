# ML_SC2 

Clean-room remake of the StarCraft II outcome-prediction project.

This repository is designed to replace an older patch-heavy project structure with a single canonical pipeline, explicit manifests, replay-group-aware evaluation, and reproducible experiment artifacts.

## Goals

- rebuild the dataset pipeline from scratch
- keep raw/heavy datasets zipped for GitHub size constraints
- make every final claim traceable to saved artifacts
- enforce replay-group-aware splitting and evaluation
- unify training, ablation, calibration, and reporting workflows

## Repository layout

```text
ML_SC2/
├── configs/                 # model and experiment YAML configs
├── data/
│   ├── raw/                 # replay subsets, notes, manifests
│   ├── interim/             # parser audits, feature audits
│   └── processed/           # zipped datasets, manifests, split files
├── docs/                    # methodology, schema, freeze checklist
├── experiments/             # structured outputs from studies
├── results/                 # logs, summaries, tables, figures, predictions
├── scripts/                 # executable entry points
└── src/sc2proj/             # package code
```

## Core principles

- one canonical pipeline
- replay-group-aware evaluation
- explicit reason codes for replay filtering/failures
- zipped heavyweight datasets only
- reproducible outputs for every serious step
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

## End-to-end workflow

### 1) Parser validation

```bash
python scripts/00_parser_smoke_test.py   --replays-zip data/raw/replay_subsets/parser_smoke_test_replays.zip

python scripts/01_parser_audit.py   --replays-dir data/raw/replays   --output-dir experiments/parser_audit
```

### 2) Dataset build and quality report

```bash
python scripts/02_build_dataset.py   --replays-dir data/raw/replays   --dataset-name canonical_v1

python scripts/03_dataset_quality_report.py   --dataset-zip data/processed/canonical_v1.zip
```

### 3) Group-aware split generation

```bash
python scripts/04_make_group_splits.py   --dataset-zip data/processed/canonical_v1.zip   --seeds 42 43 44 45 46
```

### 4) Canonical training

```bash
python scripts/05_train_logreg.py   --dataset-zip data/processed/canonical_v1.zip   --split-json data/processed/splits/split_seed_42.json   --dataset-name canonical_v1

python scripts/06_train_rf.py   --dataset-zip data/processed/canonical_v1.zip   --split-json data/processed/splits/split_seed_42.json   --dataset-name canonical_v1

python scripts/07_train_xgb.py   --dataset-zip data/processed/canonical_v1.zip   --split-json data/processed/splits/split_seed_42.json   --dataset-name canonical_v1   --device cpu

python scripts/08_train_mlp.py   --dataset-zip data/processed/canonical_v1.zip   --split-json data/processed/splits/split_seed_42.json   --dataset-name canonical_v1

python scripts/08_train_mlp_torch.py   --dataset-zip data/processed/canonical_v1.zip   --split-json data/processed/splits/split_seed_42.json   --dataset-name canonical_v1
```

### 5) Experiment studies

```bash
python scripts/09_run_ablation.py --config configs/experiments/ablation.yaml
python scripts/10_run_temporal_study.py --config configs/experiments/temporal.yaml
python scripts/11_run_matchup_study.py --dataset-zip data/processed/canonical_v1.zip
python scripts/12_run_calibration.py --config configs/experiments/calibration.yaml
python scripts/16_run_error_analysis.py --predictions-csv results/predictions/example.csv
```

### 6) Collection, export, and validation

```bash
python scripts/13_collect_results.py --config configs/experiments/collection.yaml
python scripts/14_make_tables_figures.py --results-root results
python scripts/15_verify_reproducibility.py
python scripts/17_internal_audit.py
```

## Standard artifact contract

Every major experiment should emit, at minimum:

- `config_used.json`
- `metrics_summary.json`
- `predictions.csv`
- `raw_log.txt`
- `artifacts_manifest.json`
- split information or split manifest reference

Recommended additions by experiment type:

- feature importance CSVs
- calibration tables/plots
- ablation summaries
- replay-level error summaries
- LaTeX-ready tables

## Freeze candidate checklist

Before declaring the repository frozen:

- parser smoke test passes
- parser audit completed with reason codes
- dataset manifest exists for each official dataset
- feature audit exists for each official dataset
- all final models use group-aware splits
- multi-seed summaries collected
- figures/tables are generated from saved structured outputs
- reproducibility script passes
- internal audit script passes or documents remaining issues
- paper and repo agree on the exact canonical pipeline

See `docs/repo_freeze_checklist.md` for the detailed checklist.


## Block 6: legacy artifact integration

When the raw dataset zip is unavailable but prior frozen artifacts exist, the remake can still ingest and normalize those artifacts.

```bash
python scripts/18_probe_legacy_freeze.py --freeze-zip /path/to/sc2_project_freeze_final.zip
python scripts/19_import_legacy_freeze.py --freeze-zip /path/to/sc2_project_freeze_final.zip
python scripts/20_build_real_artifact_manifests.py
python scripts/21_validate_real_schema.py
python scripts/22_run_real_compatibility_check.py --freeze-zip /path/to/sc2_project_freeze_final.zip
```

This flow is intentionally honest: it does **not** pretend to reconstruct the full dataset when only summaries/predictions are available. Instead, it creates structured manifests and schema reports from the real zipped legacy artifacts currently available.

## Final delivery status

This delivery intentionally excludes the heavyweight datasets due to repository size constraints. To fully reproduce the final results, place these external dataset files under `data/processed/`:

- `starcraft_full_dataset_v3_1_fixed.zip`
- `starcraft_full_dataset_v3_2_combatfix.zip`

### Final recommended entry points

Use the following scripts for the final models and final calibration workflows:

- `scripts/53_run_xgb_full_gpu_fixed_v3.py` — final full-data XGBoost runner
- `scripts/54_run_rf_full.py` — final full-data Random Forest runner
- `scripts/64_run_deep_finalist.py` — cleaned final deep challenger runner
- `scripts/61_run_xgb_for_calibration.py` — XGB validation/test prediction export for rigorous calibration
- `scripts/62_run_rf_for_calibration.py` — RF validation/test prediction export for rigorous calibration
- `scripts/63_run_rigorous_calibration.py` — fit calibration on validation and evaluate on test

### Final model recommendations

- Classification-oriented recommendation: **RF + `v3_1_fixed` + `no_counter`**
- Probability-oriented recommendation: **XGB + `v3_1_fixed` + full features**

### Final result locations

- `docs/final_report_refined.md`
- `docs/final_project_status.md`
- `paper/final_results_refinement.tex`
- `tables/final_full_multiseed_summary.csv`
- `tables/deep_final_summary.csv`
- `tables/rigorous_calibration_summary.csv`
- `results/block16_final/`
- `results/final_calibration/`
- `results/xgb_full/`
- `results/rf_full/`
- `results/deep_final/`
