# ML_SC2

Clean-room remake of a StarCraft II outcome-prediction project based on replay snapshots and replay-group-aware evaluation.

This repository uses `main` as the canonical **v2.0-final remake**.
The older historical line is preserved separately in branch `v1.2`.

## Repository status

- `main` contains the cleaned remake intended for final delivery.
- `v1.2` preserves the older original structure for historical reference.
- Final processed dataset archives are committed under `data/processed/`.
- Final multi-seed tables, calibration summaries, and final runner scripts are committed.
- Replay-group-aware evaluation is the default discipline for final claims.

## Final conclusions

- **Classification-oriented recommendation:** Random Forest + `v3_1_fixed` + `no_counter`
- **Probability-oriented recommendation:** XGBoost + `v3_1_fixed` + full features
- The deep finalist is competitive, but does not beat the two final tabular candidates.

Canonical final summary files:

- `tables/final_full_multiseed_summary.csv`
- `tables/deep_final_summary.csv`
- `tables/rigorous_calibration_summary.csv`
- `docs/final_project_status.md`
- `docs/final_report_refined.md`

## Replay source

The replay corpus documented by this project is sourced from **SC2ReSet: StarCraft II Esport Replaypack Set** on Zenodo.

- Record: `14963356`
- DOI: `10.5281/zenodo.14963356`
- Version referenced in this repository: `2.0.0`
- Zenodo landing page: `https://zenodo.org/records/14963356`

SC2ReSet is described by its authors as the raw replay repository used to generate **SC2EGSet**.
The associated SC2EGSet publication describes the collection as containing replays from major and premiere StarCraft II tournaments since 2016.

Examples of replaypacks visible on the Zenodo record include:

- `2017_IEM_XI_World_Championship_Katowice.zip`
- `2017_WCS_Global_Finals.zip`
- `2018_WCS_Global_Finals.zip`
- `2019_WCS_Grand_Finals.zip`
- `2024_01_IEM_Katowice.zip`

Related provenance documents:

- `data/raw/replay_sources.md`
- `data/raw/replaypack_inventory.csv`
- `docs/data_provenance.md`
- `data/processed/manifests/upstream_replay_source_manifest.md`

License note: raw StarCraft II replay data remains subject to the Blizzard EULA and, where applicable, the Blizzard AI and Machine Learning License.
The upstream Zenodo record is labeled as `Other (Non-Commercial)`.

## Repository layout

```text
ML_SC2/
├── configs/                    # model and experiment YAML configs
├── data/
│   ├── raw/                    # replay subsets, provenance docs, inventories
│   ├── interim/                # parser audits, feature audits
│   └── processed/              # zipped datasets, manifests, split files
├── docs/                       # methodology, schema, freeze and status docs
├── experiments/                # structured outputs from studies
├── results/                    # logs, summaries, tables, figures, predictions
├── scripts/                    # executable runners and audit tools
├── tables/                     # final published summary tables
└── src/sc2proj/                # package code
```

## Core principles

- one canonical remake on `main`
- replay-group-aware evaluation
- explicit reason codes for replay filtering and parser failures
- traceable saved artifacts for final claims
- synchronized environment definitions across `requirements.txt`, `pyproject.toml`, and `environment.yml`
- no manual edits to final metrics tables
- documented upstream replay provenance

## Environment setup

### Conda

```bash
conda env create -f environment.yml
conda activate ml-sc2-remake
pip install -e .
```

### pip / virtualenv

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

On Windows PowerShell, replace the activation command with:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Final validation commands

```bash
python scripts/15_verify_reproducibility.py --strict
python scripts/17_internal_audit.py --strict-style
```

## Paper and delivery artifacts

Canonical paths:

- editable paper source: `paper/sc2_outcome_prediction_paper.tex`
- compiled paper: `paper/sc2_outcome_prediction_paper.pdf`
- final presentation: `paper/sc2_outcome_prediction_presentation.pptx`
- paper artifact guide: `paper/README_paper_repo.md`

During the B1 correction branch, the source LaTeX and published CSV/Markdown tables
were corrected to match the saved Block 16 population standard deviations. The
existing compiled PDF and PowerPoint remain historical exports until the B5
regeneration and visual verification; do not cite them as B1-corrected artifacts.

Reproduce and check the table without training:

```bash
python scripts/58_verify_final_table.py --check
```

## Provenance and scope note

The repository documents the upstream replay source and the replaypack inventory visible on the Zenodo `v2.0.0` record.

Exact reconstruction of the locally used subset still depends on project-side filtering choices, parser validity checks, and any replay-level exclusions recorded by the project pipeline.

## B2 methodology corrections (working branch)

The original results remain frozen. The original full tabular comparison is not a controlled equal-training-budget test: RF fits train+validation whereas XGB fits train and uses validation for early stopping.

A separate *retrospective* matched train-only comparison was reproduced from six saved RF/XGB run directories, with replay- and snapshot-level identity checks. Its published table can be validated without training:

```bash
python scripts/66_check_matched_train_budget.py --check
```

The corrected Deep MLP runner now evaluates test only for the validation-selected candidate (default output under `results/b2_deep_strict`); historical deep metrics remain archived.

For stricter future calibration, `scripts/67_run_strict_calibration.py` deterministically partitions original validation replay groups into **tuning** and **calibration** subsets. No full-data B2 recalibration or Deep rerun has been claimed; use `docs/audit/B2_METHODS_2026-11-17.md` for exact commands, tests, and scientific caveats. Existing PDF/PPTX exports are awaiting B5 regeneration.

## B3 parser and frozen-dataset provenance

B3 hardens the replay parser: unknown winners are filtered, recent losses follow time-aware live tags, and the builder refuses to overwrite frozen dataset archives or merge distinct replays sharing a basename.

Machine-readable lineage: `data/processed/manifests/b3_feature_lineage_v1.json`.

```bash
python scripts/68_verify_feature_lineage.py
python scripts/69_audit_replay_lineage.py --processed-zip data/processed/starcraft_full_dataset_v3_1_fixed.zip
python -m unittest discover -s tests -p 'test_*.py' -v
```

The feature-lineage checker compares the current clean-room **38** predictors to archived **64** (v3.1) and **73** (v3.2). The original feature-engineering transformations have not all been recovered, so the clean-room replay-to-dataset route is **not a full reproduction of either final dataset**.

The preserved v3.1 dataset was independently scanned: 892,047 snapshots, 23,241 distinct replay IDs, no contradictory winners or duplicate replay/time keys. Original exclusions (24,057 documented upstream vs 23,241 final) cannot be attributed without a source replay audit. See `docs/audit/B3_PARSER_AND_PROVENANCE_2026-11-17.md`.
