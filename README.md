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

## Replay source

The replay corpus used by this project is sourced from **SC2ReSet: StarCraft II Esport Replaypack Set** on Zenodo.

- Record: `14963356`
- DOI: `10.5281/zenodo.14963356`
- Version referenced here: `2.0.0`
- Zenodo landing page: `https://zenodo.org/records/14963356`

SC2ReSet is described by its authors as the raw replay repository used to generate **SC2EGSet**. The associated SC2EGSet publication describes the collection as containing replays from major and premiere StarCraft II tournaments since 2016.

Examples of replaypacks visible on the Zenodo record include:

- `2017_IEM_XI_World_Championship_Katowice.zip`
- `2017_WCS_Global_Finals.zip`
- `2018_WCS_Global_Finals.zip`
- `2019_WCS_Grand_Finals.zip`
- `2024_01_IEM_Katowice.zip`

See also:

- `data/raw/replay_sources.md`
- `data/raw/replaypack_inventory.csv`
- `docs/data_provenance.md`
- `data/processed/manifests/upstream_replay_source_manifest.md`

License note: raw StarCraft II data remains subject to the Blizzard EULA and, where applicable, the Blizzard AI and Machine Learning License. The upstream Zenodo record is labeled as `Other (Non-Commercial)`.

## Repository layout

```text
ML_SC2/
├── configs/                 # model and experiment YAML configs
├── data/
│   ├── raw/                 # replay subsets, provenance docs, inventories
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

- final paper PDF: `paper/final_sc2_remake_paper_v6_2.pdf`
- final presentation: `paper/sc2_project_rigorous_presentation.pptx`
- refinement snippet: `paper/final_results_refinement.tex`
- packaging note: `paper/README.md`

## Provenance and scope note

The repository documents the upstream replay source and the replaypack inventory visible on the Zenodo v2.0.0 record. Exact reconstruction of the locally used subset still depends on project-side filtering choices, parser validity checks, and any replay-level exclusions recorded by the project pipeline.
