# Methodology

## Project goal

Predict the final winner of a StarCraft II match from replay-derived game-state snapshots while preserving replay-level separation across training, validation, and test.

## High-level pipeline

1. collect replay files
2. parse replay metadata and game-state snapshots
3. normalize and engineer tabular features
4. build replay-group-aware splits
5. train candidate models
6. evaluate finalists on untouched replay-aware test sets
7. calibrate probabilities on validation-only predictions
8. collect final metrics, plots, and summary tables

## Replay corpus provenance

The upstream replay corpus used by this remake is documented as **SC2ReSet: StarCraft II Esport Replaypack Set** on Zenodo (`10.5281/zenodo.14963356`, version `2.0.0`).

SC2ReSet is described by its maintainers as the raw replay repository used to generate **SC2EGSet**. The associated SC2EGSet publication describes the collection as containing replays from major and premiere StarCraft II tournaments since 2016.

This repository distinguishes between:

- **upstream replay source**: the public Zenodo replaypack set
- **local raw material**: replay subsets, smoke-test samples, and project-side notes under `data/raw/`
- **processed datasets**: the zipped final tabular datasets under `data/processed/`

For exact source documentation, see:

- `data/raw/replay_sources.md`
- `data/raw/replaypack_inventory.csv`
- `docs/data_provenance.md`
- `data/processed/manifests/upstream_replay_source_manifest.md`

## Evaluation discipline

- replay IDs are the grouping key
- no replay may appear across train/validation/test boundaries
- no hyperparameter selection may use the test set
- calibration must be fit on validation-only data
- final claims should be supported by multi-seed summaries when available

## Dataset policy

Heavy datasets stay zipped in the repo. Replay provenance is documented separately from processed dataset packaging so that the source chain remains explicit even when the entire upstream raw corpus is not versioned inside this repository.
