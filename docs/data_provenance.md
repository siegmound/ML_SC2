# Data provenance

## Upstream replay source

The replay corpus referenced by this project comes from **SC2ReSet: StarCraft II Esport Replaypack Set** on Zenodo.

- Zenodo record: `14963356`
- DOI: `10.5281/zenodo.14963356`
- Version referenced in this repository: `2.0.0`
- Landing page: `https://zenodo.org/records/14963356`

SC2ReSet is described by its authors as the raw replay repository used to generate **SC2EGSet**.
The associated SC2EGSet publication describes the collection as containing replays from major and premiere StarCraft II tournaments since 2016.

## Replaypack inventory

The repository tracks visible replaypack provenance in:

- `data/raw/replaypack_inventory.csv`
- `data/raw/replay_sources.md`

This inventory is meant to document the upstream public source and the replaypack-level context available from the Zenodo record.

## Local raw material

The repository may also contain local replay subsets or smoke-test material under `data/raw/`.
These local subsets are not, by themselves, a full substitute for the upstream replaypack release.

## Processed datasets

The final processed dataset archives currently committed in the repository are:

- `data/processed/starcraft_full_dataset_v3_1_fixed.zip`
- `data/processed/starcraft_full_dataset_v3_2_combatfix.zip`

The bridge between upstream replay provenance and the processed datasets is documented in:

- `data/processed/manifests/upstream_replay_source_manifest.md`

## Scope note

Public provenance is now documented at the dataset-source level.
Exact reconstruction of the final local corpus still depends on project-side filtering, parser validity checks, replay-level exclusions, and any subset decisions made during the project workflow.
