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

## B3 integrity and lineage evidence (2026-11-17)

B3 adds machine-readable evidence without modifying the archived datasets:

- `data/processed/manifests/b3_feature_lineage_v1.json`: schema comparison between the current clean-room `SnapshotParser` and all six saved XGB full-feature configurations (three seeds for each dataset variant).
- `scripts/68_verify_feature_lineage.py`: detects schema drift; can additionally check a real dataset ZIP header using `--zip-v3-1` and `--zip-v3-2`.
- `data/processed/manifests/b3_replay_census_v3_1_fixed.json`: direct streaming census of the frozen v3.1 ZIP.
- `scripts/69_audit_replay_lineage.py`: counts replay IDs and snapshots and refuses to certify source replay exclusions without a genuine raw-parser audit.

The clean-room snapshot builder currently emits **38 predictive fields**. The archived v3.1 model uses **64** fields: **29 field names overlap** and **35 final fields are not emitted** by the clean-room parser. Nine clean-room fields do not appear in v3.1. The v3.2 full-model configurations add **nine** feature names to v3.1 (73 in total). Matching names do **not** establish matching feature formulas or values.

A full streaming scan of the local frozen v3.1 archive found **892,047 snapshots**, **23,241 replay IDs**, no duplicate replay/timestamp keys, no conflicting winner labels, and snapshot times from **135** to **1,785** seconds. The upstream total of **24,057** replays remains a documented historical source count, not a complete per-file raw audit verified here. The arithmetic difference of **816** replays has no evidenced per-replay exclusion reasons in the current repo.

**Unresolved**: trace each original replaypack/file into included or excluded IDs with source-level reason codes; recover and test the exact historical feature engineering and combatfix transformations; recheck the real v3.2 archive through a clean Git LFS checkout. The present clean-room builder must **not** be described as a reproduction of the frozen 64/73-feature datasets.
