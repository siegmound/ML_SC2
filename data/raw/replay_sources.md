# Replay sources

## Canonical upstream source

The replay corpus used by this project is sourced from:

- **SC2ReSet: StarCraft II Esport Replaypack Set**
- Zenodo record: `14963356`
- DOI: `10.5281/zenodo.14963356`
- Version: `2.0.0`
- Published: `2025-03-03`

Zenodo description summary:

- SC2ReSet is described by its authors as the raw data repository used to generate **SC2EGSet**
- the associated SC2EGSet publication describes the collection as containing replays from major and premiere StarCraft II tournaments since 2016

## Scope note

This repository uses SC2ReSet as the documented upstream replay source. The exact subset that survives into the final processed datasets may differ from the full Zenodo inventory because of:

- parser compatibility checks
- corruption or unreadable replay exclusions
- feature-building requirements
- project-side filtering decisions

## Legal note

The Zenodo record states that raw StarCraft II data is subject to the Blizzard EULA and that, in special cases, the Blizzard AI and Machine Learning License may apply.

## Repository cross-references

- `data/raw/replaypack_inventory.csv`
- `docs/data_provenance.md`
- `data/processed/manifests/upstream_replay_source_manifest.md`
