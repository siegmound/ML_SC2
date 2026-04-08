# Data provenance

## Upstream replay source

The replay corpus used by this project is sourced from **SC2ReSet: StarCraft II Esport Replaypack Set** on Zenodo.

- Record: `14963356`
- DOI: `10.5281/zenodo.14963356`
- Version: `2.0.0`
- Resource type: dataset

The Zenodo description states that SC2ReSet contains the raw data used to generate **SC2EGSet**. The related SC2EGSet paper describes the collection as containing replays from major and premiere StarCraft II tournaments since 2016.

## What is tracked inside this repository

This repository does not attempt to commit the full upstream replay corpus. Instead it tracks:

- provenance documents under `data/raw/`
- local replay subsets and smoke-test material under `data/raw/replay_subsets/`
- final processed tabular datasets under `data/processed/`
- split files and manifests under `data/processed/`

## Inventory

`data/raw/replaypack_inventory.csv` mirrors the replaypack filenames visible on the Zenodo v2.0.0 record, covering the 2016–2024 packs visible on that landing page at audit time.

This inventory is intended to answer the public question: **where did the replay material come from?**

It does **not** claim that every replaypack listed there was fully ingested into the final processed datasets without filtering. Exact local usage still depends on the project pipeline and replay-level validity checks.

## Processing chain

The provenance chain for this repository is:

1. upstream replaypacks from SC2ReSet on Zenodo
2. local raw replay material and project-side subsets
3. parser and feature-building pipeline
4. processed tabular datasets (`v3_1_fixed`, `v3_2_combatfix`)
5. replay-group-aware model training, evaluation, and calibration artifacts

## Legal and licensing note

The Zenodo record states that raw StarCraft II data is subject to the Blizzard EULA and that, in special cases, the Blizzard AI and Machine Learning License may apply. The record itself is labeled as `Other (Non-Commercial)`.

Repository-authored code and documentation should be licensed separately from upstream replay data.
