# Upstream replay source manifest

## Purpose

This manifest links the final processed datasets committed in `data/processed/` back to the documented upstream replay source.

## Upstream replay source

- dataset: **SC2ReSet: StarCraft II Esport Replaypack Set**
- record: `14963356`
- DOI: `10.5281/zenodo.14963356`
- version: `2.0.0`

## Local processed datasets

- `starcraft_full_dataset_v3_1_fixed.zip`
- `starcraft_full_dataset_v3_2_combatfix.zip`

## Interpretation

The processed datasets above are downstream tabular products of the project pipeline, not direct mirrors of the upstream replaypacks.

They should therefore be understood as:

- sourced from the SC2ReSet replay corpus
- filtered and transformed by the project parser and feature pipeline
- subject to replay-level exclusions, parser-side validity checks, and project-side dataset construction decisions

## Related repository files

- `data/raw/replay_sources.md`
- `data/raw/replaypack_inventory.csv`
- `docs/data_provenance.md`
