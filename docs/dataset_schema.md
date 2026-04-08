# Dataset schema

## Prediction artifact schema

Canonical imported prediction files should contain at least:

- `replay_id`
- `time_sec`
- `y_true`

Recommended additional columns:

- one probability column such as `y_prob`, `y_prob_xgb`, `y_prob_rf`, `y_prob_mlp`
- one hard prediction column such as `y_pred`, `y_pred_xgb`, `y_pred_rf`, `y_pred_mlp`

## Replay provenance fields

When replay-level metadata is preserved or exported, the following provenance fields are recommended:

- `source_dataset` (recommended value: `SC2ReSet`)
- `source_record` (recommended value: `14963356`)
- `source_doi` (recommended value: `10.5281/zenodo.14963356`)
- `source_version` (recommended value: `2.0.0`)
- `source_replaypack`
- `source_event_year`

These fields document where the replay came from upstream, even if the final prediction artifact stores only a reduced subset of columns.

## Legacy compatibility note

If only frozen result artifacts are available, the remake treats them as `artifact_only` inputs.

## Upstream source note

The upstream replay source documented for this repository is **SC2ReSet: StarCraft II Esport Replaypack Set** on Zenodo. The replaypack inventory visible on the v2.0.0 record is mirrored in `data/raw/replaypack_inventory.csv`.
