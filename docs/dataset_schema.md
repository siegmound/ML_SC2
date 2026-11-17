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

## B3 feature-schema mapping

The canonical schema comparison is `data/processed/manifests/b3_feature_lineage_v1.json`, generated from the parser's actual snapshot dictionary and the saved full-XGB model feature configurations.

| Source | Predictive columns | Evidence |
|---|---:|---|
| Current `src/sc2proj/parser.py` | 38 | Snapshot dictionary parsed as Python AST |
| Frozen `v3_1_fixed` | 64 | v3.1 ZIP header and saved model configurations |
| Frozen `v3_2_combatfix` | 73 | Saved v3.2 full-model configurations, three seeds |

Of the 64 v3.1 feature names, 35 do not exist in current parser output; v3.2 adds nine features. This is **schema reconciliation, not a reconstruction of formulas**. Names are treated as predictive features even if they imply engineered ratios, trends, counter heuristics or time phases. No formula is inferred solely from a column name.

Run `python scripts/68_verify_feature_lineage.py` to detect unexpected schema changes. Validate actual ZIP headers with the optional `--zip-v3-1` and `--zip-v3-2` arguments when complete archives are available.
