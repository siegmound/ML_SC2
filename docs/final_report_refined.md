# Final refined report

## Final full multi-seed benchmark

| Dataset | Model | Profile | Accuracy | Balanced Accuracy | ROC-AUC | Log Loss |
|---|---|---|---:|---:|---:|---:|
| v3_1_fixed | RF | no_counter | 0.6133 ± 0.0055 | 0.6111 ± 0.0051 | 0.6605 ± 0.0055 | 0.6503 ± 0.0023 |
| v3_1_fixed | XGB | full | 0.6128 ± 0.0054 | 0.6093 ± 0.0049 | 0.6601 ± 0.0048 | 0.6484 ± 0.0025 |
| v3_2_combatfix | RF | no_counter | 0.6128 ± 0.0052 | 0.6107 ± 0.0048 | 0.6603 ± 0.0057 | 0.6504 ± 0.0024 |
| v3_2_combatfix | XGB | full | 0.6128 ± 0.0056 | 0.6093 ± 0.0051 | 0.6597 ± 0.0048 | 0.6486 ± 0.0024 |

## Final interpretation

Two conclusions are now stable.

1. RF and XGB finish extremely close as the final tabular co-finalists.
2. The final recommendation depends on the downstream objective:
   - use **RF + `v3_1_fixed` + `no_counter`** when classification quality matters most
   - use **XGB + `v3_1_fixed` + full** when probability quality matters most

## Replay provenance note

The replay corpus behind this project is sourced from **SC2ReSet: StarCraft II Esport Replaypack Set** on Zenodo (`10.5281/zenodo.14963356`, record `14963356`, version `2.0.0`), described by its authors as the raw replay repository used to generate **SC2EGSet**.

See `docs/data_provenance.md` and `data/raw/replay_sources.md` for the explicit provenance chain from upstream replaypacks to local processed datasets.

## B2 matched-fit retrospective sensitivity analysis

The original final tabular table above uses asymmetric fit budgets (RF train+validation, XGB train+validation for early stopping but train only for fitted trees). B2 verifies six **historical train-only** RF/XGB runs against their stored per-snapshot predictions. Training rows, validation rows, test replay IDs, snapshot keys and labels match between both models for each seed.

| Metric (3-seed mean) | RF no_counter (train only) | XGB full (train only) |
|---|---:|---:|
| Accuracy | 0.612270 | 0.612792 |
| Balanced accuracy | 0.610068 | 0.609328 |
| ROC-AUC | 0.658147 | 0.660058 |
| Log loss (lower better) | 0.651698 | 0.648437 |

These differences are small and **do not establish statistical significance**. XGB used validation for early stopping; identical train rows alone do not mean identical model-selection procedures. The test set had been inspected during historic development, so this retrospective sensitivity analysis is not a new blind evaluation.

For calibration, XGB historically reuses validation for early stopping and post-hoc fitting. A four-way replay-disjoint B2 pipeline (train/tune/calibrate/test) is now implemented and smoke-tested, but full-data reruns are still required. Original calibration numbers remain unchanged.
