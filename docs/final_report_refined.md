# Final refined report

## Final full multi-seed benchmark

| Dataset | Model | Profile | Accuracy | Balanced Accuracy | ROC-AUC | Log Loss |
|---|---|---|---:|---:|---:|---:|
| v3_1_fixed | RF | no_counter | 0.6133 ± 0.0055 | 0.6111 ± 0.0051 | 0.6605 ± 0.0055 | 0.6503 ± 0.0023 |
| v3_1_fixed | XGB | full | 0.6128 ± 0.0054 | 0.6093 ± 0.0049 | 0.6601 ± 0.0048 | 0.6484 ± 0.0025 |
| v3_2_combatfix | RF | no_counter | 0.6128 ± 0.0052 | 0.6107 ± 0.0048 | 0.6603 ± 0.0057 | 0.6504 ± 0.0024 |
| v3_2_combatfix | XGB | full | 0.6128 ± 0.0069 | 0.6093 ± 0.0062 | 0.6597 ± 0.0059 | 0.6486 ± 0.0030 |

## Final interpretation

Two conclusions are now stable.

1. RF and XGB finish extremely close as the final tabular co-finalists.
2. The final recommendation depends on the downstream objective:
   - use **RF + `v3_1_fixed` + `no_counter`** when classification quality matters most
   - use **XGB + `v3_1_fixed` + full** when probability quality matters most

## Replay provenance note

The replay corpus behind this project is sourced from **SC2ReSet: StarCraft II Esport Replaypack Set** on Zenodo (`10.5281/zenodo.14963356`, record `14963356`, version `2.0.0`), described by its authors as the raw replay repository used to generate **SC2EGSet**.

See `docs/data_provenance.md` and `data/raw/replay_sources.md` for the explicit provenance chain from upstream replaypacks to local processed datasets.
