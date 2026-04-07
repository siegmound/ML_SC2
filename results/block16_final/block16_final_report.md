# Block 16 — Final Reproducible Freeze Summary

## Formalized current results

| Dataset | Model | Profile | Accuracy | Balanced Acc. | ROC-AUC | Log Loss |
|---|---|---|---:|---:|---:|---:|
| v3_1_fixed | rf | no_counter | 0.6133 ± 0.0055 | 0.6111 ± 0.0051 | 0.6605 ± 0.0055 | 0.6503 ± 0.0023 |
| v3_1_fixed | xgb | full | 0.6128 ± 0.0054 | 0.6093 ± 0.0049 | 0.6601 ± 0.0048 | 0.6484 ± 0.0025 |
| v3_2_combatfix | rf | no_counter | 0.6128 ± 0.0052 | 0.6107 ± 0.0048 | 0.6603 ± 0.0057 | 0.6504 ± 0.0024 |
| v3_2_combatfix | xgb | full | 0.6128 ± 0.0056 | 0.6093 ± 0.0051 | 0.6597 ± 0.0048 | 0.6486 ± 0.0024 |

## Interpretation

- `v3_1_fixed` is not worse than `v3_2_combatfix` in full multi-seed tests; if anything, it is marginally stronger or equivalent across the four reported metrics.
- RF with profile `no_counter` remains the best RF configuration among the tested feature profiles.
- RF and XGB are effectively tied at the project end-state. RF is slightly stronger as a classifier; XGB is slightly stronger in log loss, so it is the better probability-oriented candidate.

## Recommended final model selections

- **Classification-oriented final choice:** RF + `v3_1_fixed` + `no_counter`
- **Probability-oriented final choice:** XGB + `v3_1_fixed` + full feature set

## Remaining work (optional, not required to support the current conclusions)

- Final comparative calibration between RF and XGB on full runs.
- Deep-learning challenger as an extra, not as the main project axis.