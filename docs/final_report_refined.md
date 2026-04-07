# Final Report Refinement

## Final experimental status

The project is now closed at the level of **full multi-seed comparison** for the two main tabular finalists:

- **Random Forest** with profile `no_counter`
- **XGBoost** with the full feature set

Both were evaluated on the two final dataset variants:

- `v3_1_fixed`
- `v3_2_combatfix`

using **three full replay-aware seeds (42, 43, 44)**.

## Final full multi-seed results

| Dataset | Model | Profile | Accuracy | Balanced Acc. | ROC-AUC | Log Loss |
|---|---|---|---:|---:|---:|---:|
| v3_1_fixed | RF | no_counter | 0.6133 ± 0.0055 | 0.6111 ± 0.0051 | 0.6605 ± 0.0055 | 0.6503 ± 0.0023 |
| v3_1_fixed | XGB | full | 0.6128 ± 0.0054 | 0.6093 ± 0.0049 | 0.6601 ± 0.0048 | 0.6484 ± 0.0025 |
| v3_2_combatfix | RF | no_counter | 0.6128 ± 0.0052 | 0.6107 ± 0.0048 | 0.6603 ± 0.0057 | 0.6504 ± 0.0024 |
| v3_2_combatfix | XGB | full | 0.6128 ± 0.0069 | 0.6093 ± 0.0062 | 0.6597 ± 0.0059 | 0.6486 ± 0.0030 |

## Final interpretation

Two conclusions are now stable.

1. **`v3_1_fixed` is at least as strong as `v3_2_combatfix`** in the full multi-seed regime.  
   The earlier staged indication that `v3_2_combatfix` might dominate does **not** survive the final full-scale comparison.

2. **RF and XGB are effectively tied at project end-state**, but with a clean trade-off:
   - **RF** is marginally stronger on classification-oriented metrics (accuracy, balanced accuracy, ROC-AUC).
   - **XGB** is marginally stronger on probability-oriented metrics (log loss).

## Final Random Forest feature-profile conclusion

The strongest RF configuration remains:

- **RF + `no_counter`**

This conclusion survives staged testing, larger staged subsets, and the final full multi-seed evaluation.

## Deep-learning challenger

The deep challenger was evaluated on **full `v3_1_fixed`**, with three full replay-aware seeds and validation-based candidate selection.

| Dataset | Model | Profile | Accuracy | Balanced Acc. | ROC-AUC | Log Loss |
|---|---|---|---:|---:|---:|---:|
| v3_1_fixed | Deep finalist | best candidate per seed | 0.6093 ± 0.0051 | 0.6083 ± 0.0042 | 0.6581 ± 0.0041 | 0.6506 ± 0.0025 |

Interpretation:
- the deep model is **competitive**
- but it remains **slightly below** both RF and XGB
- therefore it should be presented as a **credible challenger, not as the final winner**

## Rigorous calibration conclusion

Calibration was performed rigorously by fitting post-hoc calibrators on validation predictions and evaluating them on test predictions.

- RF benefits noticeably from post-hoc calibration.
- XGB is already well calibrated in its uncalibrated form.
- Post-hoc calibration does **not** produce meaningful gains for XGB.

## Final model recommendations

### If the priority is classification
**Recommended final model:**  
**RF + `v3_1_fixed` + `no_counter`**

### If the priority is probability quality
**Recommended final model:**  
**XGB + `v3_1_fixed` + full feature set**

## Final one-paragraph conclusion

The final full multi-seed evaluation shows that `v3_1_fixed` is at least as strong as `v3_2_combatfix`, and that the project ends with two essentially co-finalist tabular models. Random Forest with the `no_counter` profile is marginally stronger as a classifier, while XGBoost with the full feature set is marginally stronger as a probabilistic model and remains well calibrated without significant post-hoc improvement. The deep challenger is competitive but does not surpass the two tabular finalists. Accordingly, the most defensible classification-oriented recommendation is **RF + `v3_1_fixed` + `no_counter`**, while the most defensible probability-oriented recommendation is **XGB + `v3_1_fixed` + full features**.
