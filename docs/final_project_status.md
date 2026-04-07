# Final Project Status

## Final experimental closure

The project is experimentally closed for the main comparison axes:
- model family: Random Forest vs XGBoost
- dataset version: `v3_1_fixed` vs `v3_2_combatfix`
- final challenger: deep MLP on full `v3_1_fixed`
- probability quality: rigorous post-hoc calibration on validation and evaluation on test

## Main final results

### Full multi-seed tabular finalists
- RF + `v3_1_fixed` + `no_counter`
  - Accuracy: 0.6133 ± 0.0055
  - Balanced Accuracy: 0.6111 ± 0.0051
  - ROC-AUC: 0.6605 ± 0.0055
  - Log Loss: 0.6503 ± 0.0023

- XGB + `v3_1_fixed` + full
  - Accuracy: 0.6128 ± 0.0054
  - Balanced Accuracy: 0.6093 ± 0.0049
  - ROC-AUC: 0.6601 ± 0.0048
  - Log Loss: 0.6484 ± 0.0025

### Full multi-seed on `v3_2_combatfix`
- RF + `no_counter`
  - Accuracy: 0.6128 ± 0.0052
  - Balanced Accuracy: 0.6107 ± 0.0048
  - ROC-AUC: 0.6603 ± 0.0057
  - Log Loss: 0.6504 ± 0.0024

- XGB + full
  - Accuracy: 0.6128 ± 0.0069
  - Balanced Accuracy: 0.6093 ± 0.0062
  - ROC-AUC: 0.6597 ± 0.0059
  - Log Loss: 0.6486 ± 0.0030

## Final interpretation
- `v3_1_fixed` is at least as strong as `v3_2_combatfix` at final scale.
- RF and XGB are effectively tied as final tabular candidates.
- RF is marginally stronger as a classifier.
- XGB is marginally stronger as a probability-oriented model.
- The best RF profile remains `no_counter`.
- The deep challenger is competitive, but does not surpass RF or XGB.

## Calibration conclusion
- RF benefits from post-hoc calibration.
- XGB is already well calibrated in uncalibrated form.
- Therefore:
  - classification-oriented recommendation: **RF + `v3_1_fixed` + `no_counter`**
  - probability-oriented recommendation: **XGB + `v3_1_fixed` + full**
