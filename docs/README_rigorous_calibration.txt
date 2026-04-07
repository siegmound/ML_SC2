RIGOROUS CALIBRATION BUNDLE

Purpose
-------
Run strict post-hoc calibration for the two final candidates:
- XGB on v3_1_fixed full
- RF no_counter on v3_1_fixed full

Methodology
-----------
1. Re-run the finalist models and export BOTH validation and test predictions.
2. Fit calibrators on validation predictions only.
3. Evaluate calibrated probabilities on test predictions only.

Scripts
-------
61_run_xgb_for_calibration.py
    Full-data XGB runner that writes validation_predictions.csv and test_predictions.csv.

62_run_rf_for_calibration.py
    Full-data RF runner that writes validation_predictions.csv and test_predictions.csv.

63_run_rigorous_calibration.py
    Fits:
    - Platt scaling (LogisticRegression)
    - Isotonic regression
    on validation predictions and evaluates on test.

Outputs
-------
results/xgb_calibration/<dataset>/seed_<seed>/
results/rf_calibration/<dataset>/seed_<seed>/
results/final_calibration/<dataset>/<model>/seed_<seed>/

Recommended target
------------------
Use real_v3_1_fixed as the final comparison dataset, since the current project conclusion is that
v3_1_fixed is at least as strong as v3_2_combatfix, and often slightly better.

Notes
-----
- XGB GPU uses QuantileDMatrix with validation/test referencing training, so it is GPU-safe.
- RF profile should remain no_counter for consistency with the final RF conclusion.
- This bundle does NOT change the final ranking benchmark; it only makes calibration rigorous.
