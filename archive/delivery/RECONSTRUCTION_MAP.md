# Reconstruction map

1. Place external datasets in `data/processed/`.
2. Use `scripts/53_run_xgb_full_gpu_fixed_v3.py` for final XGB full runs.
3. Use `scripts/54_run_rf_full.py` for final RF full runs.
4. Use `scripts/64_run_deep_finalist.py` for the final deep challenger.
5. Use `scripts/61_run_xgb_for_calibration.py`, `62_run_rf_for_calibration.py`, and `63_run_rigorous_calibration.py` for strict calibration.
6. Read `docs/final_report_refined.md` and `docs/final_project_status.md` for the final narrative.
