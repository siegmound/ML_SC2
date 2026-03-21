SC2 Project Final Freeze Package - Revised Audit Cleanup

This package archives the final paper and the experimental artifacts used in the closing benchmark cycle.

Frozen benchmark logic
- Principal dataset branch: V3.1 fixed
- Principal model: XGBoost
- Strongest alternative baseline: Random Forest
- Neural comparison model: PyTorch MLP
- Parser-ablation branch retained for interpretation: V3.2 combatfix

Important reproducibility note
- The large dataset CSV files are intentionally NOT bundled in this revised package because of size constraints.
- The package is therefore artifact-complete for paper audit, metric verification, and script inspection, but full end-to-end retraining still requires the external dataset files.
- A public repository can later attach the dataset paths or release instructions without changing the frozen result files archived here.

Artifact notes
- Original prediction exports and additional logs have been reintroduced at top level under artifacts/ for easier auditability.
- Some legacy ZIP bundles are preserved, but their contents are also unpacked in artifacts/ where possible.
- The XGBoost prediction export is normalized in this package as artifacts/xgb_clean_v3_1_fixed_test_predictions.csv. The originally uploaded filename xgb_test_clean_v3_1_fixed_test_predictions.csv is also preserved.
- Two Random Forest artifact lines are intentionally retained:
  * artifacts/rf_clean_v3_1_summary.json = aligned benchmark summary used for the cross-model comparison table.
  * artifacts/rf_test_clean_v3_1_summary.json and artifacts/rf_test_clean_v3_1_test_predictions.csv = export-preserved RF run used in Tests 1-4.
- The parser branch now uses the corrected constants metadata and an upgrade-multiplier implementation compatible with those constants.
- A generated XGBoost feature-importance figure is now included in figures/xgb_feature_importance_v3_1_fixed.png based on the preserved importance CSV.

Recommended reading order
1. paper/sc2_ml_paper_final.pdf
2. artifacts/model_comparison_summary_v2.csv
3. manifests/test1_methodology_xgb_vs_rf.txt through manifests/test4_methodology_calibration_xgb_rf_mlp.txt
4. scripts/ for implementation details
