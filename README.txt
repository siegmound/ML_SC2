SC2 Project Final Freeze Package - Public Repository Layout

This repository archives the final paper, code, and frozen experimental artifacts used in the closing benchmark cycle.

Frozen benchmark logic
- Principal dataset branch: V3.1 fixed
- Principal model: XGBoost
- Strongest alternative baseline: Random Forest
- Neural comparison model: PyTorch MLP
- Parser-ablation branch retained for interpretation: V3.2 combatfix

Repository layout
- datasets/  -> source CSV inputs tracked with Git LFS
- artifacts/ -> derived tables, prediction exports, JSON summaries, logs, bootstrap outputs
- figures/   -> generated plots used by the paper and artifact audit
- manifests/ -> frozen methodology notes for Tests 1-4
- paper/     -> final TeX/PDF paper and appendix reference material
- scripts/   -> training, export, parser, and evaluation scripts

Important reproducibility note
- Full end-to-end retraining requires the two dataset files to be present under datasets/:
  * datasets/starcraft_full_dataset_v3_1_fixed.csv
  * datasets/starcraft_full_dataset_v3_2_combatfix.csv
- The scripts in scripts/ are now path-aligned to this repository layout and can be launched from either the repository root or from inside scripts/.
- All prediction exports, summaries, and calibration/bootstrap outputs are read from or written to artifacts/.
- All generated plots are read from or written to figures/.

Artifact notes
- The normalized XGBoost prediction export is artifacts/xgb_clean_v3_1_fixed_test_predictions.csv.
- Two Random Forest artifact lines are intentionally retained:
  * artifacts/rf_clean_v3_1_summary.json = aligned benchmark summary used for the cross-model comparison table.
  * artifacts/rf_test_clean_v3_1_summary.json and artifacts/rf_test_clean_v3_1_test_predictions.csv = export-preserved RF run used in Tests 1-4.
- The parser branch uses the corrected constants metadata and an upgrade-multiplier implementation compatible with those constants.
- The XGBoost feature-importance figure is included as figures/xgb_feature_importance_v3_1_fixed.png.

Recommended reading order
1. paper/sc2_ml_paper_final.pdf
2. artifacts/model_comparison_summary_v2.csv
3. manifests/test1_methodology_xgb_vs_rf.txt through manifests/test4_methodology_calibration_xgb_rf_mlp.txt
4. scripts/ for implementation details
