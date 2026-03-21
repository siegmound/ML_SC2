SC2 Project Final Freeze Package - Public Repository Layout

This repository archives the final paper, code, and frozen experimental artifacts used in the closing benchmark cycle.

Frozen benchmark logic
- Principal dataset branch: V3.1 fixed
- Principal model: XGBoost
- Strongest alternative baseline: Random Forest
- Neural comparison model: PyTorch MLP
- Parser-ablation branch retained for interpretation: V3.2 combatfix

Repository layout
- datasets/  -> compressed dataset archives plus locally extracted CSV inputs
- artifacts/ -> derived tables, prediction exports, JSON summaries, logs, bootstrap outputs
- figures/   -> generated plots used by the paper and artifact audit
- manifests/ -> frozen methodology notes for Tests 1-4
- paper/     -> final TeX/PDF paper and appendix reference material
- scripts/   -> training, export, parser, and evaluation scripts

Datasets
The repository stores the two large datasets as ZIP archives inside datasets/.

ZIP files expected in the repository:
- datasets/starcraft_full_dataset_v3_1_fixed.zip
- datasets/starcraft_full_dataset_v3_2_combatfix.zip

Before running any training or evaluation script that requires raw data, extract both archives locally into the same datasets/ folder so that the CSV files become available at the expected paths.

After extraction, the following files must exist:
- datasets/starcraft_full_dataset_v3_1_fixed.csv
- datasets/starcraft_full_dataset_v3_2_combatfix.csv

Important:
- The Python scripts read the extracted CSV files, not the ZIP archives.
- The extracted CSV files are intended for local use and should not be committed back into the repository.
- If the CSV files are missing, retraining scripts will fail with a file-not-found error.
- The scripts in scripts/ are path-aligned to this repository layout and can be launched from either the repository root or from inside scripts/.
- Prediction exports, summaries, calibration outputs, bootstrap outputs, and logs are read from or written to artifacts/.
- Generated plots are read from or written to figures/.

Recommended local workflow
1. Clone the repository.
2. Go to the datasets/ folder.
3. Extract both ZIP archives there.
4. Verify that the two CSV files are present in datasets/.
5. Run the scripts from the repository root.

Example
python scripts/Xgboost_clean_with_json_v1.py

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
