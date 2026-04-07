# Real Dataset Runs

Block 7 connects the remake to real zipped tabular datasets and executes the first canonical runs.

## Main entry points
- `scripts/23_register_real_dataset.py`
- `scripts/24_import_real_calibration_bundle.py`
- `scripts/25_run_real_canonical_pipeline.py`
- `scripts/26_build_block7_status_report.py`

## Recommended order
1. Register the real dataset zip.
2. Run dataset quality report and generate split manifests.
3. Execute one canonical seed on `logreg` and optionally `xgb`.
4. Import the real calibration bundle if available.
5. Build the block 7 status report.
