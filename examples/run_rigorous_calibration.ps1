$env:PYTHONPATH = "src"

# XGB finalists on v3_1_fixed full
python scripts/61_run_xgb_for_calibration.py `
  --dataset-zip data/processed/starcraft_full_dataset_v3_1_fixed.zip `
  --split-json data/processed/splits/real_v3_1_fixed_split_seed_42.json `
  --dataset-name real_v3_1_fixed `
  --device cuda `
  --profile full

python scripts/61_run_xgb_for_calibration.py `
  --dataset-zip data/processed/starcraft_full_dataset_v3_1_fixed.zip `
  --split-json data/processed/splits/real_v3_1_fixed_split_seed_43.json `
  --dataset-name real_v3_1_fixed `
  --device cuda `
  --profile full

python scripts/61_run_xgb_for_calibration.py `
  --dataset-zip data/processed/starcraft_full_dataset_v3_1_fixed.zip `
  --split-json data/processed/splits/real_v3_1_fixed_split_seed_44.json `
  --dataset-name real_v3_1_fixed `
  --device cuda `
  --profile full

# RF finalists on v3_1_fixed full
python scripts/62_run_rf_for_calibration.py `
  --dataset-zip data/processed/starcraft_full_dataset_v3_1_fixed.zip `
  --split-json data/processed/splits/real_v3_1_fixed_split_seed_42.json `
  --dataset-name real_v3_1_fixed `
  --profile no_counter `
  --n-estimators 700 `
  --max-depth 24 `
  --min-samples-split 4 `
  --min-samples-leaf 2 `
  --max-features sqrt `
  --class-weight balanced_subsample

python scripts/62_run_rf_for_calibration.py `
  --dataset-zip data/processed/starcraft_full_dataset_v3_1_fixed.zip `
  --split-json data/processed/splits/real_v3_1_fixed_split_seed_43.json `
  --dataset-name real_v3_1_fixed `
  --profile no_counter `
  --n-estimators 700 `
  --max-depth 24 `
  --min-samples-split 4 `
  --min-samples-leaf 2 `
  --max-features sqrt `
  --class-weight balanced_subsample

python scripts/62_run_rf_for_calibration.py `
  --dataset-zip data/processed/starcraft_full_dataset_v3_1_fixed.zip `
  --split-json data/processed/splits/real_v3_1_fixed_split_seed_44.json `
  --dataset-name real_v3_1_fixed `
  --profile no_counter `
  --n-estimators 700 `
  --max-depth 24 `
  --min-samples-split 4 `
  --min-samples-leaf 2 `
  --max-features sqrt `
  --class-weight balanced_subsample

# Post-hoc calibration on validation, evaluate on test
python scripts/63_run_rigorous_calibration.py `
  --validation-predictions results/xgb_calibration/real_v3_1_fixed/seed_42/validation_predictions.csv `
  --test-predictions results/xgb_calibration/real_v3_1_fixed/seed_42/test_predictions.csv `
  --model-name xgb_full `
  --dataset-name real_v3_1_fixed `
  --seed 42

python scripts/63_run_rigorous_calibration.py `
  --validation-predictions results/xgb_calibration/real_v3_1_fixed/seed_43/validation_predictions.csv `
  --test-predictions results/xgb_calibration/real_v3_1_fixed/seed_43/test_predictions.csv `
  --model-name xgb_full `
  --dataset-name real_v3_1_fixed `
  --seed 43

python scripts/63_run_rigorous_calibration.py `
  --validation-predictions results/xgb_calibration/real_v3_1_fixed/seed_44/validation_predictions.csv `
  --test-predictions results/xgb_calibration/real_v3_1_fixed/seed_44/test_predictions.csv `
  --model-name xgb_full `
  --dataset-name real_v3_1_fixed `
  --seed 44

python scripts/63_run_rigorous_calibration.py `
  --validation-predictions results/rf_calibration/real_v3_1_fixed/seed_42/validation_predictions.csv `
  --test-predictions results/rf_calibration/real_v3_1_fixed/seed_42/test_predictions.csv `
  --model-name rf_no_counter `
  --dataset-name real_v3_1_fixed `
  --seed 42

python scripts/63_run_rigorous_calibration.py `
  --validation-predictions results/rf_calibration/real_v3_1_fixed/seed_43/validation_predictions.csv `
  --test-predictions results/rf_calibration/real_v3_1_fixed/seed_43/test_predictions.csv `
  --model-name rf_no_counter `
  --dataset-name real_v3_1_fixed `
  --seed 43

python scripts/63_run_rigorous_calibration.py `
  --validation-predictions results/rf_calibration/real_v3_1_fixed/seed_44/validation_predictions.csv `
  --test-predictions results/rf_calibration/real_v3_1_fixed/seed_44/test_predictions.csv `
  --model-name rf_no_counter `
  --dataset-name real_v3_1_fixed `
  --seed 44
