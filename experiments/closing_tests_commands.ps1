
# PowerShell commands for the remaining closure tests

# 1) XGB challenger on real_v3_2_combatfix_smoke6000
python scripts/38_run_block11_candidate_xgb.py `
  --dataset-zip data/processed/real_v3_2_combatfix_smoke6000.zip `
  --split-json data/processed/splits/real_v3_2_combatfix_smoke6000_split_seed_42.json `
  --dataset-name real_v3_2_combatfix_smoke6000 `
  --profile no_counter `
  --device cpu `
  --max-train-rows 20000 `
  --max-val-rows 8000 `
  --max-test-rows 8000 `
  --resume

python scripts/38_run_block11_candidate_xgb.py `
  --dataset-zip data/processed/real_v3_2_combatfix_smoke6000.zip `
  --split-json data/processed/splits/real_v3_2_combatfix_smoke6000_split_seed_43.json `
  --dataset-name real_v3_2_combatfix_smoke6000 `
  --profile no_counter `
  --device cpu `
  --max-train-rows 20000 `
  --max-val-rows 8000 `
  --max-test-rows 8000 `
  --resume

python scripts/38_run_block11_candidate_xgb.py `
  --dataset-zip data/processed/real_v3_2_combatfix_smoke6000.zip `
  --split-json data/processed/splits/real_v3_2_combatfix_smoke6000_split_seed_44.json `
  --dataset-name real_v3_2_combatfix_smoke6000 `
  --profile no_counter `
  --device cpu `
  --max-train-rows 20000 `
  --max-val-rows 8000 `
  --max-test-rows 8000 `
  --resume

# 2) Optional RF candidate runs with the same profile for exact apples-to-apples artifacts
python scripts/37_run_block11_candidate_rf.py `
  --dataset-zip data/processed/real_v3_2_combatfix_smoke6000.zip `
  --split-json data/processed/splits/real_v3_2_combatfix_smoke6000_split_seed_42.json `
  --dataset-name real_v3_2_combatfix_smoke6000 `
  --profile no_counter `
  --max-train-rows 20000 `
  --max-val-rows 8000 `
  --max-test-rows 8000 `
  --resume

# 3) Aggregate block11-style report
python scripts/39_make_block11_report.py `
  --dataset-name real_v3_2_combatfix_smoke6000
