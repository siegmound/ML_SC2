$env:PYTHONPATH = "src"

python scripts/53_run_xgb_full_gpu.py `
  --dataset-zip data/processed/starcraft_full_dataset_v3_2_combatfix.zip `
  --split-json data/processed/splits/real_v3_2_combatfix_split_seed_42.json `
  --dataset-name real_v3_2_combatfix_fullgpu_clean `
  --device cuda `
  --profile full
