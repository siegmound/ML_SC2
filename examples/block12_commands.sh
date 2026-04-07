python scripts/40_import_block11_seed_zips.py --zip-paths /mnt/data/rfseed_42.zip /mnt/data/rfseed_43.zip /mnt/data/rfseed_44.zip /mnt/data/xgbseed_42.zip /mnt/data/xgbseed_43.zip /mnt/data/xgbseed_44.zip
python scripts/42_run_block12_feature_stability.py
python scripts/41_run_block12_rf_family_ablation.py --dataset-zip data/processed/real_v3_1_fixed_smoke3000.zip --split-json data/processed/splits/real_v3_1_fixed_smoke3000_split_seed_42.json --dataset-name real_v3_1_fixed_smoke3000
python scripts/43_make_block12_report.py
