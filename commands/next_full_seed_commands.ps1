$env:PYTHONPATH = "src"

# Inspect the canonical XGB trainer CLI first
python scripts/07_train_xgb.py -h

# Suggested next step: full single-seed XGB on v3_2_combatfix
# If your local 07_train_xgb.py supports split-json and row budgets, adapt accordingly.
# Otherwise run the canonical trainer on the full dataset version and save the resulting metrics_summary.json.
