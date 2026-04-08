
This package adds the missing block-11 candidate runners that were absent from your local copy of the remake.

Included files:
- src/sc2proj/checkpointing.py
- scripts/37_run_block11_candidate_rf.py
- scripts/38_run_block11_candidate_xgb.py
- scripts/39_make_block11_report.py
- examples/closing_tests_commands.ps1

How to install into your existing repo:
1. Copy `src/sc2proj/checkpointing.py` into `src/sc2proj/`
2. Copy `scripts/37_run_block11_candidate_rf.py`, `38_run_block11_candidate_xgb.py`, `39_make_block11_report.py` into `scripts/`
3. Optionally copy `examples/closing_tests_commands.ps1` into `examples/`

Notes:
- These scripts assume your existing repo already contains:
  - src/sc2proj/utils.py
  - src/sc2proj/training_io.py
  - src/sc2proj/metrics.py
  - src/sc2proj/feature_registry.py
- Added `--profile` support:
  - full
  - no_counter
  - no_counter_no_losses
- Outputs are written under:
  - results/rf/<dataset_name>/seed_<seed>/
  - results/xgb/<dataset_name>/seed_<seed>/
