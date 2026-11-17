# Known limitations

- GPU-backed XGBoost and PyTorch behavior may vary across environments, especially across CUDA, driver, and library combinations.
- Parser parity with older branches is not claimed automatically; it should only be claimed after representative replay-audit validation.
- The upstream replay source is now documented, but exact reconstruction of the locally used corpus still depends on replaypack-level inventories, parser-side exclusions, and project filtering decisions.
- Final repository polish is not fully captured by committed files alone: GitHub About metadata, topics, and the final chosen license still need to be set on the hosting platform if not already completed.

## B2 methodological audit (2026-11-17)

- Historical full-data RF fits use **train+validation**, but historical final XGB fits **train only** with early stopping on validation. The original four-row benchmark is descriptive, not a controlled equal-fit-budget ablation.
- A separate retrospective **train-only RF/XGB** comparison now checks 3 matched seeds and identical test replay/snapshot labels: `results/b2_fit_budget/retrospective_train_only.csv`. This is not a new untouched holdout.
- The historical deep candidate runner computed per-candidate test predictions. The B2 version of `scripts/64_run_deep_finalist.py` gates test until validation-only selection. The old saved metrics are unchanged and cannot be reclassified as pristine after the code fix.
- XGB's historical calibration reuses the same validation partition as XGB early stopping; calibration and tuning were not independent. B2 adds a four-way grouped runner, `scripts/67_run_strict_calibration.py`, for future re-evaluation.
- The test replay subsets were inspected during earlier project development, so subsequent experiments must not claim a never-seen blind test, even when the new code evaluates selected models only.
- No B2 full-scale GPU/deep retraining has yet been performed; B2 smoke tests alone do not establish full-scale equivalence or new model rankings.
