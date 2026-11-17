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

## B3 parser and dataset provenance limitations (2026-11-17)

- The B3 parser now rejects unknown/ambiguous winners, non-1v1 games and invalid winner IDs rather than silently encoding unknown matches as `p1_wins=0`.
- Recent unit losses now use the replay-time live-tag inventory, which avoids the initial-base-lookup early-break undercount. Corrected code is regression-tested on synthetic replay events, **not** yet revalidated on all raw replay files.
- The dataset builder rejects duplicate replay basenames originating from distinct replaypack directories and refuses to overwrite either frozen final ZIP by name. A successful new run must use a distinct dataset name.
- Clean-room parsing and feature generation remain **not equivalent** to the historical 64/73-feature datasets. The B3 feature manifest names 35 v3.1 fields missing from the parser and nine new v3.2 fields, but does not invent their original feature-engineering formulas.
- The 816-replay discrepancy (24,057 documented upstream minus 23,241 processed) remains **unattributed** without raw replay IDs, extraction logs and filtering decisions. An arithmetic difference is not evidence that each of these 816 was excluded for a specific parser reason.
- The local v3.2 archive available to this B3 session did not pass ZIP central-directory validation; its actual ZIP content/header was **not** rescanned here. The v3.2 schema is established from GitHub's saved full-XGB config files. A fresh LFS pull and archive check are required in B4.
