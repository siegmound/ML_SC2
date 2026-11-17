# Final project status

## Stable final conclusions

- Random Forest and XGBoost are effectively tied as the final tabular candidates.
- Random Forest is marginally stronger as a classifier.
- XGBoost is marginally stronger as a probability-oriented model.
- The best Random Forest profile remains `no_counter`.
- The deep challenger is competitive, but does not surpass Random Forest or XGBoost.

## Calibration conclusion

- Random Forest benefits from post-hoc calibration.
- XGBoost is already well calibrated in uncalibrated form.

Therefore:

- classification-oriented recommendation: **RF + `v3_1_fixed` + `no_counter`**
- probability-oriented recommendation: **XGB + `v3_1_fixed` + full**

## Data provenance status

The upstream replay source is explicitly documented in the repository as **SC2ReSet: StarCraft II Esport Replaypack Set** (Zenodo record `14963356`, DOI `10.5281/zenodo.14963356`, version `2.0.0`).

Provenance documents linked from the final repository state:

- `data/raw/replay_sources.md`
- `data/raw/replaypack_inventory.csv`
- `docs/data_provenance.md`
- `data/processed/manifests/upstream_replay_source_manifest.md`

This closes the main public-facing provenance gap between replay origin, local raw material, and final processed datasets.

## B2 methodology correction status (2026-11-17)

The original RF-vs-XGB recommendation remains a **historical descriptive** result. Its training budget is asymmetric (RF: train+validation; XGB: train with validation early stopping). A separate matched train-only comparison is now reproducible at `results/b2_fit_budget/retrospective_train_only.csv`; it shows similarly small model differences and uses saved test scores that had already been inspected.

The original calibration tables are retrospective three-way validation-to-test results. For XGB, fitting the calibrator on the validation set reused for early stopping means tuning and calibration were not independent. The new four-way B2 runner separates these groups, but has **only passed synthetic smoke tests**; no new full-data calibration outcome is yet certified.

The historical Deep MLP test-peeking behavior has been corrected in the runner. Its original published metrics are preserved and must not be called strict one-shot-test evidence. Detailed audit: `docs/audit/B2_METHODS_2026-11-17.md`.

## B4 engineering reproducibility update (2026-11-17)

The correction branch now carries a fast GitHub Actions verification workflow,
an opt-in full LFS/data verification path, and a Git SHA/LFS checker that
compares 31 historical artifacts against the B0 baseline. The Python
minimum version has been aligned with the 3.11 environment. New RF runs
can save checkpoints, while the selected Deep candidate saves its CPU
state and preprocessing scaler.

**Status:** source-level code tests passed locally (30/30), but the clean
GitHub clone, complete v3.2 LFS pull, remote CI status and exact GPU
environment lock remain unverified. Historical scientific claims are
unchanged. See `docs/audit/B4_REPRODUCIBILITY_2026-11-17.md`.
