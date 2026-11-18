# S11 — Do sequential models beat existing tabular methods at equal game progress?

## FAIR results — 2026-11-18

The user executed S11 on FAIR and obtained **S11_MATCHED_VALIDATION_MODELS_PASS** with exact validation endpoint matching for all seven arms and seeds. The saved S11 report is `results/sequential/s11_tabular_vs_gru_progress_v1/`. This evaluation is retrospective validation, not a new unbiased test.

**Validation accuracy, unweighted means across overlapping seed splits:**

| Model | 25% | 50% | 75% | 90% |
|---|---:|---:|---:|---:|
| RF archived | 0.562578 | 0.621947 | 0.658753 | 0.688629 |
| XGB archived | 0.566406 | 0.619423 | 0.659070 | 0.688562 |
| MLP archived | 0.567338 | 0.620457 | 0.660711 | 0.686866 |
| GRU S8 last9, trained on all windows | **0.577096** | **0.622655** | 0.672514 | 0.694307 |
| GRU S9 last9, trained on 4 checkpoints | 0.565784 | 0.609598 | 0.667929 | 0.694239 |
| GRU S9 prefix, trained on 4 checkpoints | 0.563198 | 0.615039 | **0.673241** | **0.696206** |
| Constant P1 train replay win prior | 0.513407 | 0.513407 | 0.513407 | 0.513407 |

All three seeds passed checkpoint matching: seed42 2012 replays, seed43 2076, seed44 2040. Best sequential model minus best archived tabular at each stage is approximately **+0.98, +0.07, +1.25, +0.76 percentage points**, respectively. Model-by-stage selection of the maximum is descriptive and post hoc; it is *not* a deployable selected policy. Report conditional paired accuracy CIs from `paired_deltas_vs_s8_gru.csv` before any significance claims. Do not state that one architecture has an architecture-only causal advantage: the training sets, budgets, historical feature profiles and stopping criteria are different.

**Test-suite issue and fix:** the full test suite ran 106 tests with 1 historic skip and exactly 1 failing S11 unit-test assertion. The software's fail-closed cohort check correctly rejected incomplete input, but the test expected the word `checkpoint` while the exception says `S8 cohort must contain exactly the same replay IDs at every stage`. The test expectation was corrected in a separate GitHub commit, changing no production implementation, predictions, models or frozen artifacts. A new FAIR test-suite run is still required to confirm 106 tests OK, 1 skipped. The frozen B0 pointer audit and publication table check both passed.

## Motivation

S8–S10 established retrospective progress curves for fixed-nine-window and observed-prefix GRUs. Accuracy increases with game progress for both, but S9-prefix accuracy gains over S9-last9 were small and inconsistent. At 90% S9-prefix improved average Brier, log loss and ROC-AUC, but 10-bin ECE was not always lower.

**Open empirical question:** Does the recurrent design itself improve winner classification over the historical RF, XGB and MLP models when evaluating on **exactly the same replay IDs and the same four snapshot-fraction endpoints**?

A comparison of the historical whole-validation accuracy (over ~112K nine-window endpoints per seed) with the S9 90%-of-game accuracy (over ~2K replay endpoints per seed) is invalid: the evaluation populations differ. Equally, published results on different StarCraft corpora/timestamps cannot set a directly comparable performance threshold.

## The seven frozen reference arms

| Label | Source | Training protocol |
|---|---|---|
| train_prior | P1 win prevalence fit on train replay groups | No trained model, one constant predicted probability |
| rf_archived | `results/rf_calibration/real_v3_1_fixed/seed_*/validation_predictions.csv` | Historical RF; stored fitting/decision protocol |
| xgb_archived | `results/xgb_calibration/real_v3_1_fixed/seed_*/validation_predictions.csv` | Historical XGB; early stopping/validation may differ |
| mlp_archived | `results/deep_final/real_v3_1_fixed/seed_*/validation_predictions.csv` | Historical deep MLP |
| s8_gru_w9_allwindows | `results/sequential/s4_gru_w9_seed*_v1/validation_predictions.csv` | Nine-snapshot GRU trained on all valid 9-step windows |
| s9_gru_last9_fourpoints | `results/sequential/s9_prefix_vs_last9_seed*_v1/last9_validation_predictions.csv` | New GRU trained on only four checkpoint examples per eligible replay |
| s9_gru_prefix_fourpoints | `results/sequential/s9_prefix_vs_last9_seed*_v1/prefix_validation_predictions.csv` | New variable-prefix GRU trained on same four checkpoint examples |

All model predictions are preexisting files. **No new training, hyperparameter tuning, calibration, threshold fitting, model selection or test evaluation.** The frozen S8 validation checkpoint table determines the evaluation cohort for every arm: seed42 2,012 replays; seed43 2,076; seed44 2,040. Every replay contributes one observation at each of 25/50/75/90%. The training-derived constant baseline uses one label per distinct train replay rather than the highly correlated training snapshot rows.

The existing S3 `compare_archived_baselines` function audits that RF/XGB/MLP have exact coverage and valid labels over the full S1 validation endpoints (not only the progress cohort). The S4 `load_verified_gru` checks summary/source/checkpoint provenance and full validation prediction identity. The S9 previously audited report and the saved S8 prediction CSV are checked for exact identities; no future snapshot enters any new model input.

For each seed/stage/arm compute accuracy, balanced accuracy, ROC-AUC, log loss and Brier. Report means across seeds as *descriptive*, not independent runs. Whole-replay paired bootstrap (1000 draws by default) compares each candidate against the historical v3.1 nine-window GRU at each checkpoint, within the same seed. These intervals are **exploratory and uncorrected for multiple comparisons**.

## Major caveats

- **Same evaluation examples is not the same training budget.** S4 GRU trained on ~453K overlapping 9-window examples for seed42; S9 trained on only 33,132 four-checkpoint examples and excluded many replays. RF/XGB/MLP use historical snapshot-level fitting and may employ different feature profiles and threshold rules. Thus, the comparison identifies promising methods but cannot establish architecture-only causal superiority.
- The ECE/calibration and accuracy analyses reuse validation selected for checkpoints; the historical test was inspected earlier. These are retrospective validation scores, not an independent new holdout.
- Percent-of-game endpoints require knowing final replay length to select offline checkpoints; models do not receive that information.
- S8 complete-case selection excludes ~45% of validation replays with insufficient early nine-step data; findings do not immediately generalize to all games.
- The preserved v3.1 dataset is frozen and integrity-verified, but the clean-room replay parser has not recovered the full 64/73-feature historical lineage. This affects independently regenerating the complete dataset, not this read-only prediction comparison.

## FAIR — run the audit (CPU only)

```bash
cd ~/HPC-POMDP/ML-SC2/ML_SC2
source "$HOME/.venvs/ml-sc2/bin/activate"
git pull --ff-only

python -m unittest discover -s tests -p 'test_*.py' -q
python scripts/70_verify_freeze.py --mode pointer
python scripts/58_verify_final_table.py --check

python scripts/88_compare_tabular_vs_gru_at_checkpoints.py \
  --bootstrap-reps 1000 \
  --bootstrap-seed 2026 \
  --output-dir results/sequential/s11_tabular_vs_gru_progress_v1
```

Expected (not yet verified on FAIR): 101 previous tests + 5 new tests = 106 tests; a historical skip may remain. Output status **`S11_MATCHED_VALIDATION_MODELS_PASS`**. A fresh output folder is required, no overwrites allowed.

Output files:

- `summary.json`: provenance, per-seed validation, equal replay/time checkpoints and model caveats
- `metrics_by_seed_stage.csv`: 3 seeds × 4 checkpoints × 7 arms = 84 performance rows
- `paired_deltas_vs_s8_gru.csv`: paired model-score differences and conditional whole-replay intervals
- `unweighted_mean_metrics.csv`: stagewise comparison, model-by-model
- `unweighted_mean_deltas.csv`: mean stagewise differences versus the original nine-window GRU

## Decision after S11

If frozen XGB/RF/MLP beat the sequential models on the same replay checkpoints, prioritize a **controlled tabular baseline that incorporates causal history summaries** (last value, lagged values, EMA, recent trend/slope, min/max over history, combat/economy changes), or train XGB/CatBoost/LightGBM on snapshot + rolling-window statistics. If GRUs win, study the **data-efficient training strategy** before changing architecture: more prefix cutoff points per train replay or a random no-future observed cutoff policy, with stage-balanced/replay-balanced sampling. Both require fresh training on the same frozen split and evaluation cohort, with all improvements framed as validation hypotheses.

Do not assume that an LSTM, Transformer, TCN, 3D-ResNet or larger GRU will outperform a simple GBDT. Spatial networks need new spatial data, beyond the currently frozen snapshot feature CSV. A useful follow-up after S11 could audit the clock-time distribution and replay-length-dependent selection bias (same S8 cohort vs all eligible stage-by-stage replays) without another training cycle.
