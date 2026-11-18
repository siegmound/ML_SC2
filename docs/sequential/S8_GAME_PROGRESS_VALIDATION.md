# S8 — Accuracy as a function of observed game fraction (frozen 9-step GRU)

## Research question

At what observed fraction of a StarCraft II replay can an already trained winner predictor identify the eventual winner reliably? Evaluate predictions after **25%, 50%, 75%, and 90% of the recorded snapshots**, and compare the same replay cohort between checkpoints and model arms.

**Critical distinction:** the fraction and final replay length are used ONLY by this *offline evaluator*, never sent to the GRU. Each S4/S6/S7 GRU originally received a fixed 9-snapshot sequence ending at the selected checkpoint. The model does not know how long the game will last. These retrospective percent-based checkpoints are not directly deployable online without duration knowledge.

This **first S8 experiment does not retrain** any network. Later S9 can fit a GRU on varying-length prefixes from replay start and compare its predictions on the exact same checkpoints and validation replay cohort.

## Frozen inputs / experiment arms

- A = `v31_original64`: S4 9-step GRU with 64 v3.1 predictors.
- B = `v32_common64`: S7 9-step GRU with corrected values of the 64 shared predictors in v3.2.
- C = `v32_full73`: S6 9-step GRU with all 73 combatfix predictors.
- Replay-disjoint historical splits: seeds 42, 43, 44; validation only. Original full S1 index (892,047 rows, 706,952 windows), combatfix index checked for exact row/window and source-hash parity before joining variants.

## Exact checkpoint selection: no future observations

For a replay with `N` recorded snapshots, define checkpoint `p` by:

```text
observed_snapshot_count = ceil(p * N / 100)
checkpoint = snapshot at observed_snapshot_count (1-based)
```

For example, if `N=80`, checkpoints are after 20, 40, 60, and 72 snapshots.

For every validation replay, choose the **exact** snapshot timestamp for each fraction. It must be a valid S1 9-snapshot window endpoint; no inference at a later timestamp, no silent nearest-available substitution. If any of the four checkpoints is ineligible (e.g. too few early snapshots or a gap prevents a complete 9-window), exclude that replay from the **primary balanced cohort**. Report this exclusion and coverage explicitly; these results generalize to eligible, relatively long/regular games, not automatically all replays.

For each retained replay there is exactly **one prediction per checkpoint per arm**, and all four checkpoints use the **same replay IDs and true labels**. Compare GRU CSV predictions only after strict full-validation endpoint/label/schema matching using S5/S6/S7 validators. No test scores, model training or calibration.

Per seed and arm, report accuracy, balanced accuracy, ROC-AUC, log loss, Brier score at each checkpoint. For each arm, calculate paired 90%-minus-25% deltas with whole-replay percentile bootstrap (1000 draws by default), including accuracy and log-loss intervals. Three seed datasets share an original corpus and are **not independent experimental replications**. Unweighted averages across seeds are descriptive.

The existing S4/S6/S7 checkpoints were selected on validation BCE. This analysis is retrospective/descriptive and is not a fresh external estimate. Also, final length is only an offline selection device: if the user wants an online system, evaluate at predetermined absolute in-game times without relying on unknown final duration.

## Gate 1 — finish S7 ablation (already started)

Complete Common64 seeds 43 and 44 and run:

```bash
python scripts/82_compare_gru_ablation.py \
  --bootstrap-reps 1000 \
  --bootstrap-seed 2026 \
  --output-dir results/sequential/s7_combatfix_common64_ablation_v1
```

Require `S7_VALIDATION_ABLATION_PASS` before using all three arms together.

## Gate 2 — update FAIR and regression

```bash
cd ~/HPC-POMDP/ML-SC2/ML_SC2
source "$HOME/.venvs/ml-sc2/bin/activate"
git pull --ff-only
git rev-parse --short HEAD
git status --short --branch
python -m unittest discover -s tests -p 'test_*.py' -q
python scripts/70_verify_freeze.py --mode pointer
python scripts/58_verify_final_table.py --check
```

Expected: prior 76 tests plus seven new S8 tests (one historical may skip). No existing output overwrites.

## Gate 3 — evaluation of all 3 model arms (no training, CPU only)

```bash
python scripts/83_evaluate_gru_game_progress.py \
  --seeds 42 43 44 \
  --bootstrap-reps 1000 \
  --bootstrap-seed 2026 \
  --output-dir results/sequential/s8_game_progress_w9_all_arms_v1
```

Expected: `S8_VALIDATION_PROGRESS_PASS`. Print number of eligible replay(s) and unweighted mean accuracy curves; the count will be **measured**, not assumed to equal the entire validation partition. New outputs:

- `summary.json`: methods, balanced cohort sizes, all per-seed/arm/checkpoint metrics, paired progression intervals, source identity and limitations
- `cohort_coverage.csv`: per-seed full validation replay count, raw checkpoint eligibility and balanced cohort size
- `checkpoint_metrics.csv`: 3 seeds × 4 checkpoints × 3 model arms = 36 score rows
- `matched_checkpoint_predictions.csv`: keys and aligned validation probabilities for reproducibility; not test predictions

Render three descriptive figure files:

```bash
python scripts/84_plot_gru_game_progress.py \
  --metrics-csv results/sequential/s8_game_progress_w9_all_arms_v1/checkpoint_metrics.csv \
  --output-dir results/sequential/s8_game_progress_w9_all_arms_v1
```

Figures: `accuracy_by_progress.png`, `roc_auc_by_progress.png`, `log_loss_by_progress.png`. They show unweighted averages across seeds, not confidence bands or independent replications.

### Optional early S8 proof-of-concept (without S7 seed43/44)

One can run `--arms v31_original64 --seeds 42 43 44` with a **different fresh output folder**. This reuses only frozen S4 predictions, providing the progress curve even before S7 is fully finished. Do not silently conflate this with the later all-arm report.

## Next experiment after S8

S9: variable-length, replay-prefix GRU trained with a replay-disjoint split, padding/packed sequences, no future features, checkpoint-matched evaluation on **the same eligible replay cohort** as S8. Contrast expanding-prefix GRU against the S8 fixed-9 short-memory model. A fixed-percentage evaluation is offline only and does not reveal remaining duration to either model.
