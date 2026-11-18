# S10 — Probability quality and reliability diagnostics for S9

## Why perform this experiment?

The S9 matched Prefix vs Last9 experiment was completed on FAIR for seeds 42/43/44. The frozen results were independently recomputed and confirmed with `S9_MULTISEED_VALIDATION_PASS`; `s8_cohort_crosscheck=exact_key_and_label_match_PASS`. The complete test suite reported 95 tests, 1 skipped; B0 pointer and publication table checks passed.

On the same retrospective replay cohorts and at 25%, 50%, 75%, 90%, the mean Prefix-minus-Last9 differences were:

| Stage | Accuracy delta (pp) | ROC-AUC delta | Log-loss delta | Brier delta |
|---|---:|---:|---:|---:|
| 25% | -0.259 | -0.002134 | -0.001684 | -0.000750 |
| 50% | +0.544 | +0.003653 | +0.000262 | -0.000278 |
| 75% | +0.531 | +0.008026 | -0.006003 | -0.002456 |
| 90% | +0.197 | +0.014539 | -0.013084 | -0.005470 |

Log loss consistently improves on 75%- and 90%-stage predictions (all 3 seeds); accuracy deltas change sign. **This does not, by itself, prove calibration.** S10 is purely diagnostic: no recalibration, refitting, checkpoint selection, test access or new trained model.

## Metrics and reliability plots

For each seed, stage, and model, compute:

- mean predicted P1-win probability, observed P1-win frequency, and their signed difference (calibration-in-the-large bias)
- empirical reliability in **10 equal-width probability bins**, bin counts and empirical win rates, with exact handling of probabilities 0 and 1
- **Expected Calibration Error (ECE)** = sum over bins (bin count / total count) × absolute difference between mean probability and observed rate. ECE is bin-dependent and can be biased, especially for sparse bins; do not equate smaller ECE with proof of proper calibration
- probability variance, Brier score, log loss, Brier skill relative to same checkpoint replay's empirical event prevalence
- rechecked accuracy, balanced accuracy and ROC-AUC from raw probabilities
- **paired whole-replay bootstrap** 95% CI for Prefix-minus-Last9 Brier difference at each seed/stage; each stage has exactly one prediction per replay

These are distinct measures. A reduction in Brier/log loss can reflect improved discrimination or score sharpness, not necessarily a reduction in empirical calibration error. The reliability diagrams help interpret changes directly.

S10 first verifies the existing S9 aggregate report and all three raw S9 prediction files; matches both models on `(replay_id,time_sec,progress_percent,y_true)`; matches each cohort against S8; recomputes S9's saved accuracy/ROC-AUC/log loss/Brier for each checkpoint; refuses to emit success if anything differs. It never loads the dataset ZIP or the test partition.

## FAIR verification

```bash
cd ~/HPC-POMDP/ML-SC2/ML_SC2
source "$HOME/.venvs/ml-sc2/bin/activate"
export PATH="$HOME/HPC-POMDP/.tools/git-lfs/bin:$PATH"

git pull --ff-only
git rev-parse --short HEAD
git status --short --branch
python -m unittest discover -s tests -p 'test_*.py' -q
python scripts/70_verify_freeze.py --mode pointer
python scripts/58_verify_final_table.py --check
```

Expected: 95 previous tests + six S10 tests = **101 tests**, with 1 historic skip. These are expected outcomes, not yet execution-verified on FAIR.

## Execute the read-only analysis

```bash
python scripts/87_diagnose_s9_probability_quality.py \
  --bootstrap-reps 1000 \
  --bootstrap-seed 2026 \
  --bins 10 \
  --output-dir results/sequential/s10_s9_probability_quality_v1
```

Expected status **`S10_VALIDATION_PROBABILITY_AUDIT_PASS`**, conditional on successfully rereading the 3 S9 and 1 S8 frozen validation outputs.

New `results/sequential/s10_s9_probability_quality_v1` files:

- `summary.json`: protocol, source identities, mean stage diagnostics, limitations
- `probability_diagnostics_by_seed.csv`: 3 seeds × 4 stages × 2 models = 24 rows
- `reliability_bins_by_seed.csv`: 240 rows for 10 bins, with bin counts and observed frequencies
- `paired_brier_deltas_by_seed.csv`: 12 paired Brier deltas with conditional replay bootstrap confidence intervals
- `unweighted_seed_mean_diagnostics.csv`: 4 rows showing stage-specific means and gaps
- `reliability_seed{42,43,44}_p{25,50,75,90}.png`: 12 reliability diagrams, one seed/stage each, showing both models against ideal reliability.

This creates only a **new** output directory and will refuse to overwrite prior results. No recalibrated probabilities or test metrics will be produced.

## Interpretive guardrails

The ECE statistic is especially sensitive to bin count, occupancy and sample size, so report `bins=10` explicitly, show bin counts, and use the plots. At most this experiment finds evidence of better probability-quality measurements on the selected retrospective validation cohort. Calibration should be investigated on a dedicated held-out calibration set in any future prospective work; do not use the already reused validation data as an untouched calibration/test set.

The replay cohorts are complete-case selected based on 9-step eligibility at all checkpoints and exclude roughly 45% of validation games. Separate seed splits share the same original replay corpus. Neither the three-seed average nor the stage-level bootstrap intervals should be treated as three independent external experiments or adjusted multi-comparison significance.

Next decision after S10: decide whether the paper emphasizes stage-dependent discrimination/score quality or whether a separate, frozen ablation study is justified. Do not automatically add more feature or network variations.
