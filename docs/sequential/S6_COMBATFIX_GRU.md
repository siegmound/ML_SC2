# S6 — Dataset variant experiment: GRU v3.1 vs v3.2 combatfix

**Branch:** `feature/sc2-sequential-models`. Follows S5 `4f7a29d`, validated on FAIR (61 tests, one skipped; `S5_MULTISEED_VALIDATION_PASS`).

## Scientific motivation

The full tabular benchmark did not show a clear v3.2 advantage, but additional combat features and corrected combat measurements may become useful to a recurrent model. S6 tests the *dataset-variant effect* while keeping the previously validated 9-snapshot GRU architecture/training config and replay-disjoint seeds 42/43/44 fixed.

The two frozen ZIPs have been inspected directly:
- v3.1: 892,047 snapshots / 23,241 replay IDs / 64 predictors / SHA-256 `65e402884f43c03bf5450d6975e91db3d61ee7e28edf1a56483a37f15d944517`.
- v3.2: 892,047 snapshots / 23,241 replay IDs / 73 predictors / SHA-256 `87ce36435e4a0fc87b8d9421291d3b1fb04d67b5187ede6a2a3b25d0dce1942e`.
- The `(replay_id, time_sec, p1_wins)` columns are identical row-by-row and in the same order. Both have no duplicate endpoint keys.
- The B3 manifest lists nine added predictors in v3.2. Also, **15 of the 64 shared columns had differing numeric values in a separately inspected 12,000-row sample** (chiefly combat and counters). Thus the treatment changes both extra predictors and corrected values. **Do not attribute a potential gain solely to nine new features.** A later `v3.2-common64` ablation can separate these effects.
- Architecture is fixed but input-layer parameter count rises because feature count changes from 64 to 73; the comparison estimates the effect of this combined intervention, not an architecture-parameter-matched causal effect.

## S6 gates

S1 index builder now accepts `--dataset-variant v3_2_combatfix`. The v3.1 default is unchanged. A new v3.2 S1 index must have distinct directory `results/sequential/v3_2_combatfix_w9_s1`. Original S1, historical results and frozen data are never overwritten.

The S6 fit program:
- loads the original v3.1 and new v3.2 S1 indexes;
- checks **full** row and window parity, including source row offsets, labels and time keys, with pinned ZIP hashes, exact feature counts and 9-snapshot/15s/stride-1 settings;
- uses exactly the historical replay group assignments from the v3.1 split JSON (not a new randomly generated split);
- fits the unchanged S4 `GRUConfig()`: one-layer hidden-64 GRU, batch 512, lr 0.001, AdamW, weight decay 1e-4, dropout 0.1, early stopping on validation log loss with patience 4 and max 20 epochs;
- fits the StandardScaler solely on *train* rows, validates on the complete validation windows, **never** scores test;
- saves independent `best_model.pt`, `validation_predictions.csv`, `summary.json` to a fresh results/sequential path.

The validation-only comparison CLI uses strict S5 GRU-prediction verification and a paired replay-level bootstrap (95% conditional exploratory intervals for accuracy and log loss) comparing v3.2 vs original v3.1. For multiple seeds the per-seed differences are shown, but seed splits are not independent samples.

## FAIR commands — first gate (no training)

```bash
cd ~/HPC-POMDP/ML-SC2/ML_SC2
source "$HOME/.venvs/ml-sc2/bin/activate"
git pull --ff-only
git rev-parse --short HEAD
git status --short --branch
python -m unittest discover -s tests -p 'test_*.py' -q
python scripts/70_verify_freeze.py --mode pointer
python scripts/58_verify_final_table.py --check

# The correct real combatfix ZIP must exist; a Git LFS pointer is not enough.
sha256sum data/processed/starcraft_full_dataset_v3_2_combatfix.zip

python scripts/74_build_sequence_index.py \
  --dataset-variant v3_2_combatfix \
  --window 9 \
  --output-dir results/sequential/v3_2_combatfix_w9_s1
```

Before proceeding, confirm actual SHA-256 equals the pinned v3.2 value above. If the ZIP is a Git LFS pointer, resolve that environment issue instead of disabling integrity checks. The S1 builder refuses a pre-existing output directory.

## FAIR commands — second gate (smoke only)

```bash
python scripts/79_train_gru_combatfix.py --smoke --device auto
```

Expected: `S6_DATASET_PARITY_PASS` and `S6_SMOKE_PASS`; test is not iterated. Smoke uses 2 batches each train/validation for up to 2 epochs and saves no outputs. If CUDA is expected, check that reported `device` is `cuda`.

## FAIR commands — third gate (seed 42)

Only after both prior gates pass:

```bash
python scripts/79_train_gru_combatfix.py \
  --split-json data/processed/splits/real_v3_1_fixed_split_seed_42.json \
  --device auto \
  --output-dir results/sequential/s6_combatfix_gru_w9_seed42_v1

python scripts/80_compare_gru_variants.py --seeds 42 --bootstrap-reps 1000
```

Expected `S6_VALIDATION_PASS`, 112,312 seed-42 validation endpoints and `S6_VARIANT_COMPARISON_PASS`. No standalone claim is justified from one seed. Review the paired confidence interval before continuing; neither train nor test the other variants using test metrics.

## Remaining seeds — only after successful seed-42 audit

```bash
python scripts/79_train_gru_combatfix.py \
  --split-json data/processed/splits/real_v3_1_fixed_split_seed_43.json \
  --device auto \
  --output-dir results/sequential/s6_combatfix_gru_w9_seed43_v1

python scripts/79_train_gru_combatfix.py \
  --split-json data/processed/splits/real_v3_1_fixed_split_seed_44.json \
  --device auto \
  --output-dir results/sequential/s6_combatfix_gru_w9_seed44_v1

python scripts/80_compare_gru_variants.py \
  --bootstrap-reps 1000 \
  --bootstrap-seed 2026 \
  --output-dir results/sequential/s6_combatfix_vs_v31_w9_v1
```

Expected val endpoint counts: 42=112,312; 43=112,943; 44=111,899. Written report contains `summary.json` and `per_seed_comparison.csv` only after successful validation of all selected seeds.

## Decision gate and limits

If v3.2 outperforms v3.1 on multiple seeds, analyze a third arm **v3.2 corrected shared 64 features without its nine added predictors** to isolate whether improvements come from corrected source values or additional signals. If not, retain v3.1 as the first temporal reference and investigate overfitting separately. The early validation loss worsening in S4 (selected epochs 1–2) remains important.

This is validation-based **exploratory research**, not an untouched test. The S5 seed-44 replay bootstrap intervals often included zero. Avoid inflated claims of statistical significance, avoid comparing full test scores directly, and never silently drop mismatched endpoints.
