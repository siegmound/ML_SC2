# S7 — Combatfix corrected-common64 GRU ablation

**Research question:** Why does the S6 GRU on the full v3.2 combatfix 73-feature dataset improve on seed 44, worsen on seed 43, and provide negligible mean change over 42/43/44?

S6 FAIR checks have passed (both dataset ZIPs correctly present with frozen hashes, exact 892,047-row / 706,952-window S1 parity, full 73-feature GRU validation and retrospective paired bootstrap, seeds 42/43/44). No independent prospective test set was used.

## Predeclared three-arm ablation (same 9-snapshot inputs, same replay groups and fit hyperparameters)

- **A:** v3.1 original64 — previously trained S4 at `results/sequential/s4_gru_w9_seed{seed}_v1`.
- **B:** v3.2 corrected-common64 — new S7 GRU, using only the 64 columns matching A's predictor names, **in exactly A's feature order**, but reading the corrected numeric values from the **v3.2** ZIP. Training result in `results/sequential/s7_combatfix_common64_gru_w9_seed{seed}_v1`.
- **C:** v3.2 full73 — previously trained S6 at `results/sequential/s6_combatfix_gru_w9_seed{seed}_v1`.

Three contrasts are computed on **exactly the same validation endpoints for each seed**:

1. **B − A:** effect of corrected values of shared 64 predictors, holding feature names and GRU input width at 64.
2. **C − B:** effect of including nine extra v3.2 predictors, while holding all shared v3.2 values fixed. Input GRU parameter count changes, so this includes capacity/optimization consequences.
3. **C − A:** total dataset-variant difference already computed in S6; serves as consistency/reference.

The 64-column view is **in memory only**: it does not rewrite v3.1 or combatfix indexes, ZIPs, or checkpoints. The S7 saved summary explicitly identifies the combatfix SHA `87ce36435e4a0fc87b8d9421291d3b1fb04d67b5187ede6a2a3b25d0dce1942e`, the view name `v3_2_common64`, and the exact 64 predictor names from the v3.1 manifest. No new split creation and no test access.

The model uses unchanged S4/S6 `GRUConfig()`: hidden 64, one recurrent layer, dropout 0.1, AdamW lr 0.001, batch 512, weight decay 0.0001, max 20 epochs, early stopping patience 4 on validation BCE. StandardScaler is fit only on distinct train-replay snapshots.

Paired intervals are generated using **whole-replay bootstrap** (1000 draws/contrast/seed) on validation accuracy and log loss, plus descriptive point estimates for balanced accuracy, ROC-AUC and Brier score. Three seeds are dependent splits of the same corpus; these conditional bootstrap intervals are not prospective significance guarantees. Existing validation and historical test had been inspected in previous stages.

## FAIR verification gate

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

Expected: original 72 tests plus four new S7 unit tests; one historical skipped. No protected artifact changes.

## S7 smoke gate (no files written)

```bash
python scripts/81_train_gru_combatfix_common64.py --smoke --device auto
```

Expected: `S6_DATASET_PARITY_PASS` then `S7_SMOKE_PASS`, with `feature_count=64`, `source_zip_sha256` equal to the v3.2 combatfix hash, and `device=cuda`.

## First full run and strict ablation check — seed 42 only

```bash
python scripts/81_train_gru_combatfix_common64.py \
  --split-json data/processed/splits/real_v3_1_fixed_split_seed_42.json \
  --device auto \
  --output-dir results/sequential/s7_combatfix_common64_gru_w9_seed42_v1

python scripts/82_compare_gru_ablation.py --seeds 42 --bootstrap-reps 1000
```

Expected: `S7_VALIDATION_PASS` and 112,312 validation endpoints, then `S7_VALIDATION_ABLATION_PASS`. Review failure output before proceeding. A single seed does not settle the variant question.

## Remaining seeds — after seed-42 check

```bash
python scripts/81_train_gru_combatfix_common64.py \
  --split-json data/processed/splits/real_v3_1_fixed_split_seed_43.json \
  --device auto \
  --output-dir results/sequential/s7_combatfix_common64_gru_w9_seed43_v1

python scripts/81_train_gru_combatfix_common64.py \
  --split-json data/processed/splits/real_v3_1_fixed_split_seed_44.json \
  --device auto \
  --output-dir results/sequential/s7_combatfix_common64_gru_w9_seed44_v1

python scripts/82_compare_gru_ablation.py \
  --bootstrap-reps 1000 \
  --bootstrap-seed 2026 \
  --output-dir results/sequential/s7_combatfix_common64_ablation_v1
```

Expected validation endpoints for seed42/43/44: 112,312 / 112,943 / 111,899 respectively. Outputs: `summary.json` with all per-seed contrasts, conditional paired intervals and unweighted seed means; `paired_ablation_deltas.csv` with nine contrasts.

**Decision gate:** inspect A/B/C contrasts and replay intervals before attempting regularization, longer temporal windows, alternate recurrent architectures, or any new test evaluation. Do not infer causality or significance from the unweighted 3-seed mean alone.
