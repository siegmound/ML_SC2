# S5 — Paired, multi-seed validation comparison: GRU versus frozen baselines

**Branch:** `feature/sc2-sequential-models`. Depends on S3 (`a08472e`) and S4 (`0c683d0`). This stage does **not** train or recalibrate a model, touch the test partition, rewrite S4 outputs, or create a new held-out claim.

## Inputs

- S1: `results/sequential/v3_1_fixed_w9_s1/` (full v3.1 index, 9 observations, 64 predictors).
- Historical replay-disjoint split manifests: `data/processed/splits/real_v3_1_fixed_split_seed_{42,43,44}.json`.
- S3: original validation prediction CSVs in `results/rf_calibration/`, `results/xgb_calibration/`, and `results/deep_final/` (under `real_v3_1_fixed/seed_{seed}/`). No trained model is rerun.
- S4: `results/sequential/s4_gru_w9_seed{seed}_v1/summary.json` and `validation_predictions.csv` for each seed (these were generated on FAIR).

## Strict validity gates

1. For each seed, rebuild the *validation-only* S3 baseline alignment using the exact S1 temporal endpoints.
2. Require **exactly** the S4 GRU keys `(replay_id, time_sec)`—no missing, duplicate, or unexpected rows—and identical winner labels.
3. Reject null, out-of-range, inconsistent or malformed probabilities and binary predictions. Verify GRU summary status, seed, 9-window schema, feature count, dataset hash, validation-only checkpoint policy, and stored metric/row counts.
4. Check that the GRU hyperparameters are identical across all three seeds.
5. Require validation endpoint counts 112312/112943/111899. Never silently align only an intersection.

## Methods and interpretation

Output contains accuracy, balanced accuracy, ROC-AUC, log loss, and Brier score per model/seed, plus GRU minus baseline deltas and an **unweighted arithmetic mean** of each seed's metrics. We do not pool the three seeds' snapshots as independent observations; the same source replay corpus is reused by different splits.

For each seed and baseline, 95% percentile **paired replay-level bootstrap** intervals are estimated for the difference in snapshot-weighted **accuracy** and **log loss**. Each draw resamples complete validation replay groups with replacement; all prediction endpoints within a drawn replay are kept together and both models use the same resample. This avoids row-wise independence assumptions while preserving the original snapshot-weighted estimand (longer replays contribute more windows). Use the same number of bootstrap replicates and deterministic RNG seed across FAIR reruns. **AUC and balanced accuracy are point estimates only.**

These intervals are *conditional/exploratory*, not independent external confirmation, because (a) validation was previously used for selection/early stopping, including for the GRU, (b) historical models had different optimization/selection processes, and (c) original historical test outputs were previously inspected. Do not present nominal CI or multi-seed means as an unbiased, prospectively held-out estimate. No test predictions, labels, scores, or metrics are read by S5.

## FAIR run — verify and report

```bash
cd ~/HPC-POMDP/ML-SC2/ML_SC2
source "$HOME/.venvs/ml-sc2/bin/activate"
git pull --ff-only
git rev-parse --short HEAD
git status --short --branch
python -m unittest discover -s tests -p 'test_*.py' -q
python scripts/70_verify_freeze.py --mode pointer
python scripts/58_verify_final_table.py --check

python scripts/78_compare_gru_multiseed.py \
  --bootstrap-reps 1000 \
  --bootstrap-seed 2026 \
  --output-dir results/sequential/s5_gru_vs_baselines_w9_v1
```

Expected first gates: **61 tests total, one historical skipped**, artifact integrity PASS. New S5 tests are nine. Expected comparison status: `S5_MULTISEED_VALIDATION_PASS`; counts 112312/112943/111899. Data reading is CPU only; no GPU required. Run only once with this output directory; the CLI refuses overwrite.

Artifacts under the new results/sequential output:

- `summary.json` — multi-seed model metrics, explicit descriptive deltas, bootstrap CIs, source SHA and caveats;
- `per_seed_metrics.csv` — 12 rows (4 models × 3 seeds);
- `paired_deltas.csv` — 9 rows (GRU versus each of RF, XGB, MLP on each of 3 seeds).

All historical B0, S1, S2, S3 and S4 outputs remain untouched. The `results/sequential/` subtree is ignored by Git. Check the output and the checkpoint before any new architecture or window-length sweep; do not silently run test or optimize based on earlier test scores.
