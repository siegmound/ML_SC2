# S3 — Endpoint-matched historical tabular baselines

**Branch:** `feature/sc2-sequential-models` (S1 and S2 passed on FAIR). This phase adds a **read-only, retrospective** comparison; it does not train any model, recalibrate probabilities, rewrite archived results, or select hyperparameters on test.

## Scientific motivation

The S1 9-snapshot index has 706,952 valid windows. Many historical predictions occur at times before a 9-snapshot sequence exists. Comparing an existing model's full test Accuracy to a temporal model evaluated only on later time points would therefore be confounded by **different prediction endpoints**.

S3 selects the exact S1 `(replay_id, time_sec)` endpoints from the historical replay-aware validation/test partition, then matches the saved original model predictions on those same keys. It rejects missing, duplicated, mislabeled, invalid-probability, or wrong-partition predictions. Archived prediction rows before the first eligible endpoint are allowed and excluded from the matched score.

## Historical model sources

| Name | Stored path pattern | Experimental caveat |
|---|---|---|
| RF | `results/rf_calibration/real_v3_1_fixed/seed_{seed}/{validation,test}_predictions.csv` | RF trained only on train replay groups; profile `no_counter` |
| XGB | `results/xgb_calibration/real_v3_1_fixed/seed_{seed}/{validation,test}_predictions.csv` | XGB fit train-only, but its early stopping used validation |
| Deep MLP | `results/deep_final/real_v3_1_fixed/seed_{seed}/{validation,test}_predictions.csv` | Historical candidate process tested multiple candidates; the B2 correction does not retroactively blind these results |

The `{validation,test}` notation means the actual file is `validation_predictions.csv` or `test_predictions.csv`. These files contain **uncalibrated, original predictions**; no Platt or isotonic transformations are applied by S3.

The original final RF result in `results/rf_full/` used **train + validation**, whereas original final XGB fit **train only**. S3 therefore deliberately uses the preserved **train-only RF** predictions from the calibration preparation runs to make the RF/XGB fitting-row budget more comparable. This does not make the tuning or selection budgets identical.

These are not new independent test observations: the archived test set was already inspected during the original project. Validation was already used in selection/early stopping. Scores from both sets must be presented as **retrospective descriptive comparisons**, not a new unbiased holdout claim.

## FAIR verification

Run in `~/HPC-POMDP/ML-SC2/ML_SC2` with `ml-sc2` virtualenv:

```bash
git pull --ff-only
python -m unittest discover -s tests -p 'test_*.py' -q
python scripts/70_verify_freeze.py --mode pointer
python scripts/58_verify_final_table.py --check

# Default read-only, validation-only checks (no output files).
python scripts/76_compare_endpoint_baselines.py
python scripts/76_compare_endpoint_baselines.py \
  --split-json data/processed/splits/real_v3_1_fixed_split_seed_43.json
python scripts/76_compare_endpoint_baselines.py \
  --split-json data/processed/splits/real_v3_1_fixed_split_seed_44.json
```

Expected JSON: `"status": "S3_MATCHED_ENDPOINTS_PASS"`, `"index_window_length": 9`, the same `"endpoints"` for RF/XGB/MLP within each seed, and validation endpoint counts **112,312 / 112,943 / 111,899** for seeds 42/43/44.

For historical **test** audit only (not for model selection):

```bash
python scripts/76_compare_endpoint_baselines.py --partition test
```

Expected test endpoint count on seed 42: **141,682**. The full model metrics on this subset are **not** expected to equal the historical full-snapshot test metrics. Save only if an output audit is needed:

```bash
python scripts/76_compare_endpoint_baselines.py --partition val \
  --output-dir results/sequential/s3_seed42_val
```

The output directory must not exist. It will contain `summary.json` and `matched_predictions.parquet` with common keys, labels, and each model's probabilities/predictions. The complete `results/sequential/` directory is already excluded from Git history.

## Common endpoints across sequence lengths

After generating S1 indexes for 17 and 33 snapshots in **distinct** paths:

```bash
python scripts/74_build_sequence_index.py --window 17 \
  --output-dir results/sequential/v3_1_fixed_w17_s1
python scripts/74_build_sequence_index.py --window 33 \
  --output-dir results/sequential/v3_1_fixed_w33_s1

python scripts/76_compare_endpoint_baselines.py \
  --common-index-dir results/sequential/v3_1_fixed_w17_s1 \
  --common-index-dir results/sequential/v3_1_fixed_w33_s1
```

This evaluates the archived tabular predictions only where the 9-, 17-, and 33-snapshot sequences **all** exist. S3 checks matching source dataset hashes and predictor order across indexes. The later temporal models must use precisely these endpoint keys when performing a common-endpoint comparison.

## S3 completion conditions

1. The existing 42-test suite plus five S3 synthetic tests pass (one historical publication test may skip).
2. No B0 historical artifacts, frozen datasets, original predictions, or original tables change.
3. Every seed's validation-only endpoint match passes with the expected counts. If it does not, **do not silently drop mismatched observations**; investigate the group/time/label mismatch.
4. Save the FAIR observed environment and output logs, then approve S4 GRU implementation separately.

S3 does not fit a new MLP baseline; it reuses the historical Deep MLP prediction archive. New train-only MLP reruns may be added in a later, separately declared experiment if needed.

### Locally independently checked seed-42 reference (validation only)

The S3 matching logic was also run against the frozen v3.1 snapshot ZIP and archived seed-42 prediction files, and its input Git blob identities were checked against the files on the feature branch (the historical delivery ZIP used CRLF; the committed files use LF). On the **112,312 shared validation endpoints**, expected retrospective Accuracy is approximately:

| Model | Accuracy |
|---|---:|
| RF train-only | 0.628579 |
| XGB train-only | 0.628722 |
| Historical Deep MLP | 0.630574 |

These values **must not** be compared directly with the published full-snapshot test accuracies: the endpoint population is different and these are validation-set scores, not a new held-out evaluation. They are checks for alignment on FAIR, not evidence to select a new model by itself.
