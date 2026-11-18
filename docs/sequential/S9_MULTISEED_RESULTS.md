# S9 — Three-seed FAIR results and audit gate

## Status (2026-11-18)

Training finished on FAIR for seeds **42, 43, 44** using `scripts/85_train_prefix_vs_last9.py`, identical batch 64 / hidden 64 / one layer / AdamW 0.001 / patience 4 / max 20, full v3.1 64-feature ZIP. Every full run printed **S9_MATCHED_VALIDATION_PASS** and saved 2 checkpoints (model files), two validation prediction CSVs, and a summary in separate `results/sequential/s9_prefix_vs_last9_seed{seed}_v1/` directories. All selected epochs = 3 for BOTH modes across all three seeds. 90 tests (one skipped), B0 pointer/frozen integrity and historical table check had already passed before full 43/44 training.

**Cohorts (complete-case, 25/50/75/90% exact endpoints):**

| Seed | Train replays | Validation replays | Validation predictions per arm |
|---|---:|---:|---:|
| 42 | 8,283 | 2,012 | 8,048 |
| 43 | 8,274 | 2,076 | 8,304 |
| 44 | 8,262 | 2,040 | 8,160 |

No test metrics or predictions generated. Cohorts and endpoints correspond to frozen S8 checkpoint selections.

### Validation accuracy (one prediction per replay and percentage)

| Observed | Last9 seed42 | Prefix seed42 | Last9 seed43 | Prefix seed43 | Last9 seed44 | Prefix seed44 |
|---|---:|---:|---:|---:|---:|---:|
| 25% | .555169 | .552684 | .581888 | .575145 | .560294 | .561765 |
| 50% | .602386 | .619284 | .614644 | .619461 | .611765 | .606373 |
| 75% | .664513 | .670974 | .663295 | .682081 | .675980 | .666667 |
| 90% | .697316 | .704274 | .688343 | .692678 | .697059 | .691667 |

**Unweighted mean across dependent seed splits:**

| Observed | Last9 acc | Prefix acc | Prefix − Last9 accuracy | Prefix − Last9 log loss |
|---|---:|---:|---:|---:|
| 25% | .565784 | .563198 | −0.002586 | −0.001684 |
| 50% | .609598 | .615039 | +0.005441 | +0.000262 |
| 75% | .667929 | .673241 | +0.005311 | −0.006003 |
| 90% | .694239 | .696206 | +0.001967 | −0.013084 |

**Crucial distinction:** While accuracy gains change sign across splits, **Prefix has better log loss at 75% and 90% in ALL THREE seed-specific comparisons**. This is stronger *descriptive consistency*, not proof of external statistical significance or improved ECE/calibration. Prefix log loss was lower at 90% by −.012688 / −.014862 / −.011702 in seeds 42/43/44. The per-seed whole-replay bootstrap 95% accuracy intervals excluded zero only at 50% seed42 and 75% seed43; no interval excluded zero for seed44. Multiple checkpoint/seed comparisons are exploratory.

Training losses indicate **early validation-BCE minima at epoch 3** and substantially lower training BCE with continued epochs in both arms. Do not conclude Prefix always generalizes better or turn the retrospective percentages into an online duration predictor.

## Reproducible aggregate audit (new code, requires FAIR verification)

Run after pulling the new code on FAIR. This **does not retrain or load the ZIP or test**, it reads only the frozen S8 validation checkpoint CSVs and saved S9 validation outputs.

```bash
cd ~/HPC-POMDP/ML-SC2/ML_SC2
source "$HOME/.venvs/ml-sc2/bin/activate"
export PATH="$HOME/HPC-POMDP/.tools/git-lfs/bin:$PATH"
git pull --ff-only
python -m unittest discover -s tests -p 'test_*.py' -q
python scripts/70_verify_freeze.py --mode pointer
python scripts/58_verify_final_table.py --check

python scripts/86_aggregate_prefix_vs_last9.py \
  --output-dir results/sequential/s9_prefix_vs_last9_multiseed_v1
```

Expected report status `S9_MULTISEED_VALIDATION_PASS` and `s8_cohort_crosscheck=exact_key_and_label_match_PASS`. Audit validates source hash, completion markers, hyperparameter parity, exact `replay_id,time_sec,progress_percent,y_true` keys with S8, all four checkpoints per replay, sampled probabilities and class thresholds, and regenerated point metrics. Output is a **fresh** folder with:

- `summary.json`: seed-point metrics, replay bootstrap intervals, unweighted means and limitations
- `metrics_by_seed_checkpoint.csv`
- `mean_checkpoint_metrics.csv`
- `mean_paired_deltas.csv`

The historical seed splits reuse the same replay corpus, so **do not pool their bootstrap replicates, claim three independent replications, or compute a naive three-seed CI**. We have not evaluated a new untouched test; past historical test inspection remains a caveat.

## Scientific next steps

1. Publish audited S9 comparison and test that checkpoints and cohorts are exactly the same as S8.
2. Evaluate score quality further: inspect Brier score (available in S9 summaries), replay-grouped reliability/calibration and per-stage confidence, without claiming that lower log loss necessarily means the model is well calibrated.
3. If a new model experiment is justified, pre-register a controlled regularization or recurrent-history ablation; vary **one factor at a time** across all three seeds. Do not reuse prior output dirs or alter frozen historical results.
4. For a deployable prediction schedule, use observed absolute game time instead of an unknown final fraction; for offline progress characterization the retrospective percentage protocol is appropriate.
