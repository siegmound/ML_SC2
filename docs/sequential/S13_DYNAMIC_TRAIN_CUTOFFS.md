# S13 — Dynamic replay-stratified cutoffs at S9's per-epoch training budget

## Motivation and frozen observation

S12 seed42, trained on four anchors plus twelve intermediate snapshots per replay, passed all 112 tests and produced `S12_EXPANDED_TRAIN_VALIDATION_PASS`. S12 had 132,528 training examples per epoch against S9's 33,132. The S12 Prefix GRU validation BCE increased rapidly after epoch1 while train BCE fell (early stopping retained epoch1); its 90% validation log loss was **+0.0077808** worse than S9 Prefix, with a conditional paired replay-bootstrap 95% CI **[+0.0020755,+0.0138125]**. S12 Last9's modest point-accuracy gains had intervals crossing zero. These results are conditional and validation-only, not generalizable test scores.

**S13 hypothesis:** training on *different historical cutoff times across epochs* may teach a more robust representation without multiplying optimization steps per epoch by four. This is a narrow intervention to address S12's confounding of sample diversity with step budget, not a new architecture.

## Design: unchanged cohort, optimizer, validation, budget

1. Read exact frozen full v3.1 S1 index, SHA-256 and historical replay-disjoint split; reject any mismatch. As S9/S12, use the S8 complete-case replay subset requiring four exact retrospective 25/50/75/90% checkpoints.
2. Deterministically construct the exact S12 training-only candidate pool: four original S9 anchors and twelve intermediate valid snapshots between the earliest and latest train anchor. **Never sample validation or test replay groups into train.** `--extra-per-replay` is fixed to 12 in this experiment.
3. Divide the available candidates *per training replay* into four time strata using the midpoints between the observed-snapshot counts of the original four anchors. Every stratum contains its corresponding original train anchor.
4. At the beginning of every epoch, use a seed-and-epoch deterministic RNG to select **exactly one endpoint uniformly per stratum per training replay**. This produces exactly four examples per replay, 33,132 examples/epoch in seed42, and **518 batches/epoch at batch size 64**, matching S9's number of training-example updates per epoch.
5. Fit **two independent models** with the identical sampled train cutoff/time-label pairs per epoch: GRU Last9 (nine latest snapshots), GRU Prefix (all observed snapshots up to endpoint). The same sampler seed and replay strata yield identical selection per mode; only the sequence context/length changes. Reuse S9's unidirectional packed GRU, 64 hidden units, 1 layer, dropout .1, AdamW lr=0.001, weight decay=1e-4, patience4, maximum20 epochs, identical train-only normalized feature matrix, early stopping on the identical four validation checkpoints.
6. Audit source metadata and frozen S9 validation predictions for exact replay/time/labels, and optionally verify S12 seed42 read-only predictions. Compute paired whole-replay bootstrap differences for **S13 minus S9** (and, if supplied, **S13 minus S12**), separately at 25/50/75/90, with accuracy, balanced accuracy, AUC, log loss, Brier. Retain the old results, never overwrite.
7. Record deterministic SHA-256 checksums of training cutoff selections per epoch and the number of original anchors selected. These checks do not read future snapshot values. Saved models and probabilities use fresh paths.

This is *not* identical to S9: S13 changes the distribution and ordering of training cutoff times, and the optimizer trajectories/selected epoch may vary. It controls the **number of training examples and optimizer steps per epoch**, not the complete stochastic training dynamics. The four offline progress checkpoints still require retrospective replay lengths to select evaluation time, not provided to models as features. The 3 seeds share a source corpus and validation used for selection, so do not treat confidence intervals as independent or multiple-testing adjusted.

## Run on FAIR, seed42 only first

```bash
cd ~/HPC-POMDP/ML-SC2/ML_SC2
source "$HOME/.venvs/ml-sc2/bin/activate"
export PATH="$HOME/HPC-POMDP/.tools/git-lfs/bin:$PATH"
git pull --ff-only

python -m py_compile \
  src/sc2proj/sequential/dynamic_prefix.py \
  scripts/90_train_dynamic_prefixes.py

python -m unittest discover -s tests -p 'test_*.py' -q
python scripts/70_verify_freeze.py --mode pointer
python scripts/58_verify_final_table.py --check
```

Expected, but **not yet run on FAIR**: prior 112 tests + six S13 unit tests = **118 tests** total, with the historical one skipped. Do not run a full training if checks fail.

Next smoke (CPU or CUDA; no saved output):

```bash
python scripts/90_train_dynamic_prefixes.py \
  --smoke --device cuda --extra-per-replay 12 \
  --s12-reference-dir results/sequential/s12_more_prefixes_seed42_v1
```

Expected `S13_DYNAMIC_COHORT_PASS` and `S13_SMOKE_PASS`, train examples per epoch 33,132. After a successful smoke:

```bash
set -o pipefail
python scripts/90_train_dynamic_prefixes.py \
  --split-json data/processed/splits/real_v3_1_fixed_split_seed_42.json \
  --s12-reference-dir results/sequential/s12_more_prefixes_seed42_v1 \
  --extra-per-replay 12 \
  --batch-size 64 --max-epochs 20 --bootstrap-reps 1000 \
  --bootstrap-seed 2026 --device cuda \
  --output-dir results/sequential/s13_dynamic_cutoffs_seed42_v1 \
  2>&1 | tee s13_seed42_training.log
```

Expected `S13_DYNAMIC_VALIDATION_PASS`. The output path must not already exist; a crash after output creation requires a fresh v2 directory.

Generated files: `summary.json` (including S9/S12 same-checkpoint deltas and bootstrap intervals, history and per-epoch sample SHA), `last9_best_model.pt`, `prefix_best_model.pt`, `last9_validation_predictions.csv`, `prefix_validation_predictions.csv`. Frozen S8–S12 artifacts are read-only; the held-out test is never evaluated.

**Decision gate:** inspect epoch-wise train/val BCE for Prefix, 90%-stage log loss and accuracy vs S9/S12, all four progress stages and paired bootstrap uncertainty. Do not extrapolate from a seed42 point improvement; only if the intervention is scientifically promising should it be repeated unchanged on seeds43/44. Do not start Transformer experiments or tune dropout/LR simultaneously.


## FAIR results — seed 42 (2026-11-18)

Executed on FAIR after the `f24ebf3` implementation. Repository unit test suite **118 tests OK, one skipped**; frozen 31-file pointer/object integrity PASS; publication table PASS; S13 smoke and full CUDA execution PASS. Full run directory:
`results/sequential/s13_dynamic_cutoffs_seed42_v1`. The results were provided via FAIR terminal log, not independently rerun outside FAIR.

Training pool: **8,283 train replays**, **132,528 candidate cutoffs** (four historical anchors + 12 extras/replay), **33,132 dynamically sampled train examples per epoch** matching S9's step budget; fixed validation cohort **2,012 replays**. Every replay is represented once in each of four cutoff strata per epoch; all 64 features and both GRU architectures unchanged.

Training history:
- S13 Last9 selected **epoch 5**, minimum validation BCE **0.628513**; training BCE at selected epoch **0.627632**; later validation BCE worsened somewhat.
- S13 Prefix selected **epoch 2**, minimum validation BCE **0.628576**; training BCE at selected epoch **0.637592**. After epoch2, validation BCE rose to **0.640722** by epoch6, while training BCE fell to **0.595863**: overfitting is **reduced/delayed versus S12 seed42 Prefix**, but not eliminated.

Same-replay validation checkpoint accuracy:

| Progress | S9 Last9 | S13 Last9 | S9 Prefix | S13 Prefix |
|---|---:|---:|---:|---:|
| 25% | 0.555169 | 0.553181 | 0.552684 | 0.541750 |
| 50% | 0.602386 | 0.622763 | 0.619284 | 0.619781 |
| 75% | 0.664513 | 0.675944 | 0.670974 | 0.672962 |
| 90% | 0.697316 | 0.702783 | 0.704274 | **0.721670** |

Paired replay-bootstrap examples (S13 − S9, 1,000 resamples, conditional/unadjusted):
- **Last9 50%:** Δ accuracy +0.020378, 95% CI **[+0.006461,+0.036282]**; Δ log loss −0.005312, CI crosses zero.
- **Last9 75%:** Δ accuracy +0.011431, CI crosses zero; Δ log loss **−0.007903**, CI **[−0.013970,−0.001495]**.
- **Last9 90%:** Δ accuracy +0.005467, CI crosses zero; Δ log loss **−0.005293**, CI **[−0.010871,−0.000152]**.
- **Prefix 50%:** Δ accuracy +0.000497, CI crosses zero; Δ log loss **−0.008628**, CI **[−0.016263,−0.002158]**.
- **Prefix 90%:** Δ accuracy **+0.017396**, CI **[+0.002970,+0.030318]**; Δ log loss **+0.005890**, CI **[−0.000248,+0.012683]**. **More correct binary decisions do not imply better overall probabilistic scoring.**
- **Prefix 25%:** Δ accuracy −0.010934, CI crosses zero; early-game regression remains a concern.

S12 Prefix seed42 at 90% was 0.701292 accurate; S13 Prefix 0.721670, an absolute point increase of about **2.04 percentage points**; this between-run delta has not been shown with its own paired confidence interval in the terminal excerpt (although the S13 summary contains it).

**Interpretation:** S13 is promising at seed42. It does *not* establish a robust multi-seed gain, and it does not validate a prospective/online game-progress scheduler. A trained model receives observed-history snapshots only; selecting 25/50/75/90% retrospective endpoints uses completed replay length offline. The same validation data have been used repeatedly in this project, and bootstrap intervals were not corrected for multiple comparisons.

## Next replication — seeds 43 and 44, identical settings

Only when seed42 diagnosis is accepted, run exact same dynamic cutoff experiment on the historical split 43 and 44. S12 is only available for seed42, so omit optional `--s12-reference-dir` on seeds 43/44; the S9 matched reference remains mandatory. Maintain 12 extras/replay, batch 64, max epochs 20, bootstrap seed 2026, CUDA, distinct output paths:

```bash
python scripts/90_train_dynamic_prefixes.py \
  --split-json data/processed/splits/real_v3_1_fixed_split_seed_43.json \
  --extra-per-replay 12 --batch-size 64 --max-epochs 20 \
  --bootstrap-reps 1000 --bootstrap-seed 2026 --device cuda \
  --output-dir results/sequential/s13_dynamic_cutoffs_seed43_v1 \
  2>&1 | tee s13_seed43_training.log

python scripts/90_train_dynamic_prefixes.py \
  --split-json data/processed/splits/real_v3_1_fixed_split_seed_44.json \
  --extra-per-replay 12 --batch-size 64 --max-epochs 20 \
  --bootstrap-reps 1000 --bootstrap-seed 2026 --device cuda \
  --output-dir results/sequential/s13_dynamic_cutoffs_seed44_v1 \
  2>&1 | tee s13_seed44_training.log
```

No claim for these seeds until the FAIR validation outputs are supplied. Aggregate per-seed/during-stage comparisons, inspect Prefix train/val BCE histories, ROC-AUC/Brier/log loss as well as accuracy, and preserve every single frozen result.


## FAIR results — all three seeds, completed on 2026-11-18

Seeds **43 and 44 both passed `S13_DYNAMIC_COHORT_PASS` and `S13_DYNAMIC_VALIDATION_PASS`** on CUDA using the same script and hyperparameters as seed42. Their original terminal logs were supplied to the project review; these are historical **validation** results, not new held-out test runs.

| Seed | Train replay groups | Available cutoffs | Cutoffs per epoch | Validation replay groups | Last9 selected epoch | Prefix selected epoch |
|---|---:|---:|---:|---:|---:|---:|
| 42 | 8,283 | 132,528 | 33,132 | 2,012 | 5 | 2 |
| 43 | 8,274 | 132,384 | 33,096 | 2,076 | 4 | 3 |
| 44 | 8,262 | 132,192 | 33,048 | 2,040 | 4 | 2 |

Both modes continue to show the same **number of optimizer steps per epoch** as S9 (given the same batch size). Prefix validation BCE reaches its selected best checkpoint in epoch 2 or 3 in all seeds, then degrades as train BCE keeps dropping. Overfitting therefore persists, though the severe S12 seed42 Prefix validation collapse is substantially reduced.

### S13 versus historical S9, unweighted seed means

| Arm | Checkpoint | S9 accuracy | S13 accuracy | Accuracy delta (percentage points) | Mean delta log loss |
|---|---:|---:|---:|---:|---:|
| Last9 | 25% | 0.565784 | 0.560287 | −0.550 | −0.001138 |
| Last9 | 50% | 0.609598 | 0.620101 | +1.050 | −0.004706 |
| Last9 | 75% | 0.667929 | 0.671261 | +0.333 | −0.002978 |
| Last9 | 90% | 0.694239 | 0.695222 | +0.098 | −0.002921 |
| Prefix | 25% | 0.563198 | 0.557302 | −0.590 | +0.000627 |
| Prefix | 50% | 0.615039 | 0.615563 | +0.052 | −0.001739 |
| Prefix | 75% | 0.673241 | 0.672300 | −0.094 | +0.001782 |
| Prefix | 90% | **0.696206** | **0.705715** | **+0.951** | **+0.003013** |

At the 90% stage, Prefix S13 minus S9 accuracy is positive in each split:

| Seed | S9 Prefix | S13 Prefix | Accuracy delta (pp) | Within-seed paired replay-bootstrap 95% CI (pp) | Log loss delta |
|---|---:|---:|---:|---|---:|
| 42 | 0.704274 | 0.721670 | +1.740 | [+0.297, +3.032] | +0.005890 |
| 43 | 0.692678 | 0.700867 | +0.819 | [−0.530, +2.168] | −0.000695 |
| 44 | 0.691667 | 0.694608 | +0.294 | [−1.129, +1.814] | +0.003843 |

**Interpretation:** Directionally consistent late accuracy improvement is encouraging but only seed42's unadjusted CI excludes zero. Prefix S13 mean 90% log loss worsens, so it cannot be described as a probability-quality improvement. Prefix early-stage 25% accuracy fell in all three seeds. Last9 improves 50% accuracy in all seeds and its average log loss is lower at all four stages, but not all per-seed intervals exclude zero. These are seeds of the *same* replay corpus, not independent experimental replications; the means and sign counts do not provide population confidence intervals.

### Independent read-only aggregation on FAIR

An additional standalone script `scripts/91_aggregate_dynamic_prefixes.py` audits the three saved S9/S13 `summary.json` and validation prediction CSV files, requiring exact replay/time/winner-label/checkpoint identity. It recomputes 5 metrics in each of 3 × 2 × 4 model/checkpoint cells, verifies stored deltas, paired bootstrap point estimates, config parity, source hash and four train examples per replay per epoch. It **does not load test, fit a model, tune thresholds, or pool seed bootstrap samples**.

After pulling the committed script and tests and passing the expanded unit suite:

```bash
cd ~/HPC-POMDP/ML-SC2/ML_SC2
source "$HOME/.venvs/ml-sc2/bin/activate"
git pull --ff-only

python -m py_compile scripts/91_aggregate_dynamic_prefixes.py
python -m unittest discover -s tests -p 'test_*.py' -q
python scripts/70_verify_freeze.py --mode pointer
python scripts/58_verify_final_table.py --check

python scripts/91_aggregate_dynamic_prefixes.py \
  --output-dir results/sequential/s13_dynamic_cutoffs_multiseed_v1
```

Expected only if the FAIR local files pass independent audit: `S13_MULTISEED_VALIDATION_PASS`. New output files: `metrics_by_seed_progress.csv`, `mean_progress_metrics.csv`, `summary.json`. Output directory must be fresh. Do not rerun S9/S13 training.

**Research decision:** S13 is a credible exploratory training-sampling result, **not** proof that Prefix decisively improves accuracy or calibrates probabilities at 90%. Before any new model selection, present S9/S11/S12/S13 as validation-based observations, report the high test/reuse caveat, and consider a prospective, independently held-out evaluation only under a new rigorously specified protocol.
