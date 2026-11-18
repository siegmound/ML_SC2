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
