# S9 — Expanding-prefix GRU versus matched 9-snapshot GRU

## Motivation and S8 evidence

S7 completed on FAIR (`S7_VALIDATION_ABLATION_PASS`): mean validation accuracy 64.038% (v3.1 original64), 64.099% (v3.2 corrected-common64), and 64.117% (v3.2 full73). Net added combat features had little stable impact.

S8 completed on FAIR (`S8_VALIDATION_PROGRESS_PASS`; 83 tests with 1 skipped and frozen artifact audit PASS). All three frozen 9-window GRUs improve from roughly 57.6-57.7% accuracy at 25% of recorded snapshots to 69.4-69.7% at 90%. Mean v3.1 accuracy at 25/50/75/90%: 57.710%, 62.265%, 67.251%, 69.431%. But the 25% gate restricted the complete-case balanced validation cohort to 2012/2076/2040 replay(s) across seeds 42/43/44 out of 3719 per seed (about 54-56% retained). The result is retrospective for this selected population. The checkpoint fraction and full replay length were not model inputs.

**S9 scientific question:** does a recurrent predictor improve if supplied **the entire observed replay prefix** at a selected checkpoint rather than **only the latest nine snapshots**, when the training and validation checkpoint examples are identical?

## Matched experimental arms

- **S9-short9:** GRU receives the final nine snapshots ending at the 25%, 50%, 75%, 90% checkpoint (the final observation in the input is the checkpoint itself).
- **S9-prefix:** GRU receives every *available recorded* snapshot from the first stored observation of the replay through the same checkpoint, inclusive. This input can have varying sequence length. It may contain gaps between sampled timestamps because the original replay metadata has no requirement that every previous 15-second observation is present; missing observations are **not imputed** and no future observation is used.

The GRU backbone remains one layer, hidden size 64, LayerNorm+dropout 0.1+linear head, AdamW 0.001, weight decay 0.0001, validation-BCE-selected checkpoint, 20 epochs, patience 4. New training uses a **common batch size of 64 by default** for both arms (configurable), because packed variable-length sequences are larger than the S4 9-window input. It is **not** identical to the historical S4 batch 512. S9's matched short9 arm is trained from scratch on the *same checkpoint examples* as S9-prefix, which isolates the context-length intervention better than directly comparing to the historical S4 baseline trained on all 9-step windows.

Each seed reuses historical replay-disjoint split groups. Training and validation independently select a *complete-case* cohort satisfying all four exact percentile checkpoints. Their memberships do not overlap. Both models train on the same four per-replay checkpoints and use an identical train-only StandardScaler. The original v3.1 feature representation is used for the initial S9 experiment (64 columns, frozen ZIP hash `65e402884f43c03bf5450d6975e91db3d61ee7e28edf1a56483a37f15d944517`).

The 25%, 50%, 75% and 90% values are **never supplied** to the model as inputs. The model can infer the *absolute number* of received snapshots in the prefix case, which is a legitimate piece of its observable history. The full replay length is only used offline to select examples. The target is the final winner as in all previous stages.

A packed GRU with `pack_padded_sequence`, per-example valid lengths and length-bucketed mini-batches ensures padding is never treated as game history. No explicit maximum prefix length: inputs use every observed snapshot through the checkpoint. Validate memory usage on FAIR (start with smoke). No test dataset is evaluated; the validation split used for checkpoint selection is not a prospective holdout.

## FAIR gate 1 — code and frozen integrity

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

Expected: 83 previous tests + 7 S9 tests (one previous skipped). Do not proceed if any new test fails; correct the implementation in a new commit, never disable integrity checks.

## FAIR gate 2 — two-arm smoke (no artifacts)

```bash
python scripts/85_train_prefix_vs_last9.py --smoke --device auto
```

Both arms should print first-two-epoch short-batch losses and final `S9_SMOKE_PASS`. Confirm `S9_CHECKPOINT_COHORT_PASS` and `device=cuda` as available. No output files are written.

## FAIR gate 3 — seed 42

Only after gates 1 and 2:

```bash
python scripts/85_train_prefix_vs_last9.py \
  --split-json data/processed/splits/real_v3_1_fixed_split_seed_42.json \
  --device auto \
  --batch-size 64 \
  --bootstrap-reps 1000 \
  --output-dir results/sequential/s9_prefix_vs_last9_seed42_v1
```

Expected: `S9_MATCHED_VALIDATION_PASS`; train and validation replay count is reported, but **not pre-guessed**. Validation balanced replay count should be 2012 for seed 42, matching the S8 v3.1 cohort. Output in a fresh directory:

- `summary.json`: two model histories and selected epochs, exact cohorts, per-checkpoint accuracy, ROC-AUC, balanced accuracy, log loss, Brier, and paired replay-bootstrap 95% accuracy/log-loss intervals for **prefix − last9**
- `last9_best_model.pt`, `prefix_best_model.pt`: independent models
- `last9_validation_predictions.csv`, `prefix_validation_predictions.csv`: one prediction per eligible replay and checkpoint, no test data

Check for exploding epoch time or memory usage, and inspect overfitting before proceeding to the other seeds. If FAIR runs out of CUDA memory, **do not rerun with the same existing output path**; use a new path after reducing batch size to 32 for **both arms**, and document that change consistently for all seeds.

## Seeds 43 and 44

After a successful seed-42 comparison, train the exact same model configurations (including batch size) using the historical splits and distinct fresh output folders. Aggregate only after both succeed, preserving per-seed bootstrap intervals and the dependence of seeds on a common replay corpus.

## Limitations

1. Retrospective progress requires full replay length **only to select checkpoint percentages**, not to make model predictions. This is an offline characterization, not an online game-time scheduling algorithm.
2. The S8 complete-case cohort can favor longer or regularly sampled games. Report selection/coverage in any paper.
3. Full prefixes may span missing historical snapshot intervals; they are not a uniformly spaced time series across such gaps. A future improvement can include observed `delta_t` as a separately documented feature.
4. Comparing S9-short9 with S9-prefix is more controlled than comparing S8 historical GRU with S9-prefix because each S9 arm is trained on identical samples; however different input sequence lengths imply different gradient trajectories and effective temporal capacity.
5. Validation selected every checkpoint; all findings are descriptive, not prospective test or external generalization.
