# S12 — Does adding more train-only history cutoffs improve winner accuracy?

## Preregistered narrow change

S11 completed on FAIR: `S11_MATCHED_VALIDATION_MODELS_PASS`, 106 tests OK (one skipped). Across identical retrospective validation replay checkpoints, recurrent models have the best *mean* accuracy at each of 25%, 50%, 75%, and 90%, but S8 GRU's within-seed replay-bootstrap advantage against archived RF/XGB/MLP is **not uniformly supported**: at 75% seed42 all three tabular-minus-GRU accuracy CIs are negative, while the corresponding CIs for seed43/44 cross zero; at 90% seed43 XGB/MLP favor GRU, while many others cross zero. Never claim independent replications, multiple-comparison-adjusted significance, or an untouched test evaluation.

S9 trained a short-nine and full-prefix GRU on **four checkpoint examples per replay** and retained only 8,283 / 8,274 / 8,262 train replays in seeds 42/43/44. In contrast, S4 trained the fixed-nine GRU on hundreds of thousands of overlapping nine-step windows. S12 asks whether the *same S9 models* trained with more varied, strictly historical cutoffs can improve accuracy and probabilistic scoring, **without increasing network size or changing validation selection**.

**Design:** retain the exact same S9 complete-case TRAIN replay IDs and all original four (25/50/75/90%) TRAIN anchors. Add up to **12 extra time-valid S1 endpoints per train replay**, evenly spaced between the earliest and latest anchor endpoint, excluding original anchors. There are normally 16 or fewer train observations/replay (4 original plus up to 12 additional); irregular replays with fewer valid endpoints remain in the same replay cohort and contribute fewer extras. Only train windows are read for selection; test and validation replay IDs are never sampled. Percent tags or final game length are not model inputs.

Both freshly trained S12 models see exactly the same expanded train endpoint/label pairs; only the input start differs (`last9` vs `prefix`). S12 reuses `CheckpointSequenceDataset`, length-aware packed GRU, normalized feature values from S9 train-only scaler, AdamW, hidden size 64, batch size 64, learning rate 0.001, max epochs 20 and patience 4. The frozen S9 configuration is verified byte-for-byte by parsed fields before training. The **same 4 selected checkpoint examples** and identical complete-case validation replays as S9 are used for early stopping and validation reporting; frozen S9 predicted probabilities are loaded only for validation comparisons, NEVER training. Fixed S9 benchmarks remain read-only.

S12 is **not an equal compute-step experiment**: more train endpoints cause more optimizer steps per epoch, and the training time distribution changes. A positive S12−S9 difference would support the intervention *more varied training samples under the same model/configuration*, not isolate sample count from gradient budget or architecture effects. Additionally, selecting extra train endpoints based on complete train replay geometry is offline sampling, while each model only receives features from times at/before the selected endpoint.

## FAIR: test, smoke, then seed 42

```bash
cd ~/HPC-POMDP/ML-SC2/ML_SC2
source "$HOME/.venvs/ml-sc2/bin/activate"
export PATH="$HOME/HPC-POMDP/.tools/git-lfs/bin:$PATH"
git pull --ff-only
git status --short --branch
python -m py_compile src/sc2proj/sequential/extra_prefixes.py scripts/89_train_more_prefixes.py
python -m unittest discover -s tests -p 'test_*.py' -q
python scripts/70_verify_freeze.py --mode pointer
python scripts/58_verify_final_table.py --check
```

Expected **112 tests** total (106 earlier + 6 S12, 1 historic skip). This is NOT yet run on FAIR. If tests fail do not launch full training.

```bash
python scripts/89_train_more_prefixes.py \
  --smoke --device auto --extra-per-replay 12
```

Expected `S12_EXPANDED_TRAIN_COHORT_PASS` and `S12_SMOKE_PASS`. Smoke does not create an output directory.

Once smoke passes, train the first controlled pair (same existing seed42 split, no test):

```bash
python scripts/89_train_more_prefixes.py \
  --split-json data/processed/splits/real_v3_1_fixed_split_seed_42.json \
  --extra-per-replay 12 \
  --batch-size 64 \
  --max-epochs 20 \
  --bootstrap-reps 1000 \
  --device auto \
  --output-dir results/sequential/s12_more_prefixes_seed42_v1
```

Expected terminal status `S12_EXPANDED_TRAIN_VALIDATION_PASS`. The script prints per-checkpoint S9-vs-S12 same-mode accuracy and log-loss differences and saves model weights and validation checkpoint probabilities. Do NOT overwrite an existing output directory. Check overfitting, training lengths, GPU utilization and validation scores before continuing with seed43 and seed44. If the status differs or a traceback occurs, fix the source with a new commit and rerun into a fresh output dir.

## Read-only results and controlled comparisons

When seed42 succeeds, saved files:

- `summary.json`: original four S9 training anchors, new train endpoint counts and min/max extras/replay, all provenance/configuration, early stopping, matched validation scores and replay-aware paired 95% CIs for **S12 minus S9**, stage-by-stage and mode-by-mode
- `last9_best_model.pt`, `prefix_best_model.pt`: new S12 weights
- `last9_validation_predictions.csv`, `prefix_validation_predictions.csv`: predictions on exactly the *same* S8/S9 validation replay/time/label quartet

All previous S7–S11 outputs remain untouched. Stage-conditioned accuracy improvements must be evaluated jointly with log loss, Brier and ROC-AUC; improvements in ECE/calibration are not assumed. This remains retrospective validation reused for model selection, not new test evidence.

Only after successful seed42 and methodological assessment should the identical intervention be repeated for seeds 43 and 44. Avoid adding Transformer/TCN, new temporal features, extra replay populations, spatial replay tensors, weight tuning or altered gradient budgets simultaneously; those would confound the interpretation of this narrow experiment.
