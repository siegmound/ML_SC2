# S4 — First 9-snapshot GRU (validation-only training gate)

**Branch:** feature/sc2-sequential-models. **Depends on:** FAIR-validated S0/S1/S2/S3; S3 completion at commit a08472e.

## Research question and isolation

Does a unidirectional GRU using exactly the 9 consecutive v3_1_fixed snapshots improve match-winner prediction on the SAME validation endpoints as the frozen RF/XGB/MLP predictions from S3?

This first implementation fits a single predeclared architecture: input 64, hidden state 64, one recurrent layer, LayerNorm + dropout 0.1, scalar binary logit. BCEWithLogitsLoss, AdamW, clip norm 1.0. Default training is 20 epochs max, patience 4, selection by lowest complete validation BCE (minimum improvement 1e-4). Batch size 512, learning rate 1e-3, weight decay 1e-4.

- Historical replay-level seed splits (42, 43, 44) are preserved; never create a row-level random split.
- The S2 StandardScaler is fit on distinct **train** snapshots, never on validation/test.
- No changes to frozen data, historical models, predictions, S3 outputs, or B0 artifacts.
- The S4 CLI **does not evaluate test**, either during smoke or full training. It writes **validation-only** predictions after restoring the best validation-selected checkpoint.
- Comparisons to S3 are retrospective: historical validation has already been used for other models. A future test protocol must explicitly acknowledge previous test inspection and replay dependence.
- All validation predictions are checked against the exact S1/S3 replay/time/label endpoint population. Never silently drop mismatches.
- This is the **initial GRU configuration** for S4, not evidence of an optimized or superior temporal model.

## FAIR commands: gate 1 — update and check

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

Expected: existing 47 tests + 5 new S4 synthetic tests (one historical test may skip). Frozen B0 integrity and publication table still pass.

## FAIR commands: gate 2 — smoke training (no artifact writes)

```bash
python scripts/77_train_gru.py --smoke --device auto --batch-size 32
```

Smoke fits at most 2 train batches and 2 validation batches per epoch, for up to 2 epochs. It verifies a complete forward/backward/optimizer/validation cycle on FAIR. It emits JSON with `S4_SMOKE_PASS` and selected epoch. **The smoke scores are not scientific performance estimates.** If auto selects CPU and CUDA is expected, check CUDA visibility before launching a full experiment.

## FAIR commands: gate 3 — first full validation training, seed 42

Run **only after gate 1 and 2 pass**:

```bash
python scripts/77_train_gru.py \
  --split-json data/processed/splits/real_v3_1_fixed_split_seed_42.json \
  --device auto \
  --output-dir results/sequential/s4_gru_w9_seed42_v1
```

Expected full-run status: `S4_VALIDATION_PASS`, with 112,312 validation endpoints for seed 42. The target directory must not already exist. It produces:

- `best_model.pt`: model state, architecture settings, source dataset hash, selected epoch
- `validation_predictions.csv`: S3-compatible keys and probabilities `replay_id,time_sec,y_true,y_prob,y_pred`
- `summary.json`: model parameters, history, validation metrics, source identity, Python/Torch versions

Note the full script currently emits an epoch history only at completion, not live per-epoch progress.

## Later multi-seed runs — after seed-42 checks

```bash
python scripts/77_train_gru.py \
  --split-json data/processed/splits/real_v3_1_fixed_split_seed_43.json \
  --device auto \
  --output-dir results/sequential/s4_gru_w9_seed43_v1

python scripts/77_train_gru.py \
  --split-json data/processed/splits/real_v3_1_fixed_split_seed_44.json \
  --device auto \
  --output-dir results/sequential/s4_gru_w9_seed44_v1
```

Expected endpoint counts by seed (validation): 42=112,312; 43=112,943; 44=111,899. Validation metrics can be descriptively compared to S3 only for matching seeds and endpoint populations, NOT to full-snapshot published test accuracy. Confidence intervals must account for replay-level correlation; window rows are not independent draws.

## Completion criteria

1. FAIR test suite and frozen-artifact integrity remain PASS.
2. CUDA or explicitly accepted CPU smoke reaches `S4_SMOKE_PASS` without test access.
3. Seed 42 completes with `S4_VALIDATION_PASS` and 112,312 exact matching endpoints.
4. Check finite losses/metrics, saved artifact schema, model selected epoch, and clean Git worktree.
5. Only then decide whether to extend to seeds 43/44 or revise the architecture; do not use historical test scores for that decision.

Note: the Git LFS post-merge hook may still warn if git-lfs is absent from PATH. This does not invalidate already-resolved v3_1 dataset bytes verified by the S2 SHA check; it is a separate environment maintenance issue.
