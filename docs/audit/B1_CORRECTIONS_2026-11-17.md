# B1 — Structural and statistical corrections

**Date:** 2026-11-17  
**Repository:** `siegmound/ML_SC2`  
**Branch:** `audit/sc2-corrections-2026-11-17`  
**Immutable baseline:** `33660e2b4fc4bd75df0824bbb7fa7c2096d13831`  
**Pre-change inventory:** `docs/audit/b0_baseline_manifest_2026-11-17.json`

## Scope and source of truth

B1 repairs publication statistics, canonical paths, and fast repeatable validation.
No training, data generation, re-splitting, or editing of the archived run results
was carried out in this block.

The reference statistics remain:
- `results/block16_final/block16_full_detailed.csv` (12 per-seed tabular runs).
- `results/block16_final/block16_full_summary.csv` (full-precision means and population standard deviations, `ddof=0`).
- Dataset/experiment artifacts and their SHA-256 recorded by B0.

## B1 completed corrections

- [x] `tables/final_full_multiseed_summary.csv` regenerated from the frozen runs.
- [x] Exactly one summary row was corrected: `v3_2_combatfix` / XGB / full.
- [x] Published corrected standard deviations: Accuracy `0.0056`,
  Balanced Accuracy `0.0051`, ROC-AUC `0.0048`,
  Log Loss `0.0024`.
- [x] `docs/final_report_refined.md` corrected.
- [x] `paper/sc2_outcome_prediction_paper.tex` corrected.
- [x] Root `README.md`, `paper/README.md`, and `paper/README_paper_repo.md`
  point to the canonical artifact names and explicitly flag pending binary exports.
- [x] `scripts/15_verify_reproducibility.py --strict` now checks the
  canonical paths and delegates the Block 16 numeric publication check.
- [x] `scripts/17_internal_audit.py` no longer flags its own marker constants.
- [x] Added `scripts/58_verify_final_table.py` with reproducible
  `--check` and explicit `--write` modes.
- [x] Added `tests/test_block16_publication.py` with three regression tests.

## Regression evidence

A local isolated fixture was built from the **original**
`ml_sc2_remake_final_delivery.zip` Block 16 detailed and full-precision
summary files. The exact GitHub committed versions of the new checker
and tests were reconstructed locally and verified against their Git blob SHA-1:

- `scripts/58_verify_final_table.py`:
  `40cf6bc0f1c22adb0dd17655b600d6d1170db53c`
- `tests/test_block16_publication.py`:
  `ccb6d67a692e744c96f6e4161414da5584e4d4d8`

**Observed tests:** `python -m unittest discover -s tests -p 'test_*.py' -v`
→ **3 tests passed** on the isolated fixture.

Additional observed behavior:
- old publication CSV → checker returns nonzero with a one-row diff;
- regeneration using `--write` → checker returns success;
- tampering with a saved seed ID → checker returns nonzero;
- restoring the original per-seed data → checker returns success.

These results establish the correctness of the statistical checker for the
stored Block 16 CSVs. They **do not** certify a clean repository clone,
full strict audit, or GPU retraining; those are scheduled for B4/B5.

## Required publication sync in B5

**Not yet synchronized:** the committed compiled
`paper/sc2_outcome_prediction_paper.pdf` and
`paper/sc2_outcome_prediction_presentation.pptx`.
They still represent pre-correction exported text/table values.
Regenerate and visually verify both once the scientific protocol is closed.
The editable LaTeX paper source has already been corrected.

## Protected artifacts

No source dataset, split file, raw or per-seed predictions, saved
model metrics, full-precision Block 16 collector output,
or calibration report was changed in B1.

## Commands

```bash
python scripts/58_verify_final_table.py --check
python -m unittest discover -s tests -p 'test_*.py' -v
python scripts/15_verify_reproducibility.py --strict
python scripts/17_internal_audit.py --strict-style
```

The last two whole-repository commands remain to be executed and signed off
from a fresh cloned environment. Their presence here is a reproducibility
recipe, not an assertion that they have already passed.

## B1 closure criterion

B1 is complete **for source/table/README fixes** once the GitHub compare
shows only the intended B0 + B1 files have changed and `main` is unchanged.
PDF/PPTX regeneration is deliberately deferred to B5, and methodological
retraining, replay parsing and full reproducibility remain B2–B4.

## B1 sign-off — 2026-11-17

- **B1 status: COMPLETE** for source tables, documentation paths, and automated statistical regression checks.
- GitHub comparison: `main` is **identical** to baseline commit
  `33660e2b4fc4bd75df0824bbb7fa7c2096d13831`.
- Correction branch is **14 commits ahead of main**, with **13 changed paths**
  at sign-off (including the two original B0 audit documents and this report).
- GitHub readback confirms corrected CSV, Markdown and LaTeX values.
- The Git blob SHA-1 of both new Python files matched the exact locally
  tested code. **3/3 regression tests passed** in the isolated fixture.
- The compiled PDF and presentation SHA-1 values were unchanged from B0:
  `93799530ae5909e4fa6024609c8afc0b4c7e56a4` (PDF);
  `facbd43efe63b2052e11799afc95923f58671eee` (PPTX).
- The final GitHub compare was run before adding this sign-off paragraph.
  This additional B1 documentation commit does not modify program code,
  experimental data, or frozen results.

**Remaining:** full fresh-clone strict checks in B4; regenerate and verify
compiled PDF and PPTX in B5. Do not mark those tasks complete yet.
