# Paper artifacts

The canonical deliverables on the ML_SC2 v2.0 line are:

- `sc2_outcome_prediction_paper.tex`: editable LaTeX paper source.
- `sc2_outcome_prediction_paper.pdf`: compiled paper.
- `sc2_outcome_prediction_presentation.pptx`: final slide deck.

See [README_paper_repo.md](./README_paper_repo.md) for naming and provenance.

## B1+B5 source synchronization; binary exports pending review

The editable `.tex` contains the corrected XGB / v3_2_combatfix population
standard deviations, and now explicitly explains the historical test,
calibration and replay-lineage limitations found in B2–B4.

**The committed PDF and PowerPoint remain historical exports.** The B5 scripts
produce *review-only* PDF/PPTX under `dist/b5`; neither the original binaries
nor the release tag should be replaced until compilation, visual checks,
full-data protocol evidence and independent LFS/CI checks have passed.

Source-of-truth check:

```bash
python scripts/58_verify_final_table.py --check
```
