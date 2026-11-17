# Paper artifacts

The canonical deliverables on the ML_SC2 v2.0 line are:

- `sc2_outcome_prediction_paper.tex`: editable LaTeX paper source.
- `sc2_outcome_prediction_paper.pdf`: compiled paper.
- `sc2_outcome_prediction_presentation.pptx`: final slide deck.

See [README_paper_repo.md](./README_paper_repo.md) for naming and provenance.

## B1 synchronization status

The editable `.tex` has the corrected XGB / v3_2_combatfix population
standard deviations, sourced from `results/block16_final/block16_full_summary.csv`.
The committed PDF and PowerPoint are not yet regenerated. Their old table
values remain **historical** until the B5 document export step. Do not
describe the compiled files as corrected before that step is completed.

Source-of-truth check:

```bash
python scripts/58_verify_final_table.py --check
```
