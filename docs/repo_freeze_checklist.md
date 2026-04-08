# Repo Freeze Checklist

This file separates **repository evidence already visible in the current tree**
from the **sign-off steps that should be rerun before a new frozen release**.

## Repository evidence currently visible on `main`

These items are already supported by committed files or directories:

- [x] final processed datasets are present under `data/processed/`
- [x] replay-group-aware split files are present under `data/processed/splits/`
- [x] final summary tables are present under `tables/`
- [x] final result directories are present under `results/`
- [x] methodology, schema, and status documents are present under `docs/`
- [x] reproducibility and internal-audit scripts are present under `scripts/`
- [x] final paper and delivery artifacts are present in the repository

## Freeze sign-off steps to rerun before the next release tag

Check these only after rerunning the relevant commands on the intended frozen state.

### Data pipeline
- [ ] parser smoke test rerun and passed
- [ ] parser audit rerun and exported
- [ ] replay failure reasons exported
- [ ] official dataset manifest regenerated or revalidated
- [ ] feature audit regenerated or revalidated

### Evaluation discipline
- [ ] split files verified against the intended final datasets
- [ ] no train/test replay overlap rechecked
- [ ] all final models verified on the same replay-aware split policy
- [ ] model selection confirmed not to touch test

### Model layer
- [ ] final RF run verified
- [ ] final XGB run verified
- [ ] final deep finalist run verified
- [ ] multi-seed aggregation revalidated

### Calibration and export
- [ ] rigorous calibration workflow rerun or revalidated
- [ ] final tables and figures regenerated from saved structured outputs
- [ ] `scripts/15_verify_reproducibility.py` passed
- [ ] `scripts/17_internal_audit.py` passed
- [ ] README, paper, and final summary tables still agree exactly

## Sign-off note

Do not mark the repository as a freshly frozen release until the rerun-based boxes
above are completed on the exact commit being tagged.
