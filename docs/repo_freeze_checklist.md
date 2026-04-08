# Repository freeze checklist

## Repository evidence currently visible on `main`

These items are already supported by committed files or directories:

- [x] final processed datasets are present under `data/processed/`
- [x] replay-group-aware split files are present under `data/processed/splits/`
- [x] final summary tables are present under `tables/`
- [x] final result directories are present under `results/`
- [x] methodology, schema, provenance, and status documents are present under `docs/`
- [x] reproducibility and internal-audit scripts are present under `scripts/`
- [x] final paper and delivery artifacts are present in the repository
- [x] upstream replay provenance documents are present under `data/raw/` and `docs/`

## Freeze sign-off steps to rerun before the next release tag

Check these only after rerunning the relevant commands on the intended frozen state.

- [ ] parser smoke subset rerun completed
- [ ] representative replay audit rerun completed
- [ ] split integrity rechecked
- [ ] final XGB rerun completed
- [ ] final RF rerun completed
- [ ] deep finalist rerun completed
- [ ] rigorous calibration rerun completed
- [ ] `python scripts/15_verify_reproducibility.py --strict` passes
- [ ] `python scripts/17_internal_audit.py --strict-style` passes
- [ ] README, provenance docs, and final summary docs agree on replay source wording
- [ ] GitHub About metadata, topics, and repository license are finalized
- [ ] release tag prepared from the audited state
