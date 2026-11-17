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

## B2 methodological sign-off (must precede release tag)

- [x] Train-only RF/XGB retrospective evidence recorded and verifiable from stored predictions
- [x] Historical RF train+val versus XGB train-only training-budget asymmetry documented
- [x] Deep candidate runner no longer evaluates test during candidate search (toy regression test)
- [x] Calibration helper rejects validation/test replay overlap
- [x] Strict grouped train/tune/calibrate/test runner added and smoke-tested for RF and XGB
- [ ] Full-size strict four-way RF/XGB run completed for all three seeds
- [ ] Full-data Deep selected-candidate-only run completed for all three seeds
- [ ] Revised methodology conclusions synchronized with regenerated PDF and PPTX
- [ ] New full-scale outputs independently checked and evaluated for statistical uncertainty

These checks are distinct from the historical results and from B4 clean-clone reproducibility sign-off.

## B3 parser/provenance completion gates

- [x] Reject missing, ambiguous or foreign winner IDs before label generation.
- [x] Calculate recent losses from tags known at the snapshot time.
- [x] Detect colliding replay basenames and protect frozen dataset outputs.
- [x] Record per-input source paths and reason-coded exclusions in new builder manifests.
- [x] Add automatic feature-lineage manifest and drift check for 38/64/73 schemas.
- [x] Independently census frozen v3.1 replay IDs and outcome consistency.
- [x] Synthetic/parser/builder/provenance unit tests and combined B1+B2+B3 tests PASS (23/23).
- [ ] Reconstruct original 64/73-feature formulas and prove replay-to-table parity.
- [ ] Obtain raw replaypack-level audit identifying the original 816 missing replays by exact ID and exclusion reason.
- [ ] Pull full v3.2 ZIP via Git LFS and independently validate its archive/header and content.
- [ ] Re-run representative real replay parser tests and compare against frozen reference outcomes.

The unchecked items are *not* implicitly completed by the schema manifest or synthetic tests.

## B4 remote reproducibility and environment gates

- [x] Minimum Python version set to 3.11, matching `tomllib`.
- [x] Frozen Git objects and Git LFS pointer validation implemented for 31 B0 artifacts.
- [x] Separate actual dataset payload SHA-256/size/ZIP-header verification implemented.
- [x] GitHub Actions workflow added to correction branch, with fast and opt-in full jobs.
- [x] Optional RF checkpoint and selected Deep weights/scaler persistence tested.
- [x] Local combined regression suite PASS (30/30).
- [ ] Fresh independent GitHub clone and exact dependency installation verified.
- [ ] GitHub Actions fast job confirmed successful from its actual run logs.
- [ ] Git LFS full v3.2 payload re-fetched and SHA-256 verified.
- [ ] GitHub Actions full-data opt-in job run and evidence archived.
- [ ] Full ML dependency/driver/torch/XGBoost/CUDA lock reproduced.
- [ ] Full B2 calibration/Deep reruns completed, checked against claimed final results.

The checked items document engineering and local test evidence only;
they do not imply that either remote CI job has passed or that the
scientific run results have been exactly reproduced.
