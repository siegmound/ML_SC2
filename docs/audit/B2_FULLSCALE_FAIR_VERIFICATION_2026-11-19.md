# B2 — Full-scale FAIR execution and numerical review handoff

**Date:** 2026-11-19  
**Branch:** `audit/sc2-corrections-2026-11-17`  
**Dataset:** frozen `real_v3_1_fixed` (64 features)  
**Result namespace:** `results/b2_fullscale_audit_20261119/`

## Evidence status

**Full-size run and structural verification: 9/9 PASS**, observed in the FAIR
terminal transcript supplied by the operator on 2026-11-19.

- RF seeds 42, 43, 44: PASS (strict four-way protocol).
- XGBoost seeds 42, 43, 44: PASS (strict four-way protocol).
- Deep MLP seeds 42, 43, 44: PASS (validation-selected winner only).
- `bash scripts/74_run_b2_full_fair.sh --verify` reported:
  `PASS B2 verified 9 run(s); retrospective test was previously inspected`.
- The full-data preflight verified the protected B0 Git objects and LFS
  payload hashes, plus replay/label identity against archived B2 evidence.
- Operator runtime: Linux FAIR, Python 3.12.3 in `~/.venvs/ml-sc2`,
  PyTorch 2.15.0+cu132, XGBoost 3.4.1, RTX 5090 (CUDA GPU 0).
- Synthetic B2 full-FAIR regression tests: 4/4 PASS (operator transcript).
- The original calibration RF/XGB results and original paper/PPTX remain frozen.

**Important provenance limit:** this file records the operator's supplied
terminal transcript. It does not contain the nine numeric `metrics_summary.json`
files or prediction CSVs, which remain in the FAIR result directory. The
independent 9/9 verifier PASS is a structural and numerical consistency check;
it does not itself establish accuracy, superiority or statistical significance.
The old test replay groups had already been examined during development.

## Pending scientific inspection

Use `scripts/76_summarize_b2_full_fair.py` to generate a read-only-derived
analysis from the nine reports. It first re-runs the nine-run verifier, then
writes only to the new `analysis/` directory inside the B2 result namespace:

```bash
cd ~/HPC-POMDP/ML-SC2/ML_SC2_tabular_audit_clean
source ~/.venvs/ml-sc2/bin/activate
export SC2_PYTHON="$(command -v python)"
export PYTHONPATH="$PWD/src"
git pull --ff-only origin audit/sc2-corrections-2026-11-17
"$SC2_PYTHON" -m unittest discover -s tests -p 'test_b2_summary.py' -v
"$SC2_PYTHON" scripts/76_summarize_b2_full_fair.py
```

Expected files:

- `results/b2_fullscale_audit_20261119/analysis/per_seed_metrics.csv` — 21
  seed-variant rows (3 RF + 3 XGB calibrations × 3 seeds; 3 Deep rows).
- `results/b2_fullscale_audit_20261119/analysis/three_seed_summary.csv` —
  seven model/variant rows with arithmetic means and sample SDs.
- `results/b2_fullscale_audit_20261119/analysis/three_seed_summary.md` —
  short, source-derived table, shown on stdout as well.
- `results/b2_fullscale_audit_20261119/analysis/evidence_manifest.json` —
  SHA-256 of all nine source metrics summaries and analysis scope.

The summary deliberately keeps Deep's unavailable Brier/ECE/MCE cells blank
rather than manufacturing values; RF and XGB have all seven saved metrics.
All three calibration variants for RF/XGB are reported. Do **not** select a
calibration method based on the previously inspected test scores.
The three seeds are not independent holdout datasets: sample SD is descriptive
and must not be misrepresented as a confidence interval or a significance
test. Deep's three-way selection split is **not** identical to the RF/XGB
four-way tuning/calibration design; feature profiles also differ.

## Remaining closure gates

- [x] Run all nine full-size seed/model jobs on FAIR.
- [x] Independently verify frozen partitions, output integrity and saved scores.
- [x] Preserve historical evidence and avoid overwriting final PDF/slides.
- [ ] Inspect the **actual numerical table** from FAIR and calibration changes.
- [ ] Decide whether any scientific text or figure needs to change.
- [ ] Approve B2 scientific sign-off and update the release gate if warranted.
- [ ] Resolve B3 raw-replay-to-feature/exclusion provenance separately.
- [ ] Complete B5 paper/PPTX reconciliation and release candidate review.

**Do not mark the full project release approved from the 9/9 PASS alone.**

## Numerical review completed — operator transcript, 2026-11-19

The operator successfully ran `scripts/76_summarize_b2_full_fair.py`
on the same FAIR checkout. Its fresh nine-run verification returned
**9/9 PASS** and the new summary regression suite returned **2/2 PASS**.
It emitted seven model/variant mean ± sample-SD rows and 21 per-seed
variant rows.

The actual summary table, numerical comparison, historical cross-check and
scientific limitations are now recorded in
[B2_SCIENTIFIC_RESULTS_2026-11-19.md](B2_SCIENTIFIC_RESULTS_2026-11-19.md).

**Revised checklist (supersedes only the prior pending numerical-review items):**
- [x] Inspect all seven three-seed aggregate rows and calibrator outcomes.
- [x] Prepare a separate B2 scientific table and publication guidance.
- [x] Mark the factual nine-run B2 completion gate in B5 release manifest.
- [ ] Archive the FAIR-generated small numerical CSV/manifest inputs and
      complete the independently gated B5 corrected paper/deck release review.
- [ ] Resolve B3 original raw replay lineage and exclusions.

The source result JSON and per-snapshot prediction CSVs were not copied
here. The B2 claims are retrospective and do not authorize a release tag.
