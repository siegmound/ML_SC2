# B5 — Paper, presentation and release-readiness audit

**Date:** 2026-11-17  
**Repository:** `siegmound/ML_SC2`  
**Working branch:** `audit/sc2-corrections-2026-11-17`  
**Frozen B0 baseline:** `33660e2b4fc4bd75df0824bbb7fa7c2096d13831`

## 1. B5 status and boundaries

**Canonical editable paper source: audited and corrected.**  
**Publication consistency tooling: implemented.**  
**Compiled canonical PDF/PPTX: NOT YET REGENERATED OR CERTIFIED.**  
**Formal scientific release: BLOCKED_PENDING_EVIDENCE.**

The B0 source and models, final data ZIPs, historical per-seed predictions,
compiled paper, and original presentation were not overwritten. B5
does not perform new experiments or resolve the missing B2/B3/B4 gates.

## 2. Source-of-truth evidence

Primary retained results:

| Evidence | Repository path | Status |
|---|---|---|
| Full tabular RF/XGB 3-seed table | `tables/final_full_multiseed_summary.csv` | B1 corrected, matched to frozen runs |
| Deep historical results | `tables/deep_final_summary.csv` | unchanged; retrospective |
| Calibration historical results | `tables/rigorous_calibration_summary.csv` | unchanged; retrospective |
| Frozen per-run originals | `results/block16_final/`, `results/*/seed_*/` | unchanged |
| Schema/replay census | `data/processed/manifests/b3_*.json` | checked source mappings / v3.1 count |
| B0 protected Git objects | `docs/audit/b0_baseline_manifest_2026-11-17.json` | retained |

The XGB v3.2 full row remains:
Accuracy `0.6128 ± 0.0056`; Balanced Accuracy `0.6093 ± 0.0051`;
ROC-AUC `0.6597 ± 0.0048`; Log loss `0.6486 ± 0.0024`.
These are **mean ± population standard deviation (ddof=0)**.

## 3. Canonical manuscript changes

Edited `paper/sc2_outcome_prediction_paper.tex` with no new numerical
experiments or invented raw replay exclusions.

- Corrected stale claim of full independent reproduction: report is now
  labelled as an **audited retrospective study** until gates are closed.
- Updated PDF metadata to project author from `pyproject.toml`.
- Added a dedicated
  `Audit Status and Reproducibility Boundaries` section before conclusions.
- Disclosed RF train+val vs XGB train/early-stopping budget asymmetry.
- Explicitly documented premature historical Deep candidate test inference.
- Disclosed XGB validation reuse for tuning + calibrator fitting.
- Clarified replay provenance: 24,057 documented source replays,
  23,241 actual frozen v3.1 replay IDs, 816 with unavailable individual
  exclusion reasons.
- Clarified 38 current clean-room feature outputs vs 64/73 frozen
  full-model schemas, with missing raw-to-feature parity.
- Updated appendices: Git LFS datasets are **already tracked**, while
  full v3.2 payload verification and scientific environment locks are open.
- Preserved the four-row historical benchmark table and labelled the
  comparison as descriptive, not a statistically established winner.

**GitHub source consistency** was evaluated with the current table on
the branch: all 4 LaTeX benchmark rows matched their authoritative CSV
numbers, and all B5 methodological caveat text was present.

## 4. PowerPoint review export and display checks

New `scripts/71_refresh_presentation.py` modifies a historical 18-slide
deck **in memory**, generates a separate 19-slide *review* deck, and adds a
visible B5 release-status slide. It patches the four old v3.2 XGB
standard deviations, historical test/calibration claims, and wording
that overstated statistical significance. The source historical PPTX is
never modified by this script.

**Observed local presentation smoke:** the user-supplied source PPTX
`sc2_project_rigorous_presentation.pptx` was revised, emitted 19 slides,
and was exported to a 19-page LibreOffice preview PDF. Slides 10, 13, 14
and 19 were rendered and inspected; overly long cards on slides 10/13
were shortened and re-rendered. Corrected v3.2 XGB metrics were
visually checked on slide 10, with a clear audit-status slide 19.

**Critical source distinction:** the locally supplied PPTX has Git blob
SHA-1 `96975db19ed9fb0145e20ce0a2c3b26ebff178c5`, which differs
from the canonical tracked PPTX Git blob
`facbd43efe63b2052e11799afc95923f58671eee`. Therefore this
local visual export **does not** establish that a clean CI run against
the canonical GitHub PPTX will succeed or reproduce identical layout.
That remains a required review gate.

The separate local two-page
`SC2_B5_Audit_Addendum_REVIEW.pdf` is an explanatory supplement,
**not** the compiled canonical paper.

## 5. Reproducible publication generation

New files:
- `scripts/71_refresh_presentation.py`: creates corrected review PPTX;
- `scripts/72_verify_publication.py`: checks CSV/LaTeX, optional review deck,
  and fails on unapproved release gates;
- `scripts/73_build_b5_artifacts.py`: builds canonical LaTeX PDF and a
  corrected deck into ignored `dist/b5/`, hashes review artifacts and
  writes their build manifest;
- `requirements-publishing.txt`: pins PowerPoint/PDF utilities
  independently from the scientific training dependencies;
- `.github/workflows/b5-publication-review.yml`: runs source checks,
  verifies release is still blocked, then attempts to build and upload
  review-only artifacts without changing original Git binaries;
- `tests/test_b5_publication.py`: protects against stale values,
  premature release, and incorrect 19-slide review construction;
- `docs/audit/b5_release_gates_2026-11-17.json`: explicit
  default-deny release conditions.

Expected operator commands:

```bash
python scripts/72_verify_publication.py

# Requires Python 3.11+, python-pptx, pypdf, latexmk and the full
# TeX Live packages for the canonical paper.
python -m pip install -r requirements-publishing.txt
python scripts/73_build_b5_artifacts.py --output-dir dist/b5
python scripts/72_verify_publication.py \
  --deck dist/b5/sc2_outcome_prediction_presentation_B5_REVIEW.pptx

# Expected to FAIL until all evidence conditions are independently met:
python scripts/72_verify_publication.py --release
```

The review PDFs/PPTXs under `dist/b5/` must not be called the released
canonical files without reviewing compiled text, visuals and source SHA
and closing the conditions below.

## 6. Remaining blocking conditions

The release manifest contains explicit Boolean flags. **Release is not
approved** while any flag is false. Current blocking categories:

1. B2: independent full-scale four-way calibration and Deep finalist reruns;
2. B3: raw replay exclusion IDs, complete original feature formula recovery
   and replay-to-final parity;
3. B4: verified clean checkout, remote CI, v3.2 full Git LFS payload and
   complete scientific runtime lock;
4. B5: compile the exact GitHub LaTeX to PDF and canonical historical PPTX
   to a corrected review deck; verify final fonts/graphics/statistics,
   approve the binary artifacts and release candidate.

No release tag, branch merge, overwrite of canonical PDF/PPTX, or assertion
of newly reproducible held-out model results is authorized yet.

## 7. Engineering sign-off scope

**B5 source correction, publication tooling and local presentation visual
review are complete. End-to-end publication/release is still OPEN.**

An actual GitHub Actions B5 build and its binary output were **not**
observed as completed in this session. The corrected full paper and deck
must be generated from the exact repository sources and visually certified
before treating the GitHub binary artifact versions as final.


## 8. Final remote integrity comparison

GitHub comparison after B5 source, CI, tests and publication-document edits
(2026-11-17), before appending this sign-off:

- `main` is **identical** to B0 immutable baseline commit
  `33660e2b4fc4bd75df0824bbb7fa7c2096d13831`
  (ahead=0, behind=0).
- The correction branch is **96 commits ahead of main**, with **54 changed
  paths** across B0–B5.
- The two canonical binary exports are byte-for-byte unchanged:
  - `paper/sc2_outcome_prediction_paper.pdf` Git blob
    `93799530ae5909e4fa6024609c8afc0b4c7e56a4`
  - `paper/sc2_outcome_prediction_presentation.pptx` Git blob
    `facbd43efe63b2052e11799afc95923f58671eee`
- No changes appear in frozen dataset ZIPs, final RF/XGB/Deep results or
  historical calibration results.
- The B5 audit gate manifest remains **BLOCKED_PENDING_EVIDENCE**,
  with **10 explicitly false gates**.
- Source consistency was cross-checked against all four CSV/LaTeX rows.
  A GitHub Actions B5 run has not been independently confirmed as PASS.
- This sign-off adds **documentation only**, not release authority.

**Final engineering disposition:** B5 editable source corrections and
review-generation machinery are complete, but **release remains blocked**.
Do not merge or tag this working branch as the scientifically certified
final state until the remaining B2, B3, B4 and visual-export gates pass.

## 2026-11-19 B2 progress (supersedes B2 execution-pending statements)

The full FAIR B2 rerun has now **passed 9/9** and a seven-variant
three-seed descriptive summary has been generated. New scientific
interpretation and the source transcript limitations are recorded in
[B2_SCIENTIFIC_RESULTS_2026-11-19.md](B2_SCIENTIFIC_RESULTS_2026-11-19.md).
The dedicated `b2_full_scale_four_way_and_deep_completed` gate was
updated to **true** after those observations, without modifying the
historical frozen results or canonical PDF/PPTX.

B3 raw replay-to-final provenance and the explicit release-candidate
approval remain false. B5 publication text/figures must separately be
reconciled with the retrospective B2 table before any candidate can
be released. `release_status` remains `BLOCKED_PENDING_EVIDENCE`.
