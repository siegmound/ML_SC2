# Paper and presentation artifacts

This directory contains the canonical paper and presentation artifacts for the ML_SC2 final delivery.

## Canonical files

- `sc2_outcome_prediction_paper.tex` - canonical editable LaTeX source of the final paper
- `sc2_outcome_prediction_paper.pdf` - compiled final paper PDF
- `sc2_outcome_prediction_presentation.pptx` - final project presentation deck

## Naming policy

The repo now uses stable, deliverable-oriented names instead of version-heavy working names.
These filenames should be treated as the canonical public artifacts for the final delivery.

## Current narrative freeze

The paper and presentation should stay aligned on the following dataset summary:

- Raw replay corpus: `71 replaypacks`, `24,057 replays`
- Eligible replay corpus after replay parsing and eligibility filtering: `23,241 replays`
- Final tabular dataset: `892,047 rows`

## Recommended repo layout

```text
paper/
├── README.md
├── sc2_outcome_prediction_paper.tex
├── sc2_outcome_prediction_paper.pdf
└── sc2_outcome_prediction_presentation.pptx
```

## Migration from older filenames

Old working filenames that can now be retired or moved to an archive folder:

- `final_sc2_remake_paper_v6_2.tex`
- `final_sc2_remake_paper_v6_2.pdf`
- `sc2_project_rigorous_presentation.pptx`
- `final_results_refinement.tex` (keep only if you still want a historical snippet)

## B5 audited source and review-only export status

The canonical LaTeX source now contains the corrected population standard
deviations and a separate methodological/provenance audit section (B2-B4).
The original PDF and PowerPoint checked into `paper/` are still **historical
binary exports**; they have **not** been replaced or verified as B5-corrected.

Use `python scripts/72_verify_publication.py` for source/table consistency.
Install `requirements-publishing.txt` and a LaTeX distribution, then use
`python scripts/73_build_b5_artifacts.py` to generate new **review-only**
PDF/PPTX under ignored `dist/b5/`. The deck generator updates the old
18-slide presentation to 19 slides and adds a release-readiness slide.
Publication CI is defined in `.github/workflows/b5-publication-review.yml`;
no successful remote run is claimed until logs/artifacts have been inspected.

A clean full-data release is gated by the separate B2/B3/B4 outstanding items,
and must not be tagged or merged based only on the corrected prose.

The exact archived pre-correction state is recorded in
`docs/audit/b0_baseline_manifest_2026-11-17.json` on the audit branch.
