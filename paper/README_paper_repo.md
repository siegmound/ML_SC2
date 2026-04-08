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

## Repo note

If you have a newer manually updated presentation file with the corrected slide 3 pipeline text, replace `sc2_outcome_prediction_presentation.pptx` with that revised deck before pushing to the repository.
