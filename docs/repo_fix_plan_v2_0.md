# v2.0 Repository Fix Plan

This file lists the **non-file changes** and **structural follow-ups** that should be
applied after copying the mandatory replacement files from this fix bundle.

## A. Non-file changes to apply on GitHub

### 1) Fill the repository About box
Set a short public description such as:

`Rigorous StarCraft II outcome prediction remake with replay-group-aware evaluation, final artifacts, and calibration analysis.`

### 2) Add repository topics
Recommended topics:

- `machine-learning`
- `starcraft-ii`
- `xgboost`
- `random-forest`
- `calibration`
- `tabular-data`
- `reproducibility`

### 3) Decide and add a license
This is a policy choice, not an automatic code fix.
Pick a license explicitly instead of leaving reuse terms undefined.

## B. Structural cleanup still recommended

### 1) Archive public-facing historical text files
These files are informative but look transitional rather than final:

- `docs/README_missing_scripts_bundle.txt`
- `docs/README_rigorous_calibration.txt`

Move them to a clearer historical location such as:

- `notes/legacy_docs/`
- or `delivery/archive_docs/`

### 2) Reduce the public script surface
Keep the numbered historical runners if you need them, but expose a smaller public-facing
set of canonical wrappers or aliases for the final workflows.

Suggested future command surface:

- `scripts/final_run_xgb.py`
- `scripts/final_run_rf.py`
- `scripts/final_run_deep.py`
- `scripts/final_run_calibration.py`

These can call the existing numbered scripts internally if you want to preserve history.

### 3) Reformat compressed Python files
Run a formatter on the repository and inspect any large one-line or nearly one-line files:

```bash
python -m pip install black
black scripts src
```

Then rerun:

```bash
python scripts/17_internal_audit.py --strict-style
```

### 4) Recheck README after cleanup
After moving legacy docs or introducing canonical wrappers, update `README.md` so that:
- only the intended public commands are highlighted
- the final datasets section remains accurate
- the final status narrative matches the actual tree

### 5) Paper packaging
If the PDF is the canonical public artifact, state that explicitly.
If the source `.tex` is canonical, include the full final source, not only refinement fragments.

## C. Recommended validation after applying this bundle

From repo root:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python scripts/15_verify_reproducibility.py --strict
python scripts/17_internal_audit.py
```

If you also reformat code:

```bash
python scripts/17_internal_audit.py --strict-style
```

## D. Suggested commit sequence

```bash
git add README.md requirements.txt pyproject.toml docs/known_limitations.md docs/repo_freeze_checklist.md scripts/15_verify_reproducibility.py scripts/17_internal_audit.py docs/repo_fix_plan_v2_0.md
git commit -m "align packaging, repo status, and audit tooling for v2.0"
```
