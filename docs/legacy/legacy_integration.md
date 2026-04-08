# Legacy integration

This document describes how the remake ingests legacy frozen project artifacts when the full raw dataset is not available in the current workspace.

## Inputs supported

- top-level freeze zip, such as `sc2_project_freeze_final.zip`
- nested result zips inside the freeze package
- JSON summaries
- prediction CSV files
- calibration CSV files

## Outputs produced

- zip inventory
- nested zip inventory
- canonicalized summary JSON files
- imported prediction CSV files
- prediction schema JSON files
- artifact-only dataset manifest
- artifact-only experiment manifest
- schema validation report

## Non-goals

This flow does not claim to recreate the training dataset from summaries alone.
If the raw dataset zip is missing, the remake must say so explicitly.
