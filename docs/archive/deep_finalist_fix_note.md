# Deep finalist runner fix

This note records a cleanup applied after block 16 freeze.

## What changed
`scripts/64_run_deep_finalist.py` was replaced with a cleaned version that makes NumPy arrays contiguous and writable before converting them with `torch.from_numpy(...)`.

## Why
The previous version could emit a PyTorch warning about non-writable NumPy arrays during `TensorDataset` construction. The warning did not invalidate completed runs, but this fix removes the implementation fragility and keeps the repository cleaner for future reruns.

## Scope
This change does **not** alter the evaluation protocol:
- same replay-aware splits
- same candidate search on validation
- same final evaluation on test
- same exported artifacts

It is a repository hygiene fix, not a methodological change.
