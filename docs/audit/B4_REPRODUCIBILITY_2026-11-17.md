# B4 — Reproducibility engineering and CI

**Date:** 2026-11-17  
**Repository:** `siegmound/ML_SC2`  
**Branch:** `audit/sc2-corrections-2026-11-17`  
**Immutable B0 baseline:** `33660e2b4fc4bd75df0824bbb7fa7c2096d13831`

## Scope and status

**Engineering implementation: completed and regression-tested.**
**Remote clean-clone / Git LFS payload / full-run sign-off: pending.**

B4 does not retrain on the full StarCraft dataset, rewrite historical
metrics, regenerate the paper PDF, or merge the correction branch into main.

## 1. Git integrity and LFS verification

`scripts/70_verify_freeze.py` consumes the B0 baseline manifest at
`docs/audit/b0_baseline_manifest_2026-11-17.json`. It provides two modes:

- `--mode pointer` (fast/offline): verifies Git HEAD still contains **31
  protected historical artifact blobs** with the exact B0 Git object SHA-1;
  validates both committed Git LFS pointer OIDs and declared byte sizes,
  independently of local LFS smudge/download state.
- `--mode payload` (after `git lfs pull`): additionally streams the actual
  two archived ZIP payloads through SHA-256 hashing, checks exact sizes,
  opens each ZIP directory, and checks ordered CSV feature headers against
  archived full-XGB model configurations. This mode **rejects pointers
  masquerading as downloaded ZIPs** and truncated/corrupt archives.

The 31 protected paths include final per-seed model metric JSONs, final
calibration reports, grouped split manifests, the full-precision Block 16
aggregation, two other published tables, and both final LFS pointers. The
published B1 CSV and edited paper source are not frozen to their original
hashes because their corrections were intentional.

Baseline expected LFS payloads:

| Dataset | Size | SHA-256 |
|---|---:|---|
| v3.1 fixed | 268,539,879 bytes | `65e402884f43c03bf5450d6975e91db3d61ee7e28edf1a56483a37f15d944517` |
| v3.2 combatfix | 294,197,452 bytes | `87ce36435e4a0fc87b8d9421291d3b1fb04d67b5187ede6a2a3b25d0dce1942e` |

**Evidence available locally:** the supplied v3.1 ZIP hashes to the
recorded SHA-256 and is structurally a valid ZIP. The supplied v3.2 ZIP
is **292,044,800 bytes** (shorter than the LFS pointer's 294,197,452 bytes),
with a different SHA-256 and an invalid ZIP central directory. This means the
**local v3.2 copy is incomplete**, not that the GitHub LFS object is corrupt.
The remote LFS payload remains unverified.

GitHub's checked-in *pointer files* were read back and match B0. The local
container does not have the Git LFS executable, and `git ls-remote` to GitHub
failed due to DNS resolution. Therefore, a fresh remote clone, `git lfs pull`
and both real payload checks **could not be completed from this session**.

## 2. GitHub Actions workflow

`.github/workflows/b4-reproducibility.yml` defines:

- **Metadata job:** push to the correction branch, pull requests targeting
  `main`, and manual dispatch. Uses Python 3.11, a minimal pinned PyYAML
  dependency, `actions/checkout` without large LFS files, the Git-blob/pointer
  checker, strict repository/feature/table checks, syntax auditing, and
  stdlib-only regression tests. Writes reproducibility/audit reports.
- **Full-data job:** opt-in only via `workflow_dispatch`. Downloads *only*
  the two frozen dataset LFS payloads, verifies SHA-256 and ZIP schemas,
  runs a real replay-label census, the whole 30-test synthetic suite, and
  the saved matched-fit comparator, then saves a Python/pip/Git environment
  fingerprint. It does **not** retrain the six full-scale ML finalists.

**Workflow status is not claimed PASS until an actual GitHub Actions run
has completed and its logs are inspected.** GitHub normally requires a
`workflow_dispatch` workflow to exist on the default branch before manual
dispatch is usable, so the full-data job may require promoting the workflow
after review. No direct change to main was made during B4.

## 3. Runtime environment and dependency limits

- `pyproject.toml` now declares Python **>=3.11**, matching the committed
  `environment.yml` (Python 3.11) and standard-library `tomllib`.
- `requirements-ci.txt` pins **PyYAML 6.0.3** only for the light CI job.
  This is **not** a lock for scientific model training.
- Main `requirements.txt`/`environment.yml` still use open-ended version
  constraints for NumPy, scikit-learn, XGBoost, PyTorch and others.
  The opt-in full CI job captures `pip freeze`, interpreter version and
  commit SHA, but runtime/driver/CUDA lockfiles and matching full-scale
  replay experiments are still needed for exact scientific reproduction.
- GPU floating-point arithmetic and parallelism can vary across drivers,
  device models and library releases; exact numerical parity is not assumed
  without comparison tolerances and environment evidence.

## 4. Model checkpoints and inference evidence

- The historical XGB runner already exports `model.json`.
- `scripts/54_run_rf_full.py` now supports **`--save-model`** (opt-in),
  storing `model.joblib` in the chosen run directory and listing it in
  the artifact manifest. Previous behavior remains the default.
- The selected-only Deep MLP runner now stores
  `selected_model_state.pt` (CPU tensors, load with
  `torch.load(..., weights_only=True)`) and
  `selected_scaler.joblib`; the manifest lists both.
  Saved weights and scalers come from the validation-selected model only.
- Reloading joblib/pickle files from untrusted sources is unsafe;
  restrict loading to artifacts produced by the trusted project.
- New test `tests/test_b4_checkpoints.py` confirms the RF checkpoint
  regenerates the stored test probabilities and that the Deep weights/scaler
  are readable and correctly shaped on a small synthetic example.

## 5. Actual test evidence

A local reconstructed test workspace combined the B3 verified files with
the exact new B4 Git blobs. The source versions of the modified RF/Deep
runners, LFS checker and B4 test files were checked against GitHub blob IDs.

- B4 freeze checker tested on a fresh **local temporary Git repo**:
  accepted intact pointer/payload, rejected pointer-only download in
  payload mode, detected committed metrics tampering, and detected
  committed pointer tampering.
- Checkpoint synthetic integration tests PASS (RF and Deep).
- Combined `python -m unittest discover -s tests -p 'test_*.py' -q`:
  **30 tests run, 30 PASS**.

These are valid *local tests of code*, **not** evidence of a successful clone
from GitHub, remote CI execution, full Git LFS payload availability or GPU
training reproduction.

## 6. Recommended operator commands

```bash
git clone --branch audit/sc2-corrections-2026-11-17 https://github.com/siegmound/ML_SC2.git
cd ML_SC2
python3.11 -m venv .venv
source .venv/bin/activate

python -m pip install -r requirements-ci.txt
python scripts/70_verify_freeze.py --mode pointer
python scripts/15_verify_reproducibility.py --strict
python scripts/17_internal_audit.py
python -m unittest discover -s tests -p 'test_b4_freeze.py' -v

# Only on a machine with Git LFS installed and enough disk space:
git lfs install
git lfs pull --include="data/processed/starcraft_full_dataset_v3_1_fixed.zip,data/processed/starcraft_full_dataset_v3_2_combatfix.zip" --exclude=""
python scripts/70_verify_freeze.py --mode payload
```

For full CPU synthetic/integration tests, install all dependencies
(`pip install -r requirements.txt`, `pip install -e . --no-deps`) and run
`python -m unittest discover -s tests -p 'test_*.py' -v`.

## 7. B4 closure checklist

- [x] Verify existing remote LFS pointer OIDs from GitHub and preserve B0.
- [x] Implement offline Git SHA-1 / pointer integrity checks for 31 paths.
- [x] Implement actual ZIP SHA-256, byte-count and header verification mode.
- [x] Add GitHub Actions lightweight and opt-in full-data workflow.
- [x] Align minimum Python runtime to 3.11 and pin light CI dependency.
- [x] Support saved checkpoints for new RF and Deep training runs.
- [x] Pass local 30/30 synthetic/code regressions.
- [ ] Confirm an actual GitHub Actions metadata job has succeeded.
- [ ] Perform a clean GitHub clone from an independent host.
- [ ] Download the complete v3.2 LFS object and verify its payload and ZIP.
- [ ] Complete full-data CI run (including all code tests) and archive logs.
- [ ] Generate a validated full scientific dependency lock / CUDA environment.
- [ ] Re-run the full B2 scientific experiments and demonstrate reproducibility.

**B4 engineering controls are implemented, but end-to-end remote/environment
reproducibility sign-off is OPEN until the unchecked gates are satisfied.**
