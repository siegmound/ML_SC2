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
- [x] Confirm an actual GitHub Actions metadata job has succeeded.
- [x] Perform a clean GitHub clone from an independent host.
- [x] Download the complete v3.2 LFS object and verify its payload and ZIP.
- [ ] Complete full-data CI run (including all code tests) and archive logs.
- [ ] Generate a validated full scientific dependency lock / CUDA environment.
- [ ] Re-run the full B2 scientific experiments and demonstrate reproducibility.

**B4 engineering controls are implemented, but end-to-end remote/environment
reproducibility sign-off is OPEN until the unchecked gates are satisfied.**


## Final B4 remote change verification (2026-11-17)

- After all code and documentation changes, the GitHub comparison against
  immutable B0 SHA `33660e2b4fc4bd75df0824bbb7fa7c2096d13831`
  reports **`main` identical** (ahead=0; behind=0).
- The correction branch was **72 commits ahead of main**, changing
  **45 paths** across B0–B4, before this final documentation commit.
- GitHub file readbacks matched the locally executed source bytes for:
  `scripts/70_verify_freeze.py`,
  `tests/test_b4_freeze.py`,
  `tests/test_b4_checkpoints.py`,
  `scripts/54_run_rf_full.py` and
  `scripts/64_run_deep_finalist.py`.
- **No protected frozen dataset ZIP, RF/XGB/Deep run result,
  calibration result, PDF, or PPTX was in the branch diff.**
- Local combined regression suite **30 tests PASS**.
- The GitHub workflow definition was confirmed committed, **but no successful
  remote Actions execution is asserted**. Full LFS payload validation and
  exact scientific dependency lock remain OPEN.
- This final sign-off is a documentation-only commit.

**B4 status: source-level engineering complete; independent
clone/CI/full-data reproducibility NOT YET CERTIFIED.**


## Post-audit CI evidence update — 2026-11-18

GitHub PR #5 (`audit/sc2-corrections-2026-11-17` → `main`), at head
`397dc3c12a7ae006168a46037881e902cde1c92d`, triggered workflow run
`44393744998` (`B4 reproducibility checks`).

**Verified via GitHub Actions API:**
- `Frozen artifacts, schemas and source checks` (job `135117997597`): **completed / success**.
- All steps in that lightweight job succeeded, including checkout, Python 3.11,
  pinned metadata verifier dependency, baseline Git blobs and LFS pointers,
  repository/summary/feature-lineage checks, Python syntax, stdlib-only
  regression tests and audit-report preservation.
- `Opt-in full LFS verification and CPU integration tests` (job
  `135118000460`): **completed / skipped**.
- Workflow result: **completed / success**. This establishes the narrow CI
  metadata check, **not** full-data payload, scientific environment, or
  training reproducibility.

Reference: https://github.com/siegmound/ML_SC2/actions/runs/44393744998

The separate B4 clean-clone, v3.2 payload verification, full integration suite,
dependency/CUDA lock and B2 full-scale experiment conditions remain **OPEN**.
The B5 release manifest stays `BLOCKED_PENDING_EVIDENCE`. This evidence
update does not approve scientific release or a merge into `main`.


## Independent FAIR checkout evidence — 2026-11-18

The user performed a **fresh GitHub clone** on FAIR into the separate workspace
`~/HPC-POMDP/ML-SC2/ML_SC2_tabular_audit_clean`, from
`audit/sc2-corrections-2026-11-17`, using `GIT_LFS_SKIP_SMUDGE=1`.
The checkout completed (1,718 Git objects, 979 tracked working tree files).
It is a fresh checkout independent of the previous working tree, **but uses
the user's existing shared `~/.venvs/ml-sc2` virtual environment**, not a
newly locked or independently installed Python dependency environment.

**Observed PASS on the fresh checkout:**
- `python scripts/70_verify_freeze.py --mode pointer`: **PASS**, 31
  immutable Git objects and both Git LFS pointer metadata.
- `python scripts/58_verify_final_table.py --check`: **PASS**.
- `python scripts/66_check_matched_train_budget.py --check`: **PASS**
  on matching historical train replay/label metrics.
- `python -m unittest discover -s tests -p 'test_*.py' -q` with
  `PYTHONPATH="$PWD/src"`: **34 tests OK, 1 skipped**.

**Still OPEN:** FAIR reported `git: 'lfs' is not a git command`, so
`git lfs install --local` and `git lfs pull` could not execute.
The subsequent `python scripts/70_verify_freeze.py --mode payload`
correctly **FAILED at the v3.1 dataset**, because the checkout was intentionally
pointer-only. This is not evidence of payload corruption. Neither
v3.1 nor v3.2 full payload integrity has yet passed in this fresh clone.
Install or locate Git LFS, fetch both exact LFS objects, rerun payload audit,
and capture an independent Python/CUDA dependency lock before closing the
scientific reproducibility gate.

This update does **not** clear B5 `BLOCKED_PENDING_EVIDENCE`, assert
full B2 training reproduction, or authorize a formal release.

## Verified full LFS dataset payloads — FAIR clean checkout, 2026-11-18

The user located Git LFS **3.8.1** at
`$HOME/HPC-POMDP/.tools/git-lfs/bin/git-lfs`, activated it in PATH,
ran `git lfs install --local` inside the separate clean checkout, then
successfully fetched **both** tracked ZIP objects with `git lfs pull`
(2/2 objects, ~563 MB).

**Observed actual execution** of
`python scripts/70_verify_freeze.py --mode payload`:
`PASS: B0 protected Git blobs and LFS payload integrity`.
The script checks each file's exact byte count and SHA-256, verifies a
readable ZIP archive and checks ordered feature headers against the saved
full XGBoost feature schema, in addition to the 31 original Git objects.

| Dataset | Expected and observed bytes | Observed SHA-256 | Feature count | ZIP header |
|---|---:|---|---:|---|
| v3.1 fixed | 268,539,879 | `65e402884f43c03bf5450d6975e91db3d61ee7e28edf1a56483a37f15d944517` | 64 | PASS |
| v3.2 combatfix | 294,197,452 | `87ce36435e4a0fc87b8d9421291d3b1fb04d67b5187ede6a2a3b25d0dce1942e` | 73 | PASS |

**This closes the previous v3.2 incomplete local ZIP uncertainty** for
the new FAIR Git LFS checkout. Earlier notes about a truncated separate
upload are retained as historical context and no longer describe the
newly verified LFS artifact. Neither frozen dataset was modified.

Remaining B4: independent scientific Python/CUDA dependency inventory
and tested lock, full integration CI (the previous optional full LFS job
was skipped), and full-scale B2 reproducibility. This is **not** a
replay-by-replay B3 parser/feature-lineage proof. B5 remains blocked.


## FAIR scientific runtime inventory + strict source checks — 2026-11-18

After the clean FAIR clone's full LFS payload audit passed, the user ran
`git pull --ff-only` to commit
`21db9001583385e4ef117ecbb627f8adfd78a1e2` and collected a
read-only runtime inventory in their private FAIR workspace
`~/HPC-POMDP/ML-SC2/b4_environment_evidence/`. The inventory files
include `git_commit.txt`, `python_version.txt`, `pip_freeze.txt`,
`pip_check.txt`, `nvidia_smi.txt` and `ml_runtime.txt`.
Only the following command output has been supplied for review; the full
pip freeze and driver inventory have not yet been independently inspected.

| Component | Observed FAIR runtime |
|---|---|
| Python | 3.12.3 |
| PyTorch | 2.15.0+cu132 |
| PyTorch CUDA | 13.2 |
| CUDA available | True |
| GPU | NVIDIA GeForce RTX 5090 |
| scikit-learn | 1.9.1 |
| XGBoost | 3.4.1 |
| NumPy | 2.5.5 |
| pandas | 3.1.1 |
| `python -m pip check` | `No broken requirements found.` |

The user also invoked `python scripts/15_verify_reproducibility.py --strict`
and `python scripts/17_internal_audit.py` on the clean checkout; both
commands returned to the terminal with no visible failure message. These
tools are silent on ordinary success and save machine-readable reports to
`results/logs/reproducibility_check.json` and
`results/logs/internal_audit_report.json`. The reports and explicit exit
codes have **not yet been supplied**; do not infer zero unreported style
findings. Note that `17_internal_audit.py` without `--strict-style`
only fails for syntax errors; style warnings do not trigger failure.

GitHub Actions run
`44397656659`, from the audited branch head
`21db9001583385e4ef117ecbb627f8adfd78a1e2`, completed
**successfully** for lightweight source/metadata checks (job
`135131785094`); its full LFS+CPU integration job
`135131787784` was **skipped**.
https://github.com/siegmound/ML_SC2/actions/runs/44397656659

**B4 narrow engineering check:** GitHub clone on FAIR, full local
replay-disjoint frozen dataset/payload checks, tabular integration suite
(34 tests, 1 skipped) and remote lightweight metadata CI have all passed.
This establishes the clean-checkout + light remote CI gate, distinct from
the remaining full-data remote CI job and scientific environment lock.

**Still open:** capture the actual `pip freeze` + GPU driver versions and
archive a reproducible dependency lock; establish a new isolated environment
from that lock and rerun applicable checks; run and archive the full-data CI
job; complete B2 full-size scientific experiments; recover B3 end-to-end
raw replay-to-feature provenance. Existing retrospective test data cannot
be reclassified as an uninspected holdout.


## FAIR generated audit reports reviewed — 2026-11-18

The user read both generated JSON reports from the **clean tabular checkout**,
after completing `python scripts/15_verify_reproducibility.py --strict`
and `python scripts/17_internal_audit.py`:

`results/logs/reproducibility_check.json`
- `missing_paths: []`: **PASS**
- All six categories under `dependency_mismatches` were empty: **PASS**
- `final_table_check.passed: true`: **PASS**
- `feature_lineage_check.passed: true`: **PASS** for *schema/manifest consistency only*; this is **not** raw replay-to-feature numerical parity.
- `publication_check.passed: true`: **PASS** for editable paper/table source consistency; this does **not** certify generated PDF/PPTX.

`results/logs/internal_audit_report.json`
- `syntax_errors: 0`: **PASS**
- `marker_hits: 0`: **PASS**
- `compressed_files: 9`: **style findings, unclassified** until per-path inspection.
- `long_line_issues: 75`: **style findings, unclassified** until per-path inspection.

The internal audit was invoked without `--strict-style` and therefore **does
not certify zero style warnings**; its unclassified findings have no effect on
the explicit B4 scientific environment lock or B5 publication gates. Avoid
formatting frozen datasets, archived metrics, or unrelated generated evidence
merely to suppress style counts.

All artifacts are preserved in the user's local FAIR checkout; only summarized
terminal diagnostics were provided for external review. No new model or dataset
runs were performed. Next remaining B4 tasks are an independently recreated
and checked Python/CUDA runtime from an archived, sanitized lock and a
successfully executed opt-in full-data CI run. Existing source correction
and payload-integrity checks are already passed.


## Candidate FAIR dependency lock inventory — 2026-11-18

The user inspected the independently captured
`~/HPC-POMDP/ML-SC2/b4_environment_evidence/pip_freeze.txt` on FAIR:
**56 noncomment entries**. The inspection detected **one editable
(`-e`) reference**, whose source is not printed in the terminal excerpt.
The following installed distributions were observed with exact pins:

| Dependency | Observed pip freeze |
|---|---|
| numpy | `numpy==2.5.5` |
| pandas | `pandas==3.1.1` |
| pyarrow | `pyarrow==26.0.1` |
| sc2reader | `sc2reader==1.9.0` |
| scikit-learn | `scikit-learn==1.9.1` |
| scipy | `scipy==1.18.1` |
| torch | `torch==2.15.0` |
| xgboost | `xgboost==3.4.1` |

A previously observed `torch.__version__` reported
`2.15.0+cu132` and `torch.version.cuda` `13.2`;
`pip freeze` by itself does **not** uniquely certify the CUDA-enabled
wheel binary, its package-index origin, or driver/OS compatibility.

**Next required checks:** classify the editable reference as this
tabular repo or an external dependency *without publishing its URL*;
produce a sanitized candidate pin list excluding only confirmed local
editable installation; record full wheel and driver provenance; create
an **independent** isolated Python 3.12 virtual environment and install
from the resolved lock, installing the current clean checkout itself as
an editable project; verify `pip check`, CUDA stack, full 34-test tabular
suite, strict schema/table/source audits and frozen LFS hashes.
No independent environment reconstruction has yet been observed, and
the scientific runtime-lock condition remains **OPEN**.

This inventory is **audit evidence**, not a change to the project's
minima in `requirements.txt` or frozen scientific results.


## FAIR editable-origin classification and sanitized pin candidate — 2026-11-18

The user independently classified the single `pip freeze` editable
entry using the installed Python metadata and the originating source:
- Project distribution: `ml-sc2-remake`.
- Validated Git remote: `github.com/siegmound/ML_SC2` (no third-party editable).
- Candidate lock: `55` exact `name==version` entries; local file
  `~/HPC-POMDP/ML-SC2/b4_environment_evidence/requirements_fair_candidate.txt`.
- User-reported SHA-256 of that candidate text:
  `9dc23f0ea5936a158e8a5e271494768a2ce2163ed351c0e430a5bcfa1882787e`.
- No raw editable URL or possible credentials were included in this report.

The candidate excludes the editable GitHub source so that a separate
isolated environment can install the **checked-out tabular audit branch**
using `pip install -e . --no-deps` (avoiding accidental substitution with
the sequential branch). The SHA-256 certifies the text, *not* wheel-byte
reproducibility or the source Git SHA of the editable installation.
Original FAIR torch runtime was `2.15.0+cu132`; the freeze pin
`torch==2.15.0` alone is insufficient to assert the CUDA wheel variant.

**Open:** resolve and install in a new Python 3.12 venv, verify the actual
wheel/CUDA variant and 34-test suite, capture provenance/hashes, and
re-run the deterministic validations. Do not set the full scientific
environment lock or B5 release conditions until independent reproduction
evidence is supplied.


## Isolated Python/CUDA reproduction — FAIR evidence, 2026-11-18

The user executed a new, separately created Python **3.12.3** virtual
environment on FAIR (distinct from the original
`$HOME/.venvs/ml-sc2`) using the already checked SHA-256 candidate
`requirements_fair_candidate.txt`
(`9dc23f0ea5936a158e8a5e271494768a2ce2163ed351c0e430a5bcfa1882787e`).

**Observed in the supplied FAIR terminal log:**
- Available home disk: ~15 TB; Python 3.12.3 in the new environment.
- `pip install --dry-run -r`: resolved all 55 pinned distributions
  from package indexes/cached wheel files without an error.
- `pip install -r`: **SUCCESS**, 55 installed distributions including
  PyTorch, CUDA/NVIDIA components, XGBoost and scikit-learn.
- `pip install --no-deps -e .` from the **tabular audit clean checkout**:
  **SUCCESS**, installed `ml-sc2-remake==2.0.0`.
- `python -m pip check`: `No broken requirements found.`
- Runtime: PyTorch `2.15.0+cu132`, torch CUDA `13.2`,
  `torch.cuda.is_available() == True`, RTX 5090;
  scikit-learn `1.9.1`, XGBoost `3.4.1`, NumPy `2.5.5`,
  pandas `3.1.1`.
- Actual CUDA tensor operation: `CUDA_TENSOR_TEST_PASS`.
- Frozen artifact and both complete dataset payload validations:
  `PASS: B0 protected Git blobs and LFS payload integrity`,
  64 and 73 expected predictor fields and matching checksums.

**One reproducibility-suite regression remains for this observed run:**
The 34-test suite reported **one failure, one skipped**, solely
`test_b5_publication.PublicationChecks.test_current_release_is_blocked_by_explicit_pending_evidence`.
The test was written before verified B4 metadata CI and clone evidence
closed `b4_clean_clone_and_remote_ci_passed`. It incorrectly asserted
this specific release-gate flag must remain `false`, while the
**release approval remained explicitly BLOCKED with other flags false**.
No training failure, GPU failure, or source integrity failure was observed.

A targeted fix was committed to
`tests/test_b5_publication.py` at `dac91f05586d683a448eec05150d1414bb0c555e`,
so tests now require default-deny logic, unresolved flags, and explicit
approval rather than hardcoding a newly verified gate. Two extra
negative-case tests verify that premature `APPROVED` cannot bypass a
false condition and all-true flags cannot bypass an unapproved status.
**The post-fix tests have not yet been reported from FAIR.**

**Remaining to close the full scientific runtime-lock condition:**
a successful independent full regression rerun with the corrected test,
archiving reproducible artifact provenance (wheel hashes or equivalent,
platform/Python/GPU driver and index information), and completing the
opt-in full-data GitHub Actions workflow. Observed pip installs used
cached wheels and version pins; this establishes environment reconstruction
on FAIR, **not bitwise-identical wheel provenance or cross-host portability**.
No historical models were retrained or original results overwritten.


## Verified independent full regression rerun after B5 test fix — 2026-11-19

The user repeated verification on FAIR in the **new independent** Python
3.12.3 venv `$HOME/.venvs/ml-sc2-b4-repro` after pulling correction-branch
head `64047137b9ed9ec4deed6f0dfd706338c14a2a68`:

- `python -m pip check`: **PASS** (`No broken requirements found.`).
- `python -m unittest discover -s tests -p 'test_*.py' -q`:
  **36 tests run; OK; skipped=1** (35 passing, zero failing).
  Two new B5 default-deny release regression tests passed. The Pandas
  `DataFrame.dropna(inplace=True)` deprecation warnings are nonfatal and
  should be handled as maintenance rather than evidence of scientific
  failure.
- `python scripts/15_verify_reproducibility.py --strict`: no visible
  failure; full JSON report was separately inspected in the prior FAIR
  checkout audit.
- `python scripts/72_verify_publication.py`: **PASS**, `paper_rows=4`,
  `paper_b5_caveats=PASS`; this is source consistency only, **not**
  generated PDF/PPTX visual certification.
- `python scripts/72_verify_publication.py --release`: intentionally
  returned **`FAIL: RELEASE BLOCKED: 8 unfinished gates`**; the new
  tests therefore preserve default-deny publication gating.

**Outcome:** isolated FAIR Python/CUDA environment reproduction and the
corrected tabular test suite now **PASS**. All protected LFS/data checks
were already independently verified. The former one failing test
was a stale assertion about a completed B4 subgate, not a model failure.

**Still open for full B4 release condition:** audited wheel-byte hashes
and reproducible installation provenance, host GPU-driver/CUDA details,
and a successful GitHub Actions **full-data** run; B2's original
full-scale four-way/deep experiments are separately unresolved.
The official B5 release status remains `BLOCKED_PENDING_EVIDENCE`.


## Full-data GitHub CI PASS and mpyq source-only wheel exception — 2026-11-19

GitHub Actions workflow run
`44420825024` on `audit/sc2-corrections-2026-11-17`,
https://github.com/siegmound/ML_SC2/actions/runs/44420825024,
has now **completed / success**. Both jobs succeeded:

- Metadata source/consistency job `135212988748`: SUCCESS.
- Full-data Git LFS + CPU integration job `135212988437`: SUCCESS.
  GitHub Actions reported success for all substantive steps:
  Python 3.11 install, dependency installation, both full LFS downloads,
  SHA-256 payload and feature-header verification, schema/replay census,
  complete synthetic integration tests, matched-fit retrospective checks,
  and runtime evidence artifact upload.

This closes the former **skipped opt-in full-data CI** blocker; it does
not create or retroactively validate new held-out scientific experiments.

The user separately attempted wheel provenance capture on FAIR from a
55-exact-pin candidate. The command
`pip download --only-binary=:all: --no-deps` **failed** on
`mpyq==0.2.5` because PyPI provides only
`mpyq-0.2.5.tar.gz` as source distribution (released 2014-01-05),
not a compatible prebuilt wheel; therefore no wheels were written
and no hashed lock manifest was produced by that attempt.
This is **not** a CUDA, model, installation or checksum failure.

Corrective evidence procedure: download wheel distributions for the
other 54 pins; independently retrieve and SHA-256 hash the original
`mpyq` source distribution; build its local wheel from that exact
source and record the resulting SHA-256 and build metadata. Then
write the 55-entry platform-specific hashed requirements file and
perform a `--no-index --require-hashes` verification from a local
wheelhouse. Do not rewrite the original candidate lock, silently
replace the source version, or represent a locally built wheel
as publisher-supplied.

B5's scientific environment lock remains **OPEN** pending verified wheel
provenance, final reproducibility evidence, and sign-off. The source
replay provenance (B3), full-scale experiments (B2), and PDF/PPTX release
gates remain independently OPEN.
