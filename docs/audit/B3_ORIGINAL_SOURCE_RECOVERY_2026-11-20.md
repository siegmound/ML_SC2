# B3 — Original replay/build source recovery, 2026-11-20

**Branch:** `audit/sc2-corrections-2026-11-17`  
**Status:** recovery evidence preserved, original full raw corpus and historical
exclusion trace **NOT YET RECOVERED**. The B3 release gate remains false.

## 1. What the original archives actually prove

The original uploaded freeze archive `sc2_project_freeze_final.zip`
(SHA-256 `8d04d3175a2983ac93f502e960956e06efa2de19087f0129fc925088d7b9fabe`)
contains `paper/sc2_ml_paper_final.tex` (SHA-256
`5828574bdf249db9f986c2de1e0d418ad9ab33c872a887975e909d209c628479`).

Under *Data Acquisition and Dataset Construction / Replay source and scale*,
that paper reports **24,057 competitive or high-level replay files** and
**approximately 892,048 snapshots**. These are *historical assertions in the
paper*, **not an exported original raw source filename inventory**.

The SHA-256-verified frozen v3.1 ZIP contains **exactly 892,047 rows and
23,241 unique replay IDs**. The one-row difference between the historical
*approximate* snapshot claim and exact frozen payload is not itself proof of
data loss. The replay-count arithmetic discrepancy **24,057−23,241=816**
remains a hypothesis about missing/excluded inputs, not evidence that
exactly 816 independently identified raw replays were parsed and rejected.

The historical frozen `scripts/buildf_v3_counter120.py`, SHA-256
`345f38e84557c5570f1498c7181ea141375581f689663d961d8aa2d9ca21fc32`,
specifies: flat `replays/` directory, *case-sensitive* `.SC2Replay`
suffix, 180–1,800-second length filter, and snapshot time `>120`.
It prints `Avvio estrazione massiva su {len(files)} replay...` and
`Snapshot totali: ...`; errors are **counted in aggregate** and
`None` results for numerous distinct filters are not logged per replay.
The archived copy has malformed print string literals, so **its byte-exact
copy is not attested as the actual executed builder**.

**Neither** of the two uploaded original freeze ZIPs contains a directly
listed original `.SC2Replay` file or a per-replay original exclusion
manifest. The subsequent `ml_sc2_remake_final_delivery.zip` contains
`src/sc2proj/builder.py` which *does* track `included_replays` and
`excluded_replays`, but that is **a later remake implementation**, not
an audit log from the historical frozen dataset build. Its source ZIP
SHA-256 is `cbc0a1548c581f4c070ff748dbeb5891caa914e66db07ac94d2789b51c881e31`.

### Nested archive recovery verification

A further read-only extraction-index inspection examined **all 26
embedded ZIPs** in the two historic freeze packages and the two
remake delivery/block16 packages (4 + 4 + 9 + 9). All nested ZIP
directories were readable. **None contains a `.SC2Replay` member**:
they contain prediction, calibration, model and training-evidence
artifacts, not original raw replay binaries. This closes the
*uploaded-archive search* for raw replay payloads, **not** the
possibility that the original replay input folder survives elsewhere
on FAIR or another workstation.

## 2. Repository history actually recovered

The genuine source branch `v1.2`, immutable Git commit
`bd64a32b305cc7203d592c7fc9015a01f962d6f9`, retains the parser,
bridge, complete `UNIT_STATS`, and `UNIT_COUNTERS` at `scripts/`.
A paired raw-to-frozen replay comparison of 3 original replay bytes
demonstrated **259 technology-value mismatches resolved**, leaving
**1,373 combat/counter and downstream differences**. Thus v1.2 is
a **real independently authenticated source revision**, **not proof
of an original executed generator**.

The later Git commit
`e325a95f250e613e47f74159fe3a30bf87ec6371`
(2026-04-07) **removed** historical `scripts/sc2_parser_v3_counter120.py`,
`scripts/sc2_constants.py`, `scripts/sc2_counters.py` and the bridge
from that tree. It did *not* provide a successor at those paths.
Branches `main`, `v2.0`, `feature/sc2-sequential-models` and
`audit/sc2-corrections-2026-11-17` contain a later **clean-room remake
builder**, not authenticated execution logs of the original build.

## 3. Upstream source recovery without misattribution

The tracked `data/raw/replaypack_inventory.csv` documents
**71 tournament ZIP names** on upstream **SC2ReSet v2.0.0**,
Zenodo record `14963356`, DOI
`https://doi.org/10.5281/zenodo.14963356`. The original
uploaded archives have *no direct raw replay files*; the Git repository
separately contains a real `data/raw/replay_subsets/` folder with
300 entries. On FAIR, **291** of those 300 replay filenames overlap
frozen v3.1; **4** actual MPQ replays have durations outside the
historic 180–1,800-second band and **5** are AppleDouble macOS metadata,
not game replays. This is evidence about the subset **only**.

The related **SC2EGSet** publication explains that *its own* tournament
archives include extraction `main_log.log` and JSON
`processedFiles/failedToProcess` logs:
`https://pmc.ncbi.nlm.nih.gov/articles/PMC10491788/`.
Those potentially useful upstream metadata files are **not**
the local project builder's original 24,057→23,241 execution log.
Do not use another extractor's failure count as the 816 missing replays.

Downloading 71 complete tournament packs merely to fit a target count
would be unjustified: first recover the exact local input selection,
original per-file log (if one survived), and original generator revision.

## 4. New FA​​IR read-only discovery scanner

New `scripts/86_discover_b3_generation_sources.py` recursively
searches caller-chosen *local* folders for:

- loose `.SC2Replay` files (classifies MPQ vs AppleDouble magic,
  counts by directory, skips symlinks and hardlink duplicates);
- ZIP archive directory entries (without payload extraction),
  catalogued 71-pack membership, duplicate replay basenames and nested
  ZIP references;
- candidate historical `buildf`, parser, bridge, constants and
  counter scripts, replay selection and exclusion manifests;
- log/document witnesses for 24,057; 23,241; 892,048; 892,047;
  `Avvio estrazione massiva su`; `Snapshot totali:`;
  `File saltati per errore:` and common exclusion log names.

It skips Git internals/virtualenv paths, hashes small candidate source
text files, has file/ZIP scan limits, and creates a **new report
directory only** after scanning. It neither executes historical
code nor interprets any discovered log as authoritative without
further source/corpus identification.

### Run on FAIR first, without downloading anything

```bash
cd ~/HPC-POMDP/ML-SC2/ML_SC2_tabular_audit_clean
git pull --ff-only origin audit/sc2-corrections-2026-11-17
export SC2_PYTHON="$HOME/.venvs/ml-sc2/bin/python"

"$SC2_PYTHON" -m unittest discover -s tests \
  -p 'test_b3_generation_source_discovery.py' -v

"$SC2_PYTHON" scripts/86_discover_b3_generation_sources.py \
  --roots "$HOME/HPC-POMDP" "$HOME/Downloads" \
  --pack-catalog data/raw/replaypack_inventory.csv \
  --output-dir results/b3_historical_build_discovery_20261120

cat results/b3_historical_build_discovery_20261120/summary.json
```

The `Downloads` tree is optional: a nonexistent root is recorded as
missing without failing. If there is another locally mounted source
tree, run a **second** scan, for example
`--roots /scratch /mnt/data`, with a **distinct** output directory.

Review the four reports:
`raw_files.csv`, `zip_archives.csv`, `source_candidates.csv`,
`text_witnesses.csv`.

If the report locates a **complete actual** original `replays/`
folder or original replaypack ZIP directory, then execute existing
`scripts/77_inventory_b3_raw_sources.py` pointing
`--raw-root` to that exact folder, with a new output directory.
This compares all observed raw replay basenames to the frozen
23,241 IDs. Only if a proven complete original 24,057-file source
inventory is identified can the original gap be enumerated with
confidence. Reprocessing new downloaded ZIPs is **not** a
substitute for evidence of historical file selection.

## 5. What is and is not recovered

**Now evidenced:**
source title/DOI and 71 pack names; 300 subset raw filenames;
two historical parser revisions with pinned hashes;
original published count assertion; original builder filter rules;
exact frozen replay IDs/count and all known numerical audit results.

**Still missing:**
a cryptographically attributable list of the 24,057 original source
files; per-replay original exclusions with cause; the actual executed
counter/combat revision and pinned environment; original v3.2
combatfix raw-replay primitives.

Do **not** merge/release, update B3 gate, retune models or edit
frozen data based on this discovery stage. B5 remains
`BLOCKED_PENDING_EVIDENCE`.

## 6. FAIR incident: scan cap at 500,000 files (2026-11-20)

A real FAIR scan was attempted with `--roots "$HOME/HPC-POMDP" "$HOME/Downloads"`.
All three original regression tests passed, but the complete filesystem scan
**aborted before any output was committed**, raising:

```text
ValueError: File scan cap reached; narrow --roots
```

This is a search-space/tooling limit, not evidence of corrupt replay files,
incomplete snapshot data, or actual historical exclusions.

The updated `scripts/86_discover_b3_generation_sources.py` now supports
`--on-limit partial` (the CLI default), preserving already located evidence
while recording `scan_complete=false`,
`scan_status=PARTIAL_INCOMPLETE_LIMIT_REACHED`, `limit_reached`,
`interrupted_at_path` and unstarted roots. Crucially,
`catalogued_zenodo_packs_missing=null` in incomplete scans, so no false
claim of exhaustive absence can be made. Direct Python calls preserve the
original fail-closed `on_limit="error"` behavior.
Common non-source dependencies/caches (including `site-packages` and
`.cache`) are pruned. Two additional regressions verify partial output
and no false missing-pack claims.

**Run targeted first**, with a *new* output folder:

```bash
cd ~/HPC-POMDP/ML-SC2/ML_SC2_tabular_audit_clean
git pull --ff-only origin audit/sc2-corrections-2026-11-17
export SC2_PYTHON="$HOME/.venvs/ml-sc2/bin/python"
"$SC2_PYTHON" -m unittest discover -s tests -p 'test_b3_generation_source_discovery.py' -v

"$SC2_PYTHON" scripts/86_discover_b3_generation_sources.py \
  --roots "$HOME/HPC-POMDP/ML-SC2" "$HOME/Downloads" \
  --pack-catalog data/raw/replaypack_inventory.csv \
  --max-files 100000 \
  --on-limit partial \
  --output-dir results/b3_historical_build_discovery_targeted_20261120

cat results/b3_historical_build_discovery_targeted_20261120/summary.json
head -n 35 results/b3_historical_build_discovery_targeted_20261120/source_candidates.csv
head -n 35 results/b3_historical_build_discovery_targeted_20261120/text_witnesses.csv
```

If this also reaches its limit, **do not treat absence of an original source
as proved**. The partial report exposes exactly where inspection stopped.
Further scans should use distinct roots and different report folders, e.g.
`--roots "$HOME/HPC-POMDP/ML-SC2/ML_SC2_tabular_audit_clean/data/raw"`
or historical project checkout directories found via
`find "$HOME/HPC-POMDP" -maxdepth 4 -type d -name replays`,
without blindly raising the cap over very large unrelated workspaces.

Original 24,057 input file identity and 816-per-file exclusion provenance
remain **UNVERIFIED**; B3 and release gates stay blocked.


## 7. Actual targeted FAIR discovery result — 2026-11-20

The operator pulled the fixed scanner and ran **5/5 source-recovery
regression tests PASS**, followed by the actual constrained discovery:

- Root requested: \`/home/jlpfritas/HPC-POMDP/ML-SC2\`;
  second requested root \`/home/jlpfritas/Downloads\` did not exist.
- \`scan_complete=true\`, \`scan_status=COMPLETE_ALL_REQUESTED_ROOTS\`,
  \`n_visited_files=2,374\`. **This is complete for the one existing
  requested directory tree only; it is NOT a FAIR-wide search.**
- **600** loose \`.SC2Replay\` extension files in precisely two
  source directories:
  \`ML_SC2/data/raw/replay_subsets\` (300) and
  \`ML_SC2_tabular_audit_clean/data/raw/replay_subsets\` (300).
  Each directory has **295 MPQ-candidate headers and 5 AppleDouble
  macOS metadata headers**. This strongly suggests two checkout
  copies of the *same* 300-entry subset; byte identity between
  the two directories is **not yet SHA-256 verified**.
- **19 ZIPs** inspected, none matching the 71 catalogued original
  Zenodo replaypack names, and **zero direct \`.SC2Replay\` ZIP
  entries** in those inspected ZIPs.
- **14** possible source/manifest text files, with paths present in
  the two current worktrees: no newly discovered original executable
  parser/counter script, original executed builder, or original
  per-replay rejection log in the displayed candidates.
- **260** text witness matches (including repeated citations of
  24,057 / 23,241 / 892,047 in project audit documents and
  manifests), but these are *documentary repetitions*, not 260
  independent original build-log discoveries. A candidate would need
  a distinct original log path, dated record or verified provenance
  to count as newly found source evidence.

**Interpretation:** The search under both current ML-SC2 checkouts
did **not** find the historical 24,057-input raw corpus or 71
original replaypack ZIPs, and did not recover an original per-file
exclusion record. No *global* absence claim is justified; other
HPC-POMDP sibling directories, scratch volumes, archives elsewhere
on FAIR or the workstation that originally built the dataset have
not yet been exhaustively searched.

### Next non-destructive actions

1. Independently hash the same-named raw files in the two 300-entry
   \`data/raw/replay_subsets/\` directories to establish whether
   there are exactly 300 distinct source bytes, rather than
   assuming the 600 paths are different replay inputs.
2. Use a *directory-only*, bounded-depth search outside the two
   already-scanned clones to identify old \`replays/\`,
   \`SC2ReSet/\`, \`starcraft*\` source directories and other
   data mount points. Avoid full file traversal of HPC-POMDP
   without directory pruning.
3. Scan any newly identified likely source directory using
   script 86 with a **distinct output directory**. If it
   contains replaypack ZIPs, compare their central directory
   per-file names with the preserved frozen v3.1 IDs before
   asserting exclusion reasons.
4. If no old local original source survives, preserve this
   negative scoped-search evidence and retrieve upstream
   replaypack metadata selectively; do not download 71 large
   packs solely to force a total of 24,057.

The original count discrepancy \`24,057−23,241=816\` remains
**unattributed**. The B3 and B5 gates remain closed.
