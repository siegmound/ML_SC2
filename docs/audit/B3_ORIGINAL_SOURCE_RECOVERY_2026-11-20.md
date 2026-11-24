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


## 8. Actual SHA-256 mirror comparison and sibling-directory discovery — 2026-11-20

The FAIR operator independently hashed each same-basename file in the
two \`data/raw/replay_subsets/\` checkout directories:

- First checkout: 300 \`.SC2Replay\`-extension files.
- Audit checkout: 300 \`.SC2Replay\`-extension files.
- Basenames common to both: **300/300**.
- Basenames exclusive to either directory: **0/0**.
- Same-name **SHA-256 content differences: 0/300**.

Thus the earlier 600 filesystem paths are **two byte-identical
mirrors of exactly 300 named source entries**, not 600 independent
SC2 replay inputs. Their composition remains **295 MPQ-container
candidates and 5 AppleDouble metadata files**. This direct mirror
check is a real FAIR execution; it does not depend on a conjecture
of repository history.

A directory-name search of \`$HOME/HPC-POMDP\` up to depth 7,
pruning Git/venv/cache/dependency directories, found the two
already known SC2 replay subsets. Other path hits containing
\`replay\` predominantly belong to **unrelated** Arcana world
simulation and HPC-POMDP/OCTO replay/stress test workflows; their
names do not identify StarCraft II game replay input files.
One additional narrowly scoped historical evidence directory,
\`$HOME/HPC-POMDP/OCTO-Archivio/evidenze/research_records/2026-10/operator_runs/sc2_20261030_gUYOFXzZ\`,
contains the literal \`sc2\` identifier and warrants **metadata-only
inspection**, but **is not yet demonstrated** to hold original
SC2 game files, execution logs or source archives.

The directory search is bounded by name and depth and so cannot
establish that there are no raw SC2 replay binaries under
different directory names, on /scratch or a different workstation.
No complete 24,057-file source input manifest or original
816-case exclusion record has been found. B3 is **still OPEN**.

### Focused follow-up (FAIR; read-only)

\`\`\`bash
candidate="$HOME/HPC-POMDP/OCTO-Archivio/evidenze/research_records/2026-10/operator_runs/sc2_20261030_gUYOFXzZ"
if test -d "$candidate"; then
  find "$candidate" -maxdepth 3 -type f \
    -printf '%p\t%s bytes\n' | head -n 60
fi
\`\`\`

Review filename, purpose, timestamp and exact evidence before
attributing this directory to the original ML-SC2 dataset
generation. If its records are unrelated to the ML data builder,
do not include them as raw replay evidence. Next prioritize
original Windows/local workstation/archive backups and
a directory-only search of any designated FAIR scratch volumes.


## 9. OCTO archive named sc2: unrelated code-execution evidence

The FAIR operator listed files (depth <=3) in the narrowly scoped
`$HOME/HPC-POMDP/OCTO-Archivio/evidenze/research_records/2026-10/operator_runs/sc2_20261030_gUYOFXzZ`.
They include CUDA/C++ sources `collector.cu`, `wrapper.cu`,
`helper.cpp`, Python runners `direct_ell.py` and `runner.py`,
per-run `gate-summary.json`, `invocation.json`, `runtime.json`,
`science-gate.json`, `junit.xml`, and an approximately 205 KB
`delivery.tar`, accompanied by `delivery.tar.sha256`.

The **listed regular files do not contain any `.SC2Replay`**
extensions, original SC2ReSet pack ZIPs, named tournament source inputs,
or historical ML-SC2 exclusion manifests. The code/evidence naming
is strongly indicative of HPC-POMDP/OCTO program execution,
**not** a recovered StarCraft II replay corpus.
The internal `delivery.tar` members were *not* shown; do not claim
the archive was inspected or contains no SC2 files until its member
index is explicitly read (no extraction needed).

Read-only final check, if desired:

```bash
candidate="$HOME/HPC-POMDP/OCTO-Archivio/evidenze/research_records/2026-10/operator_runs/sc2_20261030_gUYOFXzZ"
tar -tf "$candidate/delivery.tar" | head -n 80
```

If the TAR contains only OCTO artifacts, stop searching this directory.
Do not misclassify the OCTO project use of the term "replay" as
StarCraft II game data. The original 24,057 input file identities,
per-file 816 exclusion reasons, and actual v3.1 counter/combat
generator remain **unresolved**. The release gate does not change.

## 9. Original 24,057-replay source archive recovered on FAIR — 2026-11-23

The operator uploaded the preserved original archive at
`/home/jlpfritas/HPC-POMDP/ML-SC2/replays.zip` and measured
SHA-256 `8dcb826aeb17481d592f2d564b8a3af72bcb2f11fba9421bdbd350641970241d`.
Archive size is reported as 8.5 GiB (roughly 8.8–9.1 decimal GB
depending on file/member accounting). FAIR has 15 TB of free /home storage.

The read-only central-directory census reports **24,129 files:**
**24,057 directly accessible `.SC2Replay` entries** under `replays/`,
**71 embedded tournament `.zip` files**, and **one JSON**.
All 24,057 replay basenames are distinct; none of the ZIP member
paths was identified as unsafe by the operator's path scan.
The declared total uncompressed member size is 9.11 GB.
These original replay bytes were not present in the smaller historic
freeze archives; they have now been recovered as a separate input asset.

**Major correction to previous recovery status:** the actual raw
corpus source count **24,057** is now physically evidenced rather
than merely asserted by the old paper. However, the arithmetic
gap 24,057 - 23,241 = 816 still has *unknown per-file causality*.
Do not assume all 23,241 frozen replay IDs are contained in the
original archive until their exact basenames have been compared.
The embedded JSON may contain useful original provenance, but its
schema is not yet inspected. The 71 embedded ZIPs are separate
packaged tournament sources and must not be counted again as new
unique replay inputs.

New read-only script 93, in the independent branch
`rebuild/sc2-v4-clean-2026-11-20`, is available for a SHA-pinned
raw-archive vs frozen-ID reconciliation. Script 94 can subsequently
CRC- and SHA-verify selective extraction of the 24,057 replay
members, leaving the original outer ZIP and nested ZIPs untouched.
These tools are not a repair of historical executed code, and
**historical B3 remains OPEN and B5 remains BLOCKED_PENDING_EVIDENCE**
pending actual identity reconciliation and raw-to-feature attribution.

## 10. Actual FAIR source-to-frozen identity reconciliation — 2026-11-23

The FAIR operator successfully ran `scripts/93_b3_reconcile_original_replays_zip.py`
against immutable source ZIP SHA-256
`8dcb826aeb17481d592f2d564b8a3af72bcb2f11fba9421bdbd350641970241d`
and frozen v3.1 ZIP SHA-256
`65e402884f43c03bf5450d6975e91db3d61ee7e28edf1a56483a37f15d944517`.
**4/4 script-93 synthetic tests PASS** on FAIR; an actual
read-only ZIP index vs frozen CSV replay-ID column reconciliation completed.

| Observed source | Value |
| --- | ---: |
| Original outer ZIP file entries | 24,129 |
| Direct raw `.SC2Replay` entries, all unique by basename | **24,057** |
| Nested tournament `.zip` entries (not recursed) | 71 |
| Other embedded `.json` entries | 1 |
| Declared uncompressed bytes of direct replay members | **4,568,567,819** |
| Frozen v3.1 snapshots | **892,047** |
| Unique frozen v3.1 replay IDs | **23,241** |
| Original replay basenames also present in frozen v3.1 | **23,241** |
| Frozen IDs absent from original outer ZIP | **0** |
| Raw basenames absent from frozen v3.1 | **816** |

The exact identity-level subset relation is now **proved for these two
SHA-pinned assets**: every frozen v3.1 replay basename is a member of
the recovered archive's 24,057 distinct replay basenames; the
816-name set difference is enumerated at
`results/b3_recovered_full_zip_census_20261123/raw_not_in_frozen.csv`
on FAIR. The outer ZIP also contains 71 tournament ZIPs but these
were **not** double-counted. The 4.57-GB uncompressed figure refers
specifically to directly accessible `.SC2Replay` members, not the
entire 9.11-GB nested-archive aggregate.

**Resolved:** source inventory identity and exact count discrepancy.
**Not resolved:** original execution-level exclusion reasons for
each of the 816, historical parser/bridge/combat-counter source
attestation or full raw-to-frozen feature parity. File presence
is not evidence that a particular historical filter rejected it;
no individual historical exclusion cause is attested. Preserve
the baseline source/ZIP SHA values and do not retroactively edit
historical datasets or relabel the original scientific results.

Next: inspect the single embedded JSON in bounded fashion, classify
the 816 separately by *current observed* MPQ/metadata and duration
evidence, and verify any proposed explanation using positive control
replays already in v3.1. Historic B3 overall remains OPEN, B5
remains `BLOCKED_PENDING_EVIDENCE`.

## 11. FAIR real replay metadata smoke: 30 unmatched vs 15 matched controls

Operator executed the real `scripts/95_b3_probe_raw_only_replays.py`
on the verified SHA-pinned 24,057-replay ZIP and v3.1 frozen data.
All 45 intended source members were inspected; the complete output is
`results/b3_raw_only_metadata_smoke30_20261123/` in the FAIR V4 worktree
(not committed). The output `per_replay_diagnostic.csv` SHA-256 is
`3b86ae84bd9e3c3f12ee3287d7e5d2f8c315c0210016c925ccf0a1afcb92892b`.
Three synthetic diagnostic tests PASS in the same worktree.

| Category, from CURRENT metadata checks | Count of 30 unmatched |
| --- | ---: |
| AppleDouble metadata (not SC2 game replay) | 13 |
| Duration >1800 s, otherwise no additional tested condition | 11 |
| Duration >1800 s AND not 1v1 | 1 |
| Duration <180 s | 3 |
| Not 1v1, without other tested condition | 1 |
| None of the tested conditions | **1** |
| Total | **30** |

Of 15 randomly selected *included-in-v3.1* positive-control replays,
**15/15 passed** these limited metadata checks. Thus 29/30 sampled
unmatched have at least one prospective reason for V4/current filtering,
but the observed sample does NOT prove that 29/30 of the complete
816 have similar statuses. Moreover these are not original execution
records or proof of original exclusion causality.

The surviving unmatched example
`843f17beaef8e0821010aa0738d71890.SC2Replay`
is a two-player game 1,111 seconds long with a single identifiable
winner and no current duration/player/winner flag. It requires additional
parser/feature or other original-filter inspection; it is not evidence
of historical parser behavior alone. Notably the scan does not inspect
starting locations, tracker coverage, feature validity or original
exception logs.

Next staged proof: run the same diagnostic across all 816 unmatched
with a larger frozen-positive control group (e.g. 100), retain
individual modern diagnostic statuses, and only then run separate
provenance-preserving V4 parse/extraction. Historic B3 still open;
historical B5 status is unchanged (`BLOCKED_PENDING_EVIDENCE`).
