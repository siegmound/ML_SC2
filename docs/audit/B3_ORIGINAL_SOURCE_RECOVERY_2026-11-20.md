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
