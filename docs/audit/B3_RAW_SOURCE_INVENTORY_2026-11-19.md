# B3 — Original raw replay source recovery and inventory phase

**Date:** 2026-11-19  
**Branch:** `audit/sc2-corrections-2026-11-17`  
**Status:** original raw replay identities and exclusions **NOT YET RECONCILED**.

## 1. Evidence actually located

The tracked `data/raw/replay_sources.md` identifies the original published
**SC2ReSet: StarCraft II Esport Replaypack Set**:
- Zenodo **record 14963356**, DOI `10.5281/zenodo.14963356`;
- tracked `data/raw/replaypack_inventory.csv` lists **71** named archive
  files, with version **2.0.0** in the catalog.

**These 71 catalogue rows are a list of known upstream pack names, not
verified downloads, not a total replay count and not a record of the
specific packs used in the frozen CSV.**

The two user-supplied freeze archives include original historical files:
`scripts/sc2_parser_v3_counter120.py`,
`scripts/sc2_bridge_v3_counter120.py`,
`scripts/sc2_counters.py`, and
`scripts/buildf_v3_counter120.py`. SHA-256 checks of the *archived copies*:

| File | SHA-256 |
| --- | --- |
| `sc2_parser_v3_counter120.py` | `80ed2d83835c85fc03b4707c5431a2a29aeeb4166fc35777d3ca1bd590b3da1c` |
| `sc2_bridge_v3_counter120.py` | `89886649bfffba88a938f86518dd395ecc7caf0731c10520e3f96a52d12086e4` |
| `sc2_counters.py` | `ab4fad87512363d096f89d3e305d9c970aa9c0b13354eacb7d39ae64e658ce7b` |
| `buildf_v3_counter120.py` | `345f38e84557c5570f1498c7181ea141375581f689663d961d8aa2d9ca21fc32` |

The archived historical builder specifically:
1. iterates a *flat* `replays/` folder, finding filenames ending
   literally in `.SC2Replay`;
2. reads the replay header with `sc2reader.load_replay(..., load_level=0)`;
3. ignores durations below 180 seconds and above 1,800 seconds;
4. calls the historical bridge and retains snapshots where `time_sec > 120`;
5. returns `None` for many different failure conditions without an
   individual-replay reason log.

The historical bridge also returns `None` on replay load errors, fewer
than two players and absent worker-start locations; it does not strictly
check `len(replay.players) == 2`. Its winner-label fallback is not
sufficient to certify that unknown winners were all excluded historically.
The archived builder copy has malformed string literals, so its **stored
copy is not attested as the executable used for dataset generation**.

The uploaded processed dataset ZIPs, historical freeze ZIPs and final
delivery ZIPs contained **no directly listed `.SC2Replay` members** in
their top-level ZIP directories. Nested ZIP payloads in those deliveries
were not used as proof of the raw Zenodo collection. Raw source packs on
FAIR (if present) must be inventoried separately.

## 2. Identity gap to resolve

The frozen `v3.1` and `v3.2` data each contain **892,047** rows,
**23,241** distinct replay IDs and identical `replay_id,time_sec,p1_wins`
triples. The original upstream narrative says **24,057** replays.
The arithmetic difference is **816**, but it is **not** currently
certified as 816 parsed-and-excluded individual raw files; it could
also involve source-set choice, duplicate basenames, partial downloads
or other upstream bookkeeping differences.

Per-replay candidate identities and original failure reasons MUST be
observed before attributing any exclusions to duration, invalid winner,
non-1v1, missing tracker data, replay parsing or other causes.

## 3. New conservative read-only source inventory tool

`scripts/77_inventory_b3_raw_sources.py` scans:
- loose `.SC2Replay` filenames in a supplied raw root;
- direct `.SC2Replay` members of ZIP replaypacks, without extracting
  or decompressing their payload bytes;
- the tracked 71-pack source catalogue;
- frozen v3.1 replay identities, independently read via the B3 census.

Outputs are written into a **new** directory only; existing outputs are
not overwritten. The generated `summary.json`, `raw_entries.csv`,
`identity_reconciliation.csv` and `unresolved_ids.csv` distinguish
unique basenames from file entries, duplicate replay names, catalogued
packs absent from the local scan, raw IDs absent from the frozen dataset
and frozen IDs missing from scanned raw sources.

`unresolved_ids.csv` deliberately uses
`SOURCE_PRESENT_FROZEN_ABSENT_REASON_UNKNOWN`, not an invented
exclusion label. ZIP members that are themselves archives are flagged
but not recursively unpacked. Only ZIP central-directory member CRC32,
size and names are used; a CRC32 is **not** a cryptographic checksum.
The frozen ZIP and catalogue are SHA-256 hashed.

### A. Locate possible raw replay files on FAIR

```bash
cd ~/HPC-POMDP/ML-SC2/ML_SC2_tabular_audit_clean
git pull --ff-only origin audit/sc2-corrections-2026-11-17
source ~/.venvs/ml-sc2/bin/activate
export SC2_PYTHON="$(command -v python)"
export PYTHONPATH="$PWD/src"

"$SC2_PYTHON" -m unittest discover -s tests \
  -p 'test_b3_raw_source_inventory.py' -v

# Search for original replays / tournament ZIPs, without modifying them.
find "$HOME/HPC-POMDP" -type f \
  \( -iname '*.SC2Replay' -o -name '2016_IEM_10_Taipei.zip' \
     -o -name '2024_01_IEM_Katowice.zip' \) \
  -print | head -40
```

A flat `replays/` tree and a folder containing downloaded tournament
ZIPs both work, but the raw root must actually contain
`.SC2Replay` files or ZIP members. **Do not use
`data/processed/` or `data/raw/replaypack_inventory.csv` as a
substitute for real raw archives.**

### B. Only if genuine original replay sources are present

Choose a directory containing the real downloaded replaypacks or
replay files, e.g. `/path/to/actual/replaypacks`:

```bash
"$SC2_PYTHON" scripts/77_inventory_b3_raw_sources.py \
  --raw-root /path/to/actual/replaypacks \
  --processed-zip data/processed/starcraft_full_dataset_v3_1_fixed.zip \
  --pack-catalog data/raw/replaypack_inventory.csv \
  --output-dir results/b3_source_inventory_20261119

cat results/b3_source_inventory_20261119/summary.json
```

This scan will **fail closed** if no original replays are found or if
a source ZIP is unreadable. It never treats absence of an audit log as
proof of a legitimate exclusion. If original source data is not yet
present on FAIR, preserve this stage as **WAITING_FOR_RAW_SOURCES**:
locate the original workstation download or acquire the relevant
SC2ReSet packs from the DOI without silently changing dataset version.

## 3C. Newly located tracked real replay subset (important)

The repository itself already contains `data/raw/replay_subsets/`: a GitHub
directory inspection on 2026-11-19 listed **300 actual `.SC2Replay` files**
plus `placeholder.txt` (total visible replay payload sizes **57,690,402
bytes**, approximately 57.7 MB; this is raw source file size metadata, not
a cryptographic validation). This corrects any impression that the project
checkout has *no* raw replay files. The separately uploaded freeze and
processed ZIPs lacked direct `.SC2Replay` members, but the tracked subset
is usable for immediate source-identity and clean-room parser experiments.

**Immediate next command on FAIR (works without fetching the 71 packs):**

```bash
"$SC2_PYTHON" scripts/77_inventory_b3_raw_sources.py \
  --raw-root data/raw/replay_subsets \
  --processed-zip data/processed/starcraft_full_dataset_v3_1_fixed.zip \
  --output-dir results/b3_source_inventory_subset_20261119

cat results/b3_source_inventory_subset_20261119/summary.json
```

The reported `catalogued_packs_not_observed` will include all 71 packs
because this deliberately scans a 300-file *loose replay subset*, **not**
the full tournament ZIP source. That is expected and does not mean the
packs were excluded during historical dataset generation. Likewise, a
partial 300-file subset cannot reconcile the 24,057 upstream count or the
816 numerical difference. Its value is establishing which subset IDs are
present in the frozen dataset and preparing raw event/feature comparisons.

The recorded exact source provenance of the 300-file subset is not yet
attested; do not equate it automatically with the original full 71-pack
build or mark B3 closed.

## 4. What must follow a successful inventory

1. **Identity reconciliation:** verify actual pack files, unique
   replay names and 24,057 count; enumerate candidate absent replay IDs
   and any collisions. Do not assume the numeric difference is 816
   individually parsed failures until the actual source set is proven.
2. **Original failure evidence:** recover original per-replay processing
   status/logs or rerun the *historical semantics* on raw files with a
   pinned `sc2reader` version; document duration/parse/winner issues
   individually. A newly recomputed reason is evidence about the new
   execution, not necessarily the historical reason.
3. **Feature parity:** compare original raw-event primitive features
   and final snapshots for a reproducibly chosen replay subset, with
   exact parser/bridge/environment versions and numerical tolerances.
   Recover v3.2 upstream combat/matchup logic separately.
4. **Provenance certification:** require an attested original source
   selection/build manifest or clearly label any remaining historical
   execution uncertainties. The source-selection gate remains blocked
   until these steps are supported by actual evidence.

## Release disposition

This phase adds a source inventory tool and traceable historical source
observations, but **does not close B3**. The manifest
`b3_raw_replay_to_final_feature_and_exclusion_provenance_resolved`
must remain `false`. B2 numeric evidence and B4 frozen payloads are
unaffected; B5 release remains `BLOCKED_PENDING_EVIDENCE`.
