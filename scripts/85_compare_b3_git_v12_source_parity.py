#!/usr/bin/env python3
"""B3: paired raw-to-frozen parity using genuine Git v1.2 source revision.

This is a read-only, controlled alternative-source experiment, NOT a
replacement for the original generator or an implicit release approval.
Load four files from one immutable Git commit with pinned blob SHA-1,
then compare the *same seed-42 replay sample* against frozen v3.1 and
the already preserved historical freeze-source reconstruction.
No model tuning, dataset writes, replay changes or formula edits.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parent
V12_COMMIT = "bd64a32b305cc7203d592c7fc9015a01f962d6f9"
GIT_SOURCE_BLOBS = {
    "sc2_parser_v3_counter120.py": "d839291989936b5867d42f427d419e291f6090da",
    "sc2_bridge_v3_counter120.py": "7856a1c74a5c475bf7414c97772b3461693cb663",
    "sc2_constants.py": "0a0a9faf5e5ea07f696084db37e399a5bb06aac3",
    "sc2_counters.py": "e881ee2513bab91ea8eaaa6852a662c66361e46a",
}


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def git_blob_sha(payload: bytes) -> str:
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload).hexdigest()


def load_git_v12_sources(repo: Path, destination: Path,
                         *, commit: str = V12_COMMIT) -> dict:
    require(commit == V12_COMMIT,
            f"Only independently identified immutable Git v1.2 source revision is supported: {V12_COMMIT}")
    require((repo / ".git").exists(), f"Not a Git checkout: {repo}")
    try:
        resolved = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--verify", f"{commit}^{{commit}}"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(
            "Pinned historical v1.2 commit not available locally: run 'git fetch origin v1.2'"
        ) from exc
    require(resolved == commit, "Historical Git commit mismatch")
    results = {}
    for filename, blob in GIT_SOURCE_BLOBS.items():
        try:
            process = subprocess.run(
                ["git", "-C", str(repo), "show", f"{commit}:scripts/{filename}"],
                capture_output=True, check=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise ValueError(f"Git v1.2 source unavailable: {filename}") from exc
        payload = process.stdout
        observed = git_blob_sha(payload)
        require(observed == blob, f"Historical v1.2 Git blob differs from independently checked source: {filename}")
        (destination / filename).write_bytes(payload)
        results[filename] = {
            "git_blob_sha1": observed,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }
    return results


def original_comparator():
    spec = importlib.util.spec_from_file_location(
        "b3_v31_original_comparator", HERE / "79_compare_b3_historical_raw_parity.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def verify_baseline(baseline: Path, selected: list[tuple[str, Path]],
                    expected_frozen_sha: str, sha256_file) -> dict:
    require(baseline.is_file(), f"Missing initial script79 audit report: {baseline}")
    with baseline.open(encoding="utf-8") as src:
        manifest = json.load(src)
    require(manifest["schema"] == "ml_sc2.b3.historical_raw_to_v31_sample.v1",
            "Unexpected original audit provenance schema")
    require(manifest["frozen_v31_sha256"] == expected_frozen_sha,
            "Original baseline frozen SHA mismatches v1.2 comparison")
    ids = [item[0] for item in selected]
    require(manifest["sampled_replay_ids"] == ids,
            "Source v1.2 comparison must use identical baseline replay IDs")
    existing = {
        item["replay_id"]: item["raw_sha256"]
        for item in manifest["sampled_raw_file_sha256"]
    }
    require(len(existing) == len(ids) and set(existing) == set(ids),
            "Missing/duplicate original raw replay identities in baseline")
    for replay_id, path in selected:
        require(sha256_file(path) == existing[replay_id],
                f"Raw replay SHA differs from original script79 comparison: {replay_id}")
    return {
        "baseline_json_sha256": sha256_file(baseline),
        "baseline_original_sampled_parity": manifest["sampled_parity_count"],
        "baseline_original_feature_mismatches": manifest["total_feature_mismatches"],
        "same_raw_source_bytes_as_baseline": True,
        "baseline_sampled_replay_ids": ids,
    }


def summarize_comparisons(rows: list[dict], detail: list[dict]) -> dict:
    by_feature = Counter()
    feature_comparisons = Counter()
    for row in detail:
        by_feature[row["feature"]] += row["mismatches"]
        feature_comparisons[row["feature"]] += row["comparisons"]
    return {
        "number_of_replays": len(rows),
        "full_replay_parity_count": sum(r.get("status") == "SAMPLED_PARITY" for r in rows),
        "replays_with_mismatches_or_errors": sum(
            r.get("status") != "SAMPLED_PARITY" for r in rows
        ),
        "total_compared_feature_values": sum(
            r.get("feature_comparisons", 0) for r in rows
        ),
        "total_feature_mismatches": sum(
            r.get("feature_mismatches", 0) for r in rows
        ),
        "total_label_mismatches": sum(r.get("label_mismatches", 0) for r in rows),
        "total_time_keys_missing_or_added": sum(
            r.get("time_keys_only_frozen", 0) + r.get("time_keys_only_reconstructed", 0)
            for r in rows
        ),
        "per_feature_mismatch_counts": dict(sorted(by_feature.items())),
        "per_feature_comparison_counts": dict(sorted(feature_comparisons.items())),
        "feature_columns_with_mismatches": sum(n > 0 for n in by_feature.values()),
    }


def csv_save(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPOSITORY)
    parser.add_argument("--raw-root", type=Path,
                        default=Path("data/raw/replay_subsets"))
    parser.add_argument("--inventory-csv", type=Path, default=Path(
        "results/b3_source_inventory_subset_20261119/identity_reconciliation.csv"))
    parser.add_argument("--processed-zip", type=Path, default=Path(
        "data/processed/starcraft_full_dataset_v3_1_fixed.zip"))
    parser.add_argument("--original-parity-dir", type=Path, default=Path(
        "results/b3_raw_to_v31_parity_sample3_20261119"))
    parser.add_argument("--max-replays", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path(
        "results/b3_git_v12_raw_parity_sample3_20261120"))
    args = parser.parse_args()
    out = args.output_dir.resolve()
    original_root = args.raw_root.resolve()
    dataset = args.processed_zip.resolve()
    baseline_dir = args.original_parity_dir.resolve()
    require(not out.exists(), f"Refusing to overwrite historical report: {out}")
    for area in [original_root, baseline_dir]:
        require(not (out == area or area in out.parents),
                "Cannot write new audit results inside input evidence directories")
    require(out != dataset, "Cannot replace frozen data")
    baseline = original_comparator()
    frozen_sha = baseline.sha256(dataset)
    require(frozen_sha == baseline.EXPECTED_V31_ZIP_SHA256,
            "Frozen v3.1 SHA256 changed since original sample")
    require(args.seed == 42 and args.max_replays == 3,
            "Paired comparison must use the original three seed-42 replays")
    selected = baseline.select_present_replays(
        args.inventory_csv, original_root, args.max_replays, args.seed
    )
    original_evidence = verify_baseline(
        baseline_dir / "summary.json", selected, frozen_sha, baseline.sha256
    )
    frozen = baseline.frozen_selected(dataset, {n for n, _ in selected})

    reports, features, examples = [], [], []
    with tempfile.TemporaryDirectory(prefix="b3_gitv12_") as folder:
        old_path = Path(folder)
        git_source_hashes = load_git_v12_sources(args.repo.resolve(), old_path)
        bridge = baseline.load_historical_bridge(old_path)
        for replay_id, path in selected:
            try:
                candidate = bridge.process_single_replay(str(path))
                if candidate is None or candidate.empty:
                    reports.append({
                        "replay_id": replay_id,
                        "status": "GIT_V12_BRIDGE_RETURNED_NO_ROWS",
                        "frozen_rows": len(frozen[frozen.replay_id == replay_id]),
                        "reconstructed_rows": 0,
                    })
                    continue
                candidate = candidate[candidate["time_sec"] > 120].copy()
                candidate_report, per_field, discrepancies = baseline.compare_replay(
                    replay_id, frozen[frozen.replay_id == replay_id], candidate,
                    atol=1e-6, rtol=1e-6,
                )
                reports.append(candidate_report)
                features.extend(per_field)
                examples.extend(discrepancies)
            except Exception as exc:
                reports.append({
                    "replay_id": replay_id, "status": "GIT_V12_RECONSTRUCTION_EXCEPTION",
                    "error_type": type(exc).__name__, "error_detail": str(exc)[:300],
                })
    summary = summarize_comparisons(reports, features)
    manifest = {
        "schema": "ml_sc2.b3.git_v12_source_raw_parity.v1",
        "git_revision": V12_COMMIT,
        "original_repo_branch": "v1.2",
        "original_git_source_blobs": git_source_hashes,
        "source_description": (
            "Historical Git v1.2 parser/bridge/constants/counters, immutable "
            "revision and four independently checked Git blob identities."
        ),
        "frozen_v31_sha256": frozen_sha,
        "original_sample": original_evidence,
        "sample_seed": 42,
        "numerical_atol": 1e-6, "numerical_rtol": 1e-6,
        "comparison": summary,
        "original_generator_attested": False,
        "whole_corpus_reconstruction_attested": False,
        "raw_exclusion_provenance_resolved": False,
        "v3_2_upstream_attested": False,
        "b3_release_gate_closed": False,
        "warning": (
            "Historical Git revision is a genuine historical alternative, "
            "not proof that this is the executed original build. All three "
            "raw replays have been inspected before this hypothesis was posed."
        ),
    }
    out.mkdir(parents=True, exist_ok=False)
    (out / "summary.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    csv_save(out / "per_replay.csv", reports,
             sorted({key for r in reports for key in r}))
    csv_save(out / "per_feature.csv", features,
             ["replay_id", "feature", "comparisons", "mismatches"])
    csv_save(out / "mismatch_examples.csv", examples,
             ["replay_id", "time_sec", "feature", "frozen_value", "reconstructed_value"])
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
