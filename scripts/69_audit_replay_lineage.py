"""Audit processed replay IDs and optionally reconcile a genuine raw-parser audit.

Without --raw-audit-csv, source exclusions are UNATTRIBUTED, not verified.
No dataset archive or historical result is modified by this tool.
"""
from __future__ import annotations

import argparse
import csv
import json
import zipfile
from collections import Counter
from pathlib import Path


def processed_census(dataset_zip: Path) -> tuple[dict, set[str]]:
    if not zipfile.is_zipfile(dataset_zip):
        raise ValueError(f'Invalid/incomplete dataset ZIP: {dataset_zip}')
    labels: dict[str, int] = {}
    seen: dict[str, set[int]] = {}
    n_rows = 0
    min_t, max_t = float('inf'), float('-inf')
    with zipfile.ZipFile(dataset_zip) as archive:
        files = [n for n in archive.namelist() if n.lower().endswith('.csv')]
        if len(files) != 1:
            raise ValueError('Expected exactly one processed CSV in dataset ZIP')
        with archive.open(files[0]) as stream:
            import io
            reader = csv.DictReader(io.TextIOWrapper(stream, encoding='utf-8-sig', newline=''))
            if not {'replay_id', 'time_sec', 'p1_wins'}.issubset(reader.fieldnames or []):
                raise ValueError('Missing required replay, time or label fields')
            for row in reader:
                n_rows += 1
                replay = row['replay_id'].strip()
                if not replay:
                    raise ValueError(f'Blank replay ID at CSV row {n_rows}')
                time = float(row['time_sec'])
                if not time.is_integer():
                    raise ValueError(f'Non-integer snapshot time: {replay}, {time}')
                sec = int(time)
                if sec < 135 or sec % 15:
                    raise ValueError(f'Unexpected snapshot time: {replay}, {sec}')
                try:
                    label = int(row['p1_wins'])
                except (TypeError, ValueError) as error:
                    raise ValueError(f'Invalid winner label: {replay}') from error
                if label not in (0, 1):
                    raise ValueError(f'Unknown binary label: {replay}')
                if replay in labels and labels[replay] != label:
                    raise ValueError(f'Conflicting winner labels: {replay}')
                labels[replay] = label
                times = seen.setdefault(replay, set())
                if sec in times:
                    raise ValueError(f'Duplicate replay/time snapshot: {replay}, {sec}')
                times.add(sec)
                min_t, max_t = min(min_t, time), max(max_t, time)
    return {
        'snapshot_rows': n_rows, 'unique_replays': len(labels),
        'p1_winning_replays': sum(labels.values()),
        'snapshot_time_min_sec': int(min_t) if n_rows else None,
        'snapshot_time_max_sec': int(max_t) if n_rows else None,
        'duplicate_snapshot_keys': 0, 'conflicting_replay_labels': 0,
    }, set(labels)


def reconcile_raw_audit(raw_audit_csv: Path, final_ids: set[str]) -> dict:
    with raw_audit_csv.open(encoding='utf-8-sig', newline='') as source:
        reader = csv.DictReader(source)
        if not {'replay_id', 'parse_status', 'failure_reason'}.issubset(reader.fieldnames or []):
            raise ValueError('Raw audit requires replay_id, parse_status, failure_reason')
        statuses = {}
        for entry in reader:
            name = entry['replay_id']
            if not name or name in statuses:
                raise ValueError(f'Missing/duplicated raw replay ID: {name}')
            status, reason = entry['parse_status'], entry['failure_reason']
            if status != 'ok' and not reason:
                raise ValueError(f'Missing exclusion reason: {name}')
            statuses[name] = (status, reason)
    if final_ids - set(statuses):
        raise ValueError('Final replay IDs absent from raw-parser audit')
    if any(statuses[k][0] != 'ok' for k in final_ids):
        raise ValueError('Included processed replay listed as excluded in raw audit')
    unexplained = [name for name, (status, reason) in statuses.items()
                   if name not in final_ids and status == 'ok']
    if unexplained:
        raise ValueError(f'Raw parser accepted {len(unexplained)} replays absent from final dataset; downstream filtering log required')
    reasons = Counter(reason for name, (status, reason) in statuses.items() if name not in final_ids)
    return {'raw_replays': len(statuses), 'documented_excluded_replays': sum(reasons.values()),
            'exclusion_reason_counts': dict(sorted(reasons.items())),
            'raw_to_final_reconciled': True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--processed-zip', type=Path, required=True)
    parser.add_argument('--raw-audit-csv', type=Path)
    parser.add_argument('--upstream-replay-count', type=int, default=24057)
    parser.add_argument('--output-json', type=Path)
    args = parser.parse_args()
    report, final_ids = processed_census(args.processed_zip)
    report['upstream_replay_count_documented_not_raw_verified'] = args.upstream_replay_count
    report['unattributed_difference_from_documented_upstream'] = args.upstream_replay_count - report['unique_replays']
    if args.raw_audit_csv:
        raw = reconcile_raw_audit(args.raw_audit_csv, final_ids)
        if raw['raw_replays'] != args.upstream_replay_count:
            raise ValueError('Raw audit replay count does not match documented upstream count')
        report.update(raw)
        report['exclusion_provenance_status'] = 'RECONCILED_RAW_AUDIT'
    else:
        report['exclusion_provenance_status'] = 'UNVERIFIED_NO_RAW_AUDIT'
    if args.output_json:
        if args.output_json.exists():
            raise FileExistsError(f'Refusing to overwrite existing audit report: {args.output_json}')
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
