"""Streaming replay census and exclusion traceability regression tests."""
from __future__ import annotations

import importlib.util
import io
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('b3_replay_lineage', ROOT / 'scripts/69_audit_replay_lineage.py')
assert spec and spec.loader
lineage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lineage)


class ReplayLineageTests(unittest.TestCase):
    def make_zip(self, root: Path, rows: str) -> Path:
        path = root / 'dataset.zip'
        with zipfile.ZipFile(path, 'w') as archive:
            archive.writestr('dataset.csv', 'replay_id,time_sec,p1_wins\n' + rows)
        return path

    def test_census_and_reconciliation_only_with_actual_audit(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            processed = self.make_zip(root, 'r1,135,1\nr1,150,1\nr2,135,0\n')
            count, ids = lineage.processed_census(processed)
            self.assertEqual(count['snapshot_rows'], 3)
            self.assertEqual(count['unique_replays'], 2)
            audit = root / 'audit.csv'
            audit.write_text('replay_id,parse_status,failure_reason\n'
                             'r1,ok,\nr2,ok,\nr3,filtered,missing_winner\n')
            status = lineage.reconcile_raw_audit(audit, ids)
            self.assertEqual(status['exclusion_reason_counts'], {'missing_winner': 1})
            self.assertTrue(status['raw_to_final_reconciled'])

    def test_unknown_label_and_duplicate_snapshot_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            bad = self.make_zip(root, 'r1,135,0\nr1,135,0\n')
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                lineage.processed_census(bad)
            bad.unlink()
            bad = self.make_zip(root, 'r1,135,2\n')
            with self.assertRaisesRegex(ValueError, 'Unknown'):
                lineage.processed_census(bad)

    def test_ambiguous_downstream_exclusions_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, ids = lineage.processed_census(self.make_zip(root, 'r1,135,1\n'))
            audit = root / 'audit.csv'
            audit.write_text('replay_id,parse_status,failure_reason\nr1,ok,\nr2,ok,\n')
            with self.assertRaisesRegex(ValueError, 'downstream filtering log'):
                lineage.reconcile_raw_audit(audit, ids)


if __name__ == '__main__':
    unittest.main()
