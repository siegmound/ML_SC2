"""B3 dataset-builder safety and replay inclusion audit regressions."""
from __future__ import annotations

import sys
import tempfile
import types
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
try:
    import sc2reader  # type: ignore
except ImportError:
    sys.modules['sc2reader'] = types.ModuleType('sc2reader')

from sc2proj.bridge import ParseResult
from sc2proj.builder import BuildConfig, DatasetBuilder


class BuilderSafetyTests(unittest.TestCase):
    def build_paths(self, temp: Path):
        return temp / 'dataset.zip', temp / 'manifest.json', temp / 'audit.csv'

    def test_duplicate_replay_basename_rejected_before_parsing(self):
        builder = DatasetBuilder(BuildConfig(dataset_name='new_dataset'))
        with tempfile.TemporaryDirectory() as dirname:
            targets = self.build_paths(Path(dirname))
            with patch.object(builder.bridge, 'process_replay') as process:
                with self.assertRaisesRegex(ValueError, 'Ambiguous replay_id'):
                    builder.build_from_replays([
                        Path('/packA/match.SC2Replay'), Path('/packB/match.SC2Replay')
                    ], *targets)
                process.assert_not_called()

    def test_frozen_archive_and_existing_output_protected(self):
        builder = DatasetBuilder(BuildConfig(dataset_name='new_dataset'))
        with tempfile.TemporaryDirectory() as dirname:
            root = Path(dirname)
            with self.assertRaisesRegex(ValueError, 'frozen final dataset'):
                builder.build_from_replays([], root / 'starcraft_full_dataset_v3_1_fixed.zip',
                                           root / 'new.json', root / 'new.csv')
            target, manifest, audit = self.build_paths(root)
            target.write_text('preserve me')
            with self.assertRaises(FileExistsError):
                builder.build_from_replays([], target, manifest, audit)
            self.assertEqual(target.read_text(), 'preserve me')

    def test_builder_tracks_exclusion_reasons_without_label_corruption(self):
        builder = DatasetBuilder(BuildConfig(dataset_name='new_dataset'))
        with tempfile.TemporaryDirectory() as dirname:
            root = Path(dirname)
            target, manifest, audit = self.build_paths(root)
            accepted = ParseResult('valid.SC2Replay', 'ok', '', pd.DataFrame({
                'replay_id': ['valid.SC2Replay'] * 2,
                'time_sec': [135, 150], 'p1_wins': [1, 1], 'diff_workers': [1., 2.],
            }), {'replay_id': 'valid.SC2Replay', 'parse_status': 'ok',
                 'failure_reason': '', 'num_snapshots': 2})
            rejected = ParseResult('bad.SC2Replay', 'filtered', 'missing_winner', None,
                                   {'replay_id': 'bad.SC2Replay',
                                    'parse_status': 'filtered', 'failure_reason': 'missing_winner',
                                    'num_snapshots': 0})
            with patch.object(builder.bridge, 'process_replay', side_effect=[accepted, rejected]):
                summary = builder.build_from_replays([
                    root / 'packA/valid.SC2Replay', root / 'packB/bad.SC2Replay'
                ], target, manifest, audit)
            self.assertEqual(summary['number_replays'], 1)
            self.assertEqual(summary['number_rows'], 2)
            self.assertEqual(summary['excluded_replay_count'], 1)
            self.assertEqual(summary['exclusion_reason_counts'], {'missing_winner': 1})
            self.assertEqual(summary['lineage_status'],
                             'independent_clean_room_not_reproduction_of_frozen_v3_1_or_v3_2')
            log = pd.read_csv(audit)
            self.assertIn('source_path', log.columns)
            self.assertEqual(len(log), 2)
            with zipfile.ZipFile(target) as archive:
                with archive.open('new_dataset.csv') as stream:
                    df = pd.read_csv(stream)
            self.assertEqual(set(df.p1_wins), {1})
            self.assertEqual(df.replay_id.nunique(), 1)

    def test_conflicting_outcome_rows_refused(self):
        builder = DatasetBuilder(BuildConfig(dataset_name='new_dataset'))
        with tempfile.TemporaryDirectory() as dirname:
            targets = self.build_paths(Path(dirname))
            bad = ParseResult('x.SC2Replay', 'ok', '', pd.DataFrame({
                'replay_id': ['x.SC2Replay'] * 2, 'time_sec': [135, 150],
                'p1_wins': [0, 1], 'feature': [2., 3.],
            }), {'replay_id': 'x.SC2Replay', 'parse_status': 'ok', 'failure_reason': ''})
            with patch.object(builder.bridge, 'process_replay', return_value=bad):
                with self.assertRaisesRegex(ValueError, 'conflicting winners'):
                    builder.build_from_replays([Path('x.SC2Replay')], *targets)
            for path in targets:
                self.assertFalse(path.exists())


if __name__ == '__main__':
    unittest.main()
