"""Schema lineage tests distinguish shared names from missing historical formulas."""
from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('b3_schema', ROOT / 'scripts/68_verify_feature_lineage.py')
assert spec and spec.loader
schema = importlib.util.module_from_spec(spec)
spec.loader.exec_module(schema)


class FeatureLineageTests(unittest.TestCase):
    def test_current_parser_and_archived_model_schema_match_manifest(self):
        derived = schema.derive(ROOT)
        manifest = json.loads((ROOT / schema.MANIFEST).read_text())
        self.assertEqual(derived, manifest)
        self.assertEqual(derived['counts'], {
            'clean_room': 38, 'v3_1': 64, 'v3_2': 73,
            'clean_room_in_v3_1': 29, 'missing_v3_1_formulas': 35,
            'clean_room_only': 9, 'added_in_v3_2': 9,
        })
        self.assertEqual(derived['reproduction_status'],
                         'NOT_REPRODUCIBLE_FROM_CURRENT_CLEAN_ROOM_PARSER')

    def test_zip_header_mismatch_is_detected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'toy.zip'
            with zipfile.ZipFile(path, 'w') as archive:
                archive.writestr('toy.csv', 'replay_id,time_sec,feature,p1_wins\na,135,0.3,1\n')
            schema.verify_zip_header(path, ['feature'])
            with self.assertRaisesRegex(ValueError, 'differs'):
                schema.verify_zip_header(path, ['wrong_name'])


if __name__ == '__main__':
    unittest.main()
