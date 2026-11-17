"""B5 source consistency and default-deny release verification."""
from __future__ import annotations

import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def import_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


check = import_script("72_verify_publication")


class PublicationChecks(unittest.TestCase):
    def test_canonical_paper_uses_archived_numeric_table_and_audit_language(self):
        result = check.verify(ROOT)
        self.assertEqual(result["paper_rows"], 4)
        self.assertEqual(result["paper_b5_caveats"], "PASS")

    def test_current_release_is_blocked_by_explicit_pending_evidence(self):
        gates = json.loads((ROOT / check.DEFAULT_GATES).read_text(encoding="utf-8"))
        self.assertEqual(gates["release_status"], "BLOCKED_PENDING_EVIDENCE")
        self.assertFalse(gates["conditions"]["b2_full_scale_four_way_and_deep_completed"])
        self.assertFalse(gates["conditions"]["b4_clean_clone_and_remote_ci_passed"])
        with self.assertRaisesRegex(ValueError, "RELEASE BLOCKED"):
            check.release_gates(ROOT)

    def test_stale_standard_deviation_is_detected(self):
        with tempfile.TemporaryDirectory() as dirname:
            root = Path(dirname)
            (root / "paper").mkdir()
            (root / "tables").mkdir()
            shutil.copy2(ROOT / check.TABLE, root / check.TABLE)
            valid = (ROOT / check.TEX).read_text(encoding="utf-8")
            self.assertIn("0.6128 $\\pm$ 0.0056", valid)
            # Deliberately corrupt one row while keeping the other rows intact.
            tampered = valid.replace(
                "0.6128 $\\pm$ 0.0056", "0.6128 $\\pm$ 0.0069", 1
            )
            (root / check.TEX).write_text(tampered, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Stale|benchmark"):
                check.verify(root)

    def test_review_deck_regenerates_if_pptx_support_available(self):
        try:
            import pptx  # noqa: F401
        except ImportError:
            self.skipTest("python-pptx only installed in publishing/full CI")
        slides = import_script("71_refresh_presentation")
        source = ROOT / slides.DEFAULT_SOURCE
        if not source.exists():
            self.skipTest("Historical PPTX not available in partial test fixture")
        with tempfile.TemporaryDirectory() as dirname:
            target = Path(dirname) / "b5_review.pptx"
            outcome = slides.build(source, target)
            self.assertEqual(outcome["slides"], 19)
            result = check.verify_deck(target, check.parse_table(ROOT))
            self.assertEqual(result["slide_metrics"], "PASS")
            self.assertEqual(result["status"], "REVIEW_ONLY_NOT_FROZEN")


if __name__ == "__main__":
    unittest.main()
