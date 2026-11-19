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
        conditions = gates["conditions"]
        self.assertEqual(gates["release_status"], "BLOCKED_PENDING_EVIDENCE")
        self.assertTrue(conditions)
        self.assertTrue(all(type(value) is bool for value in conditions.values()))
        self.assertIn("b2_full_scale_four_way_and_deep_completed", conditions)
        self.assertIn("b3_raw_replay_to_final_feature_and_exclusion_provenance_resolved", conditions)
        self.assertIn("b4_full_scientific_environment_locked", conditions)
        # Already verified conditions may progress from False to True: don't
        # hardcode their historical state. Release remains blocked until ALL
        # conditions are satisfied and final approval is explicit.
        self.assertTrue(any(not value for value in conditions.values()))
        with self.assertRaisesRegex(ValueError, "RELEASE BLOCKED"):
            check.release_gates(ROOT)

    def test_release_rejects_approval_when_one_condition_is_unmet(self):
        with tempfile.TemporaryDirectory() as dirname:
            root = Path(dirname)
            (root / "gates.json").write_text(
                json.dumps({
                    "release_status": "APPROVED",
                    "conditions": {"verified": True, "unverified": False},
                }),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "RELEASE BLOCKED"):
                check.release_gates(root, "gates.json")

    def test_release_rejects_all_true_without_explicit_approval(self):
        with tempfile.TemporaryDirectory() as dirname:
            root = Path(dirname)
            (root / "gates.json").write_text(
                json.dumps({
                    "release_status": "BLOCKED_PENDING_EVIDENCE",
                    "conditions": {"verified": True},
                }),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Release status not APPROVED"):
                check.release_gates(root, "gates.json")

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
            # Layout-risk regression: review-only shortened titles must not
            # revert to the multi-line historical header overflow.
            from pptx import Presentation
            rebuilt = Presentation(str(target))
            for page, expected in (
                (8, "RF profile: why no_counter wins"),
                (11, "Classification: RF and XGB nearly tied"),
                (15, "Calibration: RF improves; XGB starts strong"),
            ):
                texts = [shape.text for shape in rebuilt.slides[page - 1].shapes
                         if shape.has_text_frame]
                self.assertIn(expected, texts)
            charts = [shape.chart for shape in rebuilt.slides[12].shapes
                      if shape.has_chart]
            self.assertEqual(len(charts), 2)
            self.assertTrue(all(not plot.has_data_labels
                                for chart in charts for plot in chart.plots))
            # Labels can survive at the chart *series* XML level even
            # when the plot-level has_data_labels flag is False.
            from pptx.oxml.ns import qn
            for page in (8, 11, 12, 13, 15):
                for shape in rebuilt.slides[page - 1].shapes:
                    if not shape.has_chart:
                        continue
                    vals = [float(v) for s in shape.chart.series for v in s.values]
                    if vals and all(0.0 <= v < 1.0 for v in vals):
                        self.assertTrue(all(s._ser.find(qn("c:dLbls")) is None
                                            for s in shape.chart.series))
            # All rate/probability chart labels in General format were
            # rounded to 0 or 1 by the historical deck renderer.
            for page in (8, 11, 12, 13, 15):
                for shape in rebuilt.slides[page - 1].shapes:
                    if not shape.has_chart:
                        continue
                    values = [float(v) for s in shape.chart.series for v in s.values]
                    if values and all(0.0 <= v < 1.0 for v in values):
                        self.assertTrue(all(not p.has_data_labels
                                            for p in shape.chart.plots))
            text = "\n".join(s.text for sl in rebuilt.slides
                             for s in sl.shapes if s.has_text_frame)
            self.assertIn("no_counter remained strong in the full benchmark", text)
            self.assertIn("Calibration adds little; use uncalibrated XGB", text)


if __name__ == "__main__":
    unittest.main()
