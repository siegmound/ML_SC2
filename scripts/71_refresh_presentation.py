"""B5 regenerate a corrected 19-slide audit deck from the historical PPTX.

The original presentation remains untouched. This script edits evidence statements
and statistics, adds a clearly labelled audit-status slide, and writes a new
presentation for review. Never declare a frozen release from this script alone.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

DEFAULT_SOURCE = "paper/sc2_outcome_prediction_presentation.pptx"
DEFAULT_OUTPUT = "dist/b5/sc2_outcome_prediction_presentation_B5_REVIEW.pptx"


def replace_text(slide, before: str, after: str) -> None:
    matches = []
    for shape in slide.shapes:
        if not shape.has_text_frame or before not in shape.text:
            continue
        matches.append(shape)
    if len(matches) != 1:
        raise ValueError(
            f"Expected unique slide text {before!r}, found {len(matches)}"
        )
    shape = matches[0]
    # Match and update within each paragraph, preserving the original fonts,
    # colors, text-box geometry and paragraph layout.
    for paragraph in shape.text_frame.paragraphs:
        combined = "".join(run.text for run in paragraph.runs)
        if before not in combined:
            continue
        first = paragraph.runs[0]
        first.text = combined.replace(before, after)
        for run in paragraph.runs[1:]:
            run.text = ""
        return
    # For the subtitle, all text is in the same shape and paragraph.
    raise ValueError(f"Text spans paragraphs; no unsafe presentation reflow: {before!r}")


def text(slide, x, y, w, h, value, size=15, color=(29, 41, 57),
         bold=False, align=None):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = shape.text_frame
    frame.word_wrap = True
    frame.margin_left = Inches(.06)
    frame.margin_top = Inches(.04)
    frame.margin_right = Inches(.06)
    frame.margin_bottom = Inches(.04)
    frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = frame.paragraphs[0]
    p.text = value
    if align is not None:
        p.alignment = align
    for r in p.runs:
        r.font.name = "Aptos"
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.color.rgb = RGBColor(*color)
    return shape


def add_audit_slide(presentation) -> None:
    slide = presentation.slides.add_slide(presentation.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = RGBColor(243, 245, 247)
    navy = (11, 41, 74)
    blue = (17, 58, 103)
    gray = (91, 101, 114)
    white = (255, 255, 255)
    bar = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, 0, 0, presentation.slide_width, Inches(.58)
    )
    bar.fill.solid()
    bar.fill.fore_color.rgb = RGBColor(*navy)
    bar.line.fill.background()
    text(slide, .45, .04, 9.8, .42, "B5 audit and release readiness",
         size=25, color=white, bold=True)
    text(slide, .6, .8, 12.1, .5,
         "Historical benchmark preserved; revised protocol and full reproducibility are still open",
         size=17, color=blue, bold=True)

    cells = [
        ("B0 + B1 | Evidence integrity",
         "Frozen original experiment artifacts. The XGB v3.2 four standard deviations are corrected in source tables."),
        ("B2 | Model evaluation",
         "RF fitted train+val historically; XGB fitted train with validation early stopping. Deep test peeking and calibration reuse are documented."),
        ("B3 | Replay provenance",
         "892,047 rows / 23,241 replay IDs checked. 816 exclusions remain unattributed; the 38-feature clean-room parser is not the archived 64/73-feature pipeline."),
        ("B4 + B5 | Release gates",
         "Local regression checks pass. Remote full LFS, full-scale reruns, complete environment lock and final CI verification are still required."),
    ]
    for index, (heading, description) in enumerate(cells):
        col = index % 2
        row = index // 2
        x, y = .60 + col * 6.35, 1.55 + row * 2.40
        card = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y),
            Inches(5.95), Inches(2.11)
        )
        card.fill.solid()
        card.fill.fore_color.rgb = RGBColor(*white)
        card.line.color.rgb = RGBColor(214, 222, 232)
        text(slide, x+.16, y+.15, 5.5, .37, heading, size=17, color=blue, bold=True)
        text(slide, x+.16, y+.59, 5.5, 1.3, description, size=13, color=(31, 41, 55))
    text(slide, .65, 6.65, 12.0, .40,
         "AUDITED RETROSPECTIVE REPORT — not yet a fully certified blind-test release",
         size=15, color=(146, 56, 47), bold=True, align=PP_ALIGN.CENTER)
    text(slide, 10.75, 7.12, 2.1, .22, "Final project analysis · slide 19",
         size=9, color=gray)


def build(source: Path, output: Path) -> dict:
    if source.resolve() == output.resolve():
        raise ValueError("Refusing to overwrite the original historical presentation")
    if output.exists():
        raise FileExistsError(f"Review presentation already exists: {output}")
    prs = Presentation(str(source))
    if len(prs.slides) != 18:
        raise ValueError(f"Expected 18-slide historical deck, found {len(prs.slides)}")
    changes = (
        (3, "The final dataset is built from replay archives through a deterministic replay-to-snapshot pipeline.",
            "Historical replay-to-table workflow; exact final feature formulas remain unverified."),
        (4, "All final claims are based on replay-aware multi-seed evaluation.",
            "Three replay-aware seeds, with documented historical methodological caveats."),
        (4, "• Validation is used for model selection; test is used once for final evaluation",
            "• Splits are grouped by replay; the test was inspected during historical development"),
        (10, "0.6128 ± 0.0069", "0.6128 ± 0.0056"),
        (10, "0.6093 ± 0.0062", "0.6093 ± 0.0051"),
        (10, "0.6597 ± 0.0059", "0.6597 ± 0.0048"),
        (10, "0.6486 ± 0.0030", "0.6486 ± 0.0024"),
        (10, "Three replay-aware seeds (42, 43, 44) on the two final dataset variants.",
            "Archived three-seed comparison; RF and XGB fitting budgets differ."),
        (11, "The differences are real but very small, which is why the full multi-seed protocol matters.",
            "The observed model gaps are small; a statistically decisive winner is not established."),
        (13, "The deep model was evaluated on the same full-data protocol rather than as a toy add-on.",
            "Historical Deep scores are descriptive; corrected test gating awaits full-data rerun."),
        (13, "• Test used once after model selection",
            "• Original runner scored each candidate on test; B2 runner corrects this"),
        (14, "Rigorous calibration protocol",
            "Historical calibration and revised B2 protocol"),
        (14, "Post-hoc calibration was treated as a proper validation-to-test experiment, not as a descriptive afterthought.",
            "Historical XGB validation was reused for tuning and calibration; four-way B2 reruns remain open."),
        (14, "Apply calibrator to untouched test predictions",
            "Apply fitted mapping to stored test predictions"),
        (14, "• The protocol used here makes calibration a real held-out evaluation and allows a fair comparison between RF and XGB.",
            "• Calibration used no test labels for fitting, but XGB reused validation for early stopping."),
        (15, "The final probability comparison is one of the clearest distinctions between the two tabular finalists.",
            "Retrospective calibration comparison (same validation reused in XGB early stopping)."),
        (17, "The project ends with two defensible winners, depending on what the downstream use actually needs.",
            "Historical RF/XGB trade-off; rankings are not statistically decisive."),
        (18, "Use RF if you care most about classification. Use XGB if you care most about calibrated probabilities. The deep model is credible, but not the final winner.",
            "Historically RF has slightly stronger classification accuracy and XGB better log loss; do not claim a blind-test winner before full reruns."),
    )
    for page, before, after in changes:
        replace_text(prs.slides[page - 1], before, after)
    add_audit_slide(prs)
    output.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(output))
    return {"slides": len(prs.slides), "text_corrections": len(changes),
            "output": str(output), "status": "REVIEW_ONLY_NOT_FROZEN_RELEASE"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(DEFAULT_SOURCE))
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    args = parser.parse_args()
    print(build(args.input, args.output))


if __name__ == "__main__":
    main()
