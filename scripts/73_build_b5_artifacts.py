"""Build B5 review PDF/PPTX without overwriting historical release binaries.

Requires latexmk/pdflatex (texlive) and requirements-publishing.txt.
The resulting manifest is REVIEW_ONLY; do not tag/release without all gates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str], cwd: Path) -> None:
    process = subprocess.run(command, cwd=cwd, check=False, text=True,
                             capture_output=True)
    if process.returncode:
        raise RuntimeError(
            f"{' '.join(command)} FAILED\nSTDOUT:\n{process.stdout[-4500:]}\n"
            f"STDERR:\n{process.stderr[-3000:]}"
        )


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for buf in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            hasher.update(buf)
    return hasher.hexdigest()


def build(root: Path, output_dir: Path) -> dict:
    root = root.resolve()
    output_dir = output_dir.resolve()
    if output_dir == root / "paper":
        raise ValueError("Refusing to overwrite canonical historical publication directory")
    for path in (
        output_dir / "sc2_outcome_prediction_paper_B5_REVIEW.pdf",
        output_dir / "sc2_outcome_prediction_presentation_B5_REVIEW.pptx",
        output_dir / "b5_build_manifest.json",
    ):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing B5 review: {path}")
    if not shutil.which("latexmk"):
        raise RuntimeError("latexmk is required for reproducible canonical LaTeX export")

    run([sys.executable, str(root / "scripts/72_verify_publication.py"),
         "--repo-root", str(root)], root)
    source = root / "paper/sc2_outcome_prediction_paper.tex"
    original_deck = root / "paper/sc2_outcome_prediction_presentation.pptx"
    if not source.is_file() or not original_deck.is_file():
        raise FileNotFoundError("Canonical historical paper and PowerPoint are required")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compile with paper/ as working directory so ./img references resolve.
    latex_output = output_dir / "latex"
    latex_output.mkdir(exist_ok=True)
    run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error",
         "-file-line-error", f"-output-directory={latex_output}",
         str(source)], root / "paper")
    compiled = latex_output / "sc2_outcome_prediction_paper.pdf"
    if not compiled.is_file():
        raise RuntimeError("LaTeX compilation produced no PDF")
    pdf = output_dir / "sc2_outcome_prediction_paper_B5_REVIEW.pdf"
    shutil.copy2(compiled, pdf)

    slides = output_dir / "sc2_outcome_prediction_presentation_B5_REVIEW.pptx"
    run([sys.executable, str(root / "scripts/71_refresh_presentation.py"),
         "--input", str(original_deck), "--output", str(slides)], root)
    run([sys.executable, str(root / "scripts/72_verify_publication.py"),
         "--repo-root", str(root), "--deck", str(slides)], root)
    # Check the actual PDF is parseable with non-empty page text.
    from pypdf import PdfReader
    reader = PdfReader(str(pdf))
    if len(reader.pages) < 12:
        raise ValueError("Suspiciously short compiled paper")
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    if "Audit Status and Reproducibility Boundaries" not in text:
        raise ValueError("B5 audit section absent from the compiled PDF")
    if "0.0056" not in text or "0.0024" not in text:
        raise ValueError("Corrected benchmark metrics missing from compiled PDF")

    manifest = {
        "status": "REVIEW_ONLY_NOT_RELEASE_APPROVED",
        "source_branch": "audit/sc2-corrections-2026-11-17",
        "paper_tex": "paper/sc2_outcome_prediction_paper.tex",
        "historical_presentation": "paper/sc2_outcome_prediction_presentation.pptx",
        "pdf_pages": len(reader.pages),
        "pptx_slides": 19,
        "artifacts": {
            pdf.name: {"size_bytes": pdf.stat().st_size, "sha256": sha256(pdf)},
            slides.name: {"size_bytes": slides.stat().st_size, "sha256": sha256(slides)},
        },
        "unverified": [
            "Full B2 model reruns", "B3 replay-to-feature parity and exclusions",
            "B5 expert visual review and release gate approval",
        ],
    }
    (output_dir / "b5_build_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("dist/b5"))
    args = parser.parse_args()
    try:
        print(json.dumps(build(args.repo_root, args.output_dir), indent=2))
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"FAIL: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
