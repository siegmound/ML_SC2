from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

LONG_LINE_LIMIT = 140
COMPRESSED_LINE_LIMIT = 500


def python_files(repo_root: Path):
    for folder in ["scripts", "src"]:
        base = repo_root / folder
        if not base.exists():
            continue
        for path in base.rglob("*.py"):
            yield path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run internal syntax and hygiene checks on Python files."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root path.",
    )
    parser.add_argument(
        "--strict-style",
        action="store_true",
        help="Treat long-line and compressed-file findings as fatal.",
    )
    args = parser.parse_args()

    syntax_errors: list[str] = []
    empty_files: list[str] = []
    marker_hits: list[str] = []
    long_lines: list[str] = []
    compressed_files: list[str] = []

    for path in python_files(args.repo_root):
        text = path.read_text(encoding="utf-8")
        rel = str(path.relative_to(args.repo_root))

        if not text.strip():
            empty_files.append(rel)
            continue

        lines = text.splitlines()

        if path.name != "17_internal_audit.py":
            task_markers = ("TO" + "DO", "FIX" + "ME")
            if any(marker in text for marker in task_markers):
                marker_hits.append(rel)

        max_len = max((len(line) for line in lines), default=0)
        if max_len > LONG_LINE_LIMIT:
            long_lines.append(f"{rel}: max_line_length={max_len}")

        if len(lines) <= 3 and len(text) >= COMPRESSED_LINE_LIMIT:
            compressed_files.append(f"{rel}: lines={len(lines)} chars={len(text)}")

        try:
            ast.parse(text)
        except SyntaxError as exc:
            syntax_errors.append(f"{rel}:{exc.lineno}:{exc.offset}: {exc.msg}")

    lines_out: list[str] = [
        "INTERNAL AUDIT SUMMARY",
        f"syntax_errors={len(syntax_errors)}",
        f"empty_files={len(empty_files)}",
        f"marker_hits={len(marker_hits)}",
        f"long_line_hits={len(long_lines)}",
        f"compressed_files={len(compressed_files)}",
        "",
    ]

    if syntax_errors:
        lines_out.append("SYNTAX ERRORS:")
        lines_out.extend(syntax_errors)
        lines_out.append("")

    if empty_files:
        lines_out.append("EMPTY FILES:")
        lines_out.extend(empty_files)
        lines_out.append("")

    if marker_hits:
        lines_out.append("TASK MARKER HITS:")
        lines_out.extend(marker_hits)
        lines_out.append("")

    if long_lines:
        lines_out.append("LONG LINE WARNINGS:")
        lines_out.extend(long_lines)
        lines_out.append("")

    if compressed_files:
        lines_out.append("COMPRESSED FILE WARNINGS:")
        lines_out.extend(compressed_files)
        lines_out.append("")

    out_dir = args.repo_root / "results" / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    text_report = out_dir / "internal_audit.txt"
    json_report = out_dir / "internal_audit.json"

    text_report.write_text("\n".join(lines_out) + "\n", encoding="utf-8")
    json_report.write_text(
        json.dumps(
            {
                "syntax_errors": syntax_errors,
                "empty_files": empty_files,
                "marker_hits": marker_hits,
                "long_lines": long_lines,
                "compressed_files": compressed_files,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    style_fail = bool(long_lines or compressed_files)
    if syntax_errors:
        raise SystemExit("Internal audit found syntax errors. See results/logs/internal_audit.txt")
    if args.strict_style and style_fail:
        raise SystemExit("Style-risk warnings detected. See results/logs/internal_audit.txt")


if __name__ == "__main__":
    main()
