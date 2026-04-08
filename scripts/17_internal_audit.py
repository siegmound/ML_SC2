from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

TEXT_EXTENSIONS = {
    ".md",
    ".txt",
    ".csv",
    ".tex",
    ".py",
    ".yml",
    ".yaml",
    ".toml",
    ".json",
}

CODE_EXTENSIONS = {".py"}

IGNORE_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
}

TODO_PATTERNS = (
    "TODO",
    "FIXME",
    "XXX",
    "HACK",
)


def should_skip(path: Path) -> bool:
    return any(part in IGNORE_DIRS for part in path.parts)


def collect_paths(repo_root: Path) -> list[Path]:
    out: list[Path] = []
    for path in repo_root.rglob("*"):
        if path.is_dir() or should_skip(path):
            continue
        if path.suffix.lower() in TEXT_EXTENSIONS:
            out.append(path)
    return sorted(out)


def text_is_single_line(path: Path) -> bool:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        return False
    return len(lines) <= 2 and len(lines[0]) > 180 if lines else False


def file_contains_markers(path: Path) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return []
    hits = [marker for marker in TODO_PATTERNS if marker in text]
    return hits


def audit_python_syntax(path: Path) -> str | None:
    try:
        source = path.read_text(encoding="utf-8")
        ast.parse(source)
    except SyntaxError as exc:
        return f"{path}: syntax error at line {exc.lineno}: {exc.msg}"
    except UnicodeDecodeError:
        return f"{path}: could not decode as UTF-8"
    return None


def audit_long_lines(path: Path, threshold: int = 160) -> list[str]:
    issues: list[str] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        return issues

    for lineno, line in enumerate(lines, start=1):
        if len(line) > threshold:
            issues.append(f"{path}:{lineno}: line length {len(line)} > {threshold}")
    return issues


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Perform a lightweight internal audit on repository text files."
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
        help="Exit non-zero on style findings such as compressed text files or long lines.",
    )
    args = parser.parse_args()

    repo_root = args.repo_root
    paths = collect_paths(repo_root)

    syntax_errors: list[str] = []
    marker_hits: dict[str, list[str]] = {}
    compressed_files: list[str] = []
    long_line_issues: list[str] = []

    for path in paths:
        if path.suffix.lower() in CODE_EXTENSIONS:
            error = audit_python_syntax(path)
            if error:
                syntax_errors.append(error)
            long_line_issues.extend(audit_long_lines(path))

        hits = file_contains_markers(path)
        if hits:
            marker_hits[str(path)] = hits

        if text_is_single_line(path):
            compressed_files.append(str(path))

    report = {
        "syntax_errors": syntax_errors,
        "marker_hits": marker_hits,
        "compressed_files": compressed_files,
        "long_line_issues": long_line_issues,
    }

    out_dir = repo_root / "results" / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "internal_audit_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if syntax_errors:
        raise SystemExit("Python syntax issues found. See results/logs/internal_audit_report.json")

    if args.strict_style and (compressed_files or long_line_issues or marker_hits):
        raise SystemExit(
            "Style audit issues found. See results/logs/internal_audit_report.json"
        )


if __name__ == "__main__":
    main()
