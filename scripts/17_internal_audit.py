from __future__ import annotations

import argparse
import ast
from pathlib import Path


def python_files(repo_root: Path):
    for folder in ['scripts', 'src']:
        base = repo_root / folder
        if not base.exists():
            continue
        for path in base.rglob('*.py'):
            yield path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo-root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()

    syntax_errors: list[str] = []
    empty_files: list[str] = []
    marker_hits: list[str] = []

    for path in python_files(args.repo_root):
        text = path.read_text(encoding='utf-8')
        rel = str(path.relative_to(args.repo_root))
        if not text.strip():
            empty_files.append(rel)
            continue
        task_markers = ('TO' + 'DO', 'FIX' + 'ME')
        if path.name != '17_internal_audit.py' and any(marker in text for marker in task_markers):
            marker_hits.append(rel)
        try:
            ast.parse(text)
        except SyntaxError as exc:
            syntax_errors.append(f'{rel}:{exc.lineno}:{exc.offset}: {exc.msg}')

    lines: list[str] = []
    lines.append('INTERNAL AUDIT SUMMARY')
    lines.append(f'syntax_errors={len(syntax_errors)}')
    lines.append(f'empty_files={len(empty_files)}')
    lines.append(f'marker_hits={len(marker_hits)}')
    lines.append('')

    if syntax_errors:
        lines.append('SYNTAX ERRORS:')
        lines.extend(syntax_errors)
        lines.append('')
    if empty_files:
        lines.append('EMPTY FILES:')
        lines.extend(empty_files)
        lines.append('')
    if marker_hits:
        lines.append('TASK MARKER HITS:')
        lines.extend(marker_hits)
        lines.append('')

    out_path = args.repo_root / 'results' / 'logs' / 'internal_audit.txt'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')

    if syntax_errors:
        raise SystemExit('Internal audit found syntax errors. See results/logs/internal_audit.txt')


if __name__ == '__main__':
    main()
