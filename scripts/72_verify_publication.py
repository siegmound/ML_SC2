"""B5: check publication numbers, methodological language, and release gates.

By default this is a source-consistency check that succeeds even while release
is blocked. --deck points to an already regenerated *review* PowerPoint.
--release requires every named release gate to be documented as true.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

DEFAULT_GATES = "docs/audit/b5_release_gates_2026-11-17.json"
TABLE = "tables/final_full_multiseed_summary.csv"
TEX = "paper/sc2_outcome_prediction_paper.tex"
METRICS = (("acc_mean", "acc_std"), ("balacc_mean", "balacc_std"),
           ("auc_mean", "auc_std"), ("logloss_mean", "logloss_std"))
ROWS = (
    ("v3_1_fixed", "rf", "no_counter"),
    ("v3_1_fixed", "xgb", "full"),
    ("v3_2_combatfix", "rf", "no_counter"),
    ("v3_2_combatfix", "xgb", "full"),
)


def parse_table(root: Path) -> dict:
    with (root / TABLE).open(newline="", encoding="utf-8") as src:
        records = list(csv.DictReader(src))
    mapping = {(r["dataset"], r["model"], r["profile"]): r for r in records}
    if len(records) != 4 or len(mapping) != 4 or set(mapping) != set(ROWS):
        raise ValueError("Unexpected final tabular benchmark rows")
    return mapping


def tex_row(row: dict) -> str:
    return " & ".join(
        f'{float(row[mean]):.4f} $\\pm$ {float(row[std]):.4f}'
        for mean, std in METRICS
    )


def verify_paper(root: Path, mapping: dict) -> dict:
    source = (root / TEX).read_text(encoding="utf-8")
    if r"\section{Audit Status and Reproducibility Boundaries}" not in source:
        raise ValueError("B5 audit boundary section missing from canonical paper")
    tokens = (
        "38 predictive fields", "64 fields", "73 fields",
        "816 source replays", "train plus validation",
        "previously inspected test partitions",
        "full-data reruns are pending",
        "remote Git LFS payload availability",
    )
    normalized = re.sub(r"\\s+", " ", source)
    missing = [value for value in tokens if value not in normalized]
    if missing:
        raise ValueError(f"Methodological caveats missing from paper: {missing}")
    if "Final project paper v6.2" in source or "pdfauthor={OpenAI}" in source:
        raise ValueError("Outdated LaTeX metadata remains")
    if "0.6128 $\\pm$ 0.0069" in source:
        raise ValueError("Stale v3.2 XGB population standard deviations remain")
    verified = {}
    for (dataset, model, profile), row in mapping.items():
        escaped_name = dataset.replace("_", r"\_")
        needle = f"\\texttt{{{escaped_name}}} & {model.upper() if model == 'xgb' else 'RF'}"
        # The paper's profile column is set to 'full' or \texttt{no\_counter}.
        matches = [line.strip() for line in source.splitlines()
                   if needle in line and "$\\pm$" in line]
        if len(matches) != 1 or tex_row(row) not in matches[0]:
            raise ValueError(f"Paper benchmark does not match table for {(dataset,model,profile)}")
        verified[f"{dataset}/{model}/{profile}"] = tex_row(row)
    return {"paper_rows": len(verified), "paper_b5_caveats": "PASS"}


def verify_deck(path: Path, mapping: dict) -> dict:
    from pptx import Presentation

    deck = Presentation(str(path))
    if len(deck.slides) != 19:
        raise ValueError(f"B5 review deck must have 19 slides, got {len(deck.slides)}")
    text = "\n".join(shape.text for slide in deck.slides
                     for shape in slide.shapes if shape.has_text_frame)
    for row in mapping.values():
        # All distinct metric strings must appear in the updated benchmark.
        for mean, std in METRICS:
            expected = f'{float(row[mean]):.4f} ± {float(row[std]):.4f}'
            if expected not in text:
                raise ValueError(f"Missing benchmark metric in deck: {expected}")
    if "0.6128 ± 0.0069" in text or "0.6093 ± 0.0062" in text:
        raise ValueError("Old population-standard-deviation row still in deck")
    if "B5 audit and release readiness" not in text:
        raise ValueError("Release-readiness slide missing")
    if "Test used once after model selection" in text:
        raise ValueError("Historical deep-test claim remains in deck")
    return {"slide_count": len(deck.slides), "slide_metrics": "PASS",
            "status": "REVIEW_ONLY_NOT_FROZEN"}


def release_gates(root: Path, rel_path: str = DEFAULT_GATES) -> dict:
    gates = json.loads((root / rel_path).read_text(encoding="utf-8"))
    conditions = gates.get("conditions", {})
    if not conditions or any(not isinstance(v, bool) for v in conditions.values()):
        raise ValueError("Every release condition must be an explicit boolean")
    pending = sorted(name for name, value in conditions.items() if not value)
    if pending:
        raise ValueError(f"RELEASE BLOCKED: {len(pending)} unfinished gates: {pending}")
    if gates.get("release_status") != "APPROVED":
        raise ValueError("Release status not APPROVED even though all flags are true")
    return {"release": "APPROVED"}


def verify(root: Path, deck: Path | None = None, strict_release: bool = False) -> dict:
    mapping = parse_table(root)
    summary = verify_paper(root, mapping)
    if deck is not None:
        summary.update(verify_deck(deck, mapping))
    if strict_release:
        summary.update(release_gates(root))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--deck", type=Path, help="B5 regenerated review PowerPoint (optional)")
    parser.add_argument("--release", action="store_true", help="Fail while any release gate is open")
    args = parser.parse_args()
    try:
        print(json.dumps(verify(args.repo_root, args.deck, args.release), indent=2))
    except (ValueError, KeyError, OSError) as exc:
        print(f"FAIL: {exc}")
        return 1
    print("PASS: B5 publication sources are internally consistent"
          + (" and release-approved" if args.release else "; release approval not implied"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
