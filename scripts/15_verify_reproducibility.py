from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

import yaml

REQUIRED_PATHS = [
    "README.md",
    "requirements.txt",
    "pyproject.toml",
    "environment.yml",
    "docs/methodology.md",
    "docs/dataset_schema.md",
    "docs/experiment_registry.md",
    "docs/data_provenance.md",
    "docs/known_limitations.md",
    "docs/repo_freeze_checklist.md",
    "data/raw/README.md",
    "data/raw/replay_sources.md",
    "data/raw/replaypack_inventory.csv",
    "data/processed/manifests/upstream_replay_source_manifest.md",
    "paper/README.md",
    "paper/final_results_refinement.tex",
    "tables/final_full_multiseed_summary.csv",
    "tables/deep_final_summary.csv",
    "tables/rigorous_calibration_summary.csv",
    "scripts/15_verify_reproducibility.py",
    "scripts/17_internal_audit.py",
    "scripts/53_run_xgb_full_gpu_fixed_v3.py",
    "scripts/54_run_rf_full.py",
    "scripts/61_run_xgb_for_calibration.py",
    "scripts/62_run_rf_for_calibration.py",
    "scripts/63_run_rigorous_calibration.py",
    "scripts/64_run_deep_finalist.py",
    "data/processed/starcraft_full_dataset_v3_1_fixed.zip",
    "data/processed/starcraft_full_dataset_v3_2_combatfix.zip",
]

DEPENDENCY_WHITELIST = {
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "matplotlib",
    "pyyaml",
    "tqdm",
    "joblib",
    "xgboost",
    "sc2reader",
    "pyarrow",
    "torch",
    "skorch",
}

ALIAS_MAP = {
    "pytorch": "torch",
    "scikit_learn": "scikit-learn",
    "pyyaml": "pyyaml",
}


def normalize_requirement_name(entry: str) -> str:
    token = re.split(r"[<>=!~;\[\]\s]", entry.strip(), maxsplit=1)[0]
    token = token.lower().replace("_", "-")
    return ALIAS_MAP.get(token, token)


def parse_requirements(path: Path) -> set[str]:
    names: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        names.add(normalize_requirement_name(line))
    return names


def parse_pyproject_dependencies(path: Path) -> set[str]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    deps = data.get("project", {}).get("dependencies", [])
    return {normalize_requirement_name(dep) for dep in deps}


def parse_environment_dependencies(path: Path) -> set[str]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    out: set[str] = set()
    for dep in data.get("dependencies", []):
        if isinstance(dep, str):
            out.add(normalize_requirement_name(dep))
        elif isinstance(dep, dict) and "pip" in dep:
            for pip_dep in dep["pip"]:
                out.add(normalize_requirement_name(str(pip_dep)))
    out.discard("python")
    out.discard("pip")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify required repository artifacts and dependency alignment."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root path.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero on dependency mismatches as well as missing files.",
    )
    args = parser.parse_args()

    repo_root = args.repo_root
    lines: list[str] = []
    missing_paths: list[str] = []

    lines.append("REPOSITORY PATH CHECKS")
    for rel in REQUIRED_PATHS:
        path = repo_root / rel
        exists = path.exists()
        lines.append(f"{rel}: {'OK' if exists else 'MISSING'}")
        if not exists:
            missing_paths.append(rel)

    requirements_names = parse_requirements(repo_root / "requirements.txt")
    pyproject_names = parse_pyproject_dependencies(repo_root / "pyproject.toml")
    environment_names = parse_environment_dependencies(repo_root / "environment.yml")

    req_expected = DEPENDENCY_WHITELIST - requirements_names
    py_expected = DEPENDENCY_WHITELIST - pyproject_names
    env_expected = DEPENDENCY_WHITELIST - environment_names

    dep_mismatches = {
        "missing_from_requirements": sorted(req_expected),
        "missing_from_pyproject": sorted(py_expected),
        "missing_from_environment": sorted(env_expected),
        "requirements_only": sorted(requirements_names - pyproject_names - environment_names),
        "pyproject_only": sorted(pyproject_names - requirements_names - environment_names),
        "environment_only": sorted(environment_names - requirements_names - pyproject_names),
    }

    lines.append("")
    lines.append("DEPENDENCY ALIGNMENT")
    lines.append(f"requirements={sorted(requirements_names)}")
    lines.append(f"pyproject={sorted(pyproject_names)}")
    lines.append(f"environment={sorted(environment_names)}")
    for key, values in dep_mismatches.items():
        lines.append(f"{key}={values}")

    out_dir = repo_root / "results" / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    text_report = out_dir / "reproducibility_check.txt"
    json_report = out_dir / "reproducibility_check.json"

    text_report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json_report.write_text(
        json.dumps(
            {
                "missing_paths": missing_paths,
                "dependency_mismatches": dep_mismatches,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    has_dep_issue = any(dep_mismatches.values())
    if missing_paths:
        raise SystemExit(f"Missing required files or directories: {missing_paths}")
    if args.strict and has_dep_issue:
        raise SystemExit(
            "Dependency alignment mismatch detected. "
            "See results/logs/reproducibility_check.json"
        )


if __name__ == "__main__":
    main()
