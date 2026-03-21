from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = REPO_ROOT / 'datasets'
ARTIFACTS_DIR = REPO_ROOT / 'artifacts'
FIGURES_DIR = REPO_ROOT / 'figures'
MANIFESTS_DIR = REPO_ROOT / 'manifests'
PAPER_DIR = REPO_ROOT / 'paper'
REPLAYS_DIR = REPO_ROOT / 'replays'


def ensure_standard_dirs() -> None:
    for path in (DATASETS_DIR, ARTIFACTS_DIR, FIGURES_DIR, MANIFESTS_DIR, PAPER_DIR):
        path.mkdir(parents=True, exist_ok=True)


def dataset_path(filename: str) -> str:
    return str(DATASETS_DIR / filename)


def artifact_path(filename: str) -> str:
    ensure_standard_dirs()
    return str(ARTIFACTS_DIR / filename)


def figure_path(filename: str) -> str:
    ensure_standard_dirs()
    return str(FIGURES_DIR / filename)


def replay_path(*parts: str) -> str:
    return str(REPLAYS_DIR.joinpath(*parts))
