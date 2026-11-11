"""Locate repository data files when commands run from a checkout."""

from pathlib import Path


def repo_root() -> Path:
    """Return the checkout root that holds pyproject.toml and data/."""
    start = Path.cwd().resolve()
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "data" / "corpus").is_dir():
            return candidate
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "pyproject.toml").is_file() and (candidate / "data" / "corpus").is_dir():
            return candidate
    raise FileNotFoundError("Could not find the repository root from the working directory.")
