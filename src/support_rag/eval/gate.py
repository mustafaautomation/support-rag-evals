"""Compare metric averages with the thresholds in config/thresholds.yaml."""

from dataclasses import dataclass
from pathlib import Path

import yaml

from support_rag.paths import repo_root


@dataclass(frozen=True)
class GateResult:
    passed: bool
    failures: tuple[str, ...]
    scores: dict[str, float]
    thresholds: dict[str, float]


def load_thresholds(path: Path | None = None) -> dict[str, dict[str, float]]:
    threshold_path = path or (repo_root() / "config" / "thresholds.yaml")
    loaded = yaml.safe_load(threshold_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("thresholds.yaml must be a mapping")
    parsed: dict[str, dict[str, float]] = {}
    for section, values in loaded.items():
        if not isinstance(section, str) or not isinstance(values, dict):
            raise ValueError("Each threshold section must map metric names to numbers")
        parsed[section] = {str(name): float(limit) for name, limit in values.items()}
    for required in ("offline", "live"):
        if required not in parsed:
            raise ValueError(f"thresholds.yaml is missing the {required} section")
    return parsed


def apply_gate(scores: dict[str, float], thresholds: dict[str, float]) -> GateResult:
    failures: list[str] = []
    for name, limit in thresholds.items():
        if name not in scores:
            failures.append(f"missing metric {name}")
            continue
        if scores[name] < limit:
            failures.append(f"{name}={scores[name]:.3f} < {limit:.3f}")
    return GateResult(
        passed=not failures,
        failures=tuple(failures),
        scores=dict(scores),
        thresholds=dict(thresholds),
    )
