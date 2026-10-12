"""Configuration loading and validation. Unknown keys are retained for extensibility."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    required = {"project", "data", "split", "model", "training", "evaluation"}
    missing = required - set(cfg)
    if missing:
        raise ValueError(f"Missing config sections: {sorted(missing)}")
    fractions = [cfg["split"][f] for f in (
        "train_fraction", "calibration_fraction", "validation_fraction", "test_fraction"
    )]
    if abs(sum(fractions) - 1.0) > 1e-8 or any(x <= 0 for x in fractions):
        raise ValueError("Split fractions must be positive and sum to one")
    if len(set(cfg["project"]["seeds"])) < 3:
        raise ValueError("Research runs require at least three distinct seeds")
    return cfg
