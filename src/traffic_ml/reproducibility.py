"""Determinism helpers and run provenance."""
from __future__ import annotations

import json
import os
import platform
import random
import subprocess
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def write_provenance(path: str | Path, config: dict) -> None:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.SubprocessError):
        commit = "unavailable"
    info = {
        "git_commit": commit, "python": platform.python_version(),
        "platform": platform.platform(), "torch": torch.__version__,
        "cuda": torch.version.cuda, "config": config,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info, indent=2), encoding="utf-8")
