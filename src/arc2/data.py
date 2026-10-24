"""Load ARC-AGI-2 competition JSON files."""

from __future__ import annotations

import json
from pathlib import Path

Grid = list[list[int]]

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
KAGGLE_DIR = Path("/kaggle/input/arc-prize-2026-arc-agi-2")

SPLITS = ("training", "evaluation", "test")


def data_dir() -> Path:
    """Kaggle input dir when running on Kaggle, else the repo-local data/."""
    return KAGGLE_DIR if KAGGLE_DIR.exists() else DATA_DIR


def load_challenges(split: str, root: Path | None = None) -> dict:
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    root = root or data_dir()
    return json.loads((root / f"arc-agi_{split}_challenges.json").read_text())


def load_solutions(split: str, root: Path | None = None) -> dict[str, list[Grid]]:
    if split == "test":
        raise ValueError("test solutions are hidden")
    root = root or data_dir()
    return json.loads((root / f"arc-agi_{split}_solutions.json").read_text())
