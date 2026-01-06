from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_data_dir() -> Path:
    return repo_root() / "data"


def default_processed_dir() -> Path:
    return default_data_dir() / "processed"


def default_outputs_dir() -> Path:
    return repo_root() / "outputs"
