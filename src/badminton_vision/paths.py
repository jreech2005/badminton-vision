"""Filesystem layout: the single source of truth for where data lives."""

from pathlib import Path
from typing import Final

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DATA_DIR: Final[Path] = REPO_ROOT / "data"
RAW_DIR: Final[Path] = DATA_DIR / "raw"
CURATED_DIR: Final[Path] = DATA_DIR / "curated"
DERIVED_DIR: Final[Path] = DATA_DIR / "derived"
RUNS_DIR: Final[Path] = DERIVED_DIR / "runs"
DB_DIR: Final[Path] = REPO_ROOT / "db"
CONFIGS_DIR: Final[Path] = REPO_ROOT / "configs"

# Dataset roots as the archives extract them (zip-internal directory names).
SHUTTLESET_SET_DIR: Final[Path] = RAW_DIR / "shuttleset" / "ShuttleSet" / "set"
SHUTTLESET22_SET_DIR: Final[Path] = (
    RAW_DIR / "shuttleset22" / "CoachAI-Challenge-IJCAI2023" / "ShuttleSet22" / "set"
)
BADMINTON_DB_JSON_DIR: Final[Path] = RAW_DIR / "badminton_db" / "badminton-db" / "json"
