"""Verify the raw-data extraction: file counts and read-only lockdown."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from badminton_vision.errors import DataContractError

# Counts verified by direct archive inspection on 2026-07-19 (increment plan, Context).
EXPECTED_SHUTTLESET: Final[tuple[int, int]] = (44, 104)  # (match dirs, set CSVs)
EXPECTED_SHUTTLESET22: Final[tuple[int, int]] = (58, 140)
EXPECTED_BDB_JSONS: Final[int] = 9


@dataclass(frozen=True)
class RawCheckReport:
    """Observed raw-tree counts; constructed only after every check passed."""

    shuttleset: tuple[int, int]
    shuttleset22: tuple[int, int]
    bdb_jsons: int


def _count_set_tree(set_dir: Path) -> tuple[int, int]:
    if not set_dir.is_dir():
        raise DataContractError(f"missing dataset directory: {set_dir}")
    match_dirs = [p for p in set_dir.iterdir() if p.is_dir()]
    set_csvs = [p for d in match_dirs for p in d.glob("set*.csv")]
    return len(match_dirs), len(set_csvs)


def _writable_paths(root: Path) -> list[Path]:
    writable: list[Path] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for candidate in (Path(dirpath), *(Path(dirpath) / name for name in filenames)):
            if os.access(candidate, os.W_OK):
                writable.append(candidate)
    return writable


def check_raw(
    raw_dir: Path,
    shuttleset_set_dir: Path,
    shuttleset22_set_dir: Path,
    bdb_json_dir: Path,
) -> RawCheckReport:
    """Raise DataContractError unless the raw tree has the verified counts and is read-only."""
    shuttleset = _count_set_tree(shuttleset_set_dir)
    shuttleset22 = _count_set_tree(shuttleset22_set_dir)
    if not bdb_json_dir.is_dir():
        raise DataContractError(f"missing dataset directory: {bdb_json_dir}")
    bdb_jsons = len(list(bdb_json_dir.glob("*.json")))

    problems: list[str] = []
    if shuttleset != EXPECTED_SHUTTLESET:
        problems.append(f"shuttleset counts {shuttleset}, expected {EXPECTED_SHUTTLESET}")
    if shuttleset22 != EXPECTED_SHUTTLESET22:
        problems.append(f"shuttleset22 counts {shuttleset22}, expected {EXPECTED_SHUTTLESET22}")
    if bdb_jsons != EXPECTED_BDB_JSONS:
        problems.append(f"badminton_db json count {bdb_jsons}, expected {EXPECTED_BDB_JSONS}")
    writable = _writable_paths(raw_dir)
    if writable:
        problems.append(f"{len(writable)} writable paths under {raw_dir} (first: {writable[0]})")
    if problems:
        raise DataContractError("; ".join(problems))
    return RawCheckReport(shuttleset=shuttleset, shuttleset22=shuttleset22, bdb_jsons=bdb_jsons)
