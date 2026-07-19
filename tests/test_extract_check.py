"""Extract-check policy: verified counts and read-only lockdown are both enforced."""

import os
import stat
from pathlib import Path

import pytest

from badminton_vision import paths
from badminton_vision.errors import DataContractError
from badminton_vision.ingest.extract_check import (
    EXPECTED_BDB_JSONS,
    EXPECTED_SHUTTLESET,
    EXPECTED_SHUTTLESET22,
    check_raw,
)


def _build_fake_raw(root: Path) -> tuple[Path, Path, Path]:
    ss = root / "shuttleset" / "ShuttleSet" / "set"
    ss22 = root / "shuttleset22" / "CoachAI-Challenge-IJCAI2023" / "ShuttleSet22" / "set"
    bdb = root / "badminton_db" / "badminton-db" / "json"
    for set_dir, (n_dirs, n_csvs) in ((ss, EXPECTED_SHUTTLESET), (ss22, EXPECTED_SHUTTLESET22)):
        set_dir.mkdir(parents=True)
        (set_dir / "match.csv").write_text("id\n")
        extra = n_csvs - 2 * n_dirs  # first `extra` match dirs hold 3 sets, the rest 2
        for i in range(n_dirs):
            match_dir = set_dir / f"match_{i}"
            match_dir.mkdir()
            for set_no in range(1, (3 if i < extra else 2) + 1):
                (match_dir / f"set{set_no}.csv").write_text("rally\n")
    bdb.mkdir(parents=True)
    for i in range(EXPECTED_BDB_JSONS):
        (bdb / f"match_{i}.json").write_text("{}\n")
    return ss, ss22, bdb


def _chmod_tree(root: Path, *, writable: bool) -> None:
    file_mode = stat.S_IRUSR | (stat.S_IWUSR if writable else 0)
    dir_mode = file_mode | stat.S_IXUSR
    for dirpath, _dirnames, filenames in os.walk(root, topdown=False):
        for name in filenames:
            os.chmod(Path(dirpath) / name, file_mode)
        os.chmod(dirpath, dir_mode)


@pytest.mark.unit
def test_check_raw_passes_on_verified_counts(tmp_path: Path) -> None:
    ss, ss22, bdb = _build_fake_raw(tmp_path)
    _chmod_tree(tmp_path, writable=False)
    try:
        report = check_raw(tmp_path, ss, ss22, bdb)
    finally:
        _chmod_tree(tmp_path, writable=True)
    assert report.shuttleset == EXPECTED_SHUTTLESET
    assert report.shuttleset22 == EXPECTED_SHUTTLESET22
    assert report.bdb_jsons == EXPECTED_BDB_JSONS


@pytest.mark.unit
def test_check_raw_rejects_count_mismatch(tmp_path: Path) -> None:
    ss, ss22, bdb = _build_fake_raw(tmp_path)
    (ss / "match_0" / "set1.csv").unlink()
    _chmod_tree(tmp_path, writable=False)
    try:
        with pytest.raises(DataContractError, match="shuttleset counts"):
            check_raw(tmp_path, ss, ss22, bdb)
    finally:
        _chmod_tree(tmp_path, writable=True)


@pytest.mark.unit
def test_check_raw_rejects_writable_tree(tmp_path: Path) -> None:
    ss, ss22, bdb = _build_fake_raw(tmp_path)
    with pytest.raises(DataContractError, match="writable"):
        check_raw(tmp_path, ss, ss22, bdb)


@pytest.mark.integration
def test_extract_check_real_tree() -> None:
    report = check_raw(
        paths.RAW_DIR,
        paths.SHUTTLESET_SET_DIR,
        paths.SHUTTLESET22_SET_DIR,
        paths.BADMINTON_DB_JSON_DIR,
    )
    assert report.shuttleset == EXPECTED_SHUTTLESET
    assert report.shuttleset22 == EXPECTED_SHUTTLESET22
    assert report.bdb_jsons == EXPECTED_BDB_JSONS
