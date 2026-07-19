"""BadmintonDB policy: filename contract, winner extraction, curated override gates."""

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from badminton_vision import paths
from badminton_vision.errors import DataContractError
from badminton_vision.ingest.aliases import load_player_aliases
from badminton_vision.ingest.badminton_db import (
    _FILENAME,
    extract_winner,
    load_bdb_match_table,
    load_date_overrides,
)

OVERRIDES_CSV = paths.CURATED_DIR / "match_date_overrides.csv"
ALIASES_CSV = paths.CURATED_DIR / "player_aliases.csv"


def _point(t1_score: int, t1_point: int, t2_score: int, t2_point: int) -> dict[str, Any]:
    return {
        "T1P1": {"Score": t1_score, "Point": t1_point, "Status": "server", "Location": "top"},
        "T2P1": {"Score": t2_score, "Point": t2_point, "Status": "receiver", "Location": "bottom"},
    }


def _write_match(dir_: Path, filename: str, last_point: dict[str, Any]) -> Path:
    (dir_ / filename).write_text(json.dumps({"Point1": {"PointInfo": last_point}}))
    return dir_


@pytest.mark.unit
def test_filename_regex_parses_both_variants() -> None:
    plain = _FILENAME.match(
        "2018-38-5F-MS-Anthony Sinisuka GINTING (INA)-Kento MOMOTA (JPN)-Victor China Open-1080p.json"
    )
    fps50 = _FILENAME.match(
        "2019-43-3QF-MS-Anthony Sinisuka GINTING (INA)-Kento Momota (JPN)-YONEX French Open-1080p-50fps.json"
    )
    assert plain is not None
    assert plain["round"] == "F"
    assert plain["week"] == "38"
    assert fps50 is not None
    assert fps50["round"] == "QF"
    assert _FILENAME.match("not-a-match.json") is None


@pytest.mark.unit
def test_extract_winner_unique_candidate() -> None:
    assert extract_winner([_point(1, 20, 0, 15)]) == "T1P1"


@pytest.mark.unit
def test_extract_winner_none_when_no_candidate() -> None:
    # Mid-game state: nobody can end the match on this point.
    assert extract_winner([_point(0, 5, 0, 3)]) is None


@pytest.mark.unit
def test_extract_winner_none_when_ambiguous() -> None:
    # 29-29 in the deciding game: either player would end the match.
    assert extract_winner([_point(1, 29, 1, 29)]) is None


@pytest.mark.unit
def test_override_outside_week_requires_flag(tmp_path: Path) -> None:
    _write_match(
        tmp_path,
        "2018-27-2R16-MS-Alpha One (AAA)-Beta Two (BBB)-Some Open-1080p.json",
        _point(1, 20, 0, 15),
    )
    overrides = pd.DataFrame(
        {
            "match_uid": ["bdb:2018-27"],
            "date": ["2018-07-10"],  # ISO week 28, filename says 27
            "winner": ["Alpha One (AAA)"],
            "week_mismatch_ok": [float("nan")],
            "provenance_url": ["https://example.org"],
            "note": [""],
        }
    )
    with pytest.raises(DataContractError, match="week_mismatch_ok"):
        load_bdb_match_table(tmp_path, {}, overrides)
    overrides["week_mismatch_ok"] = "yes"
    table = load_bdb_match_table(tmp_path, {}, overrides)
    assert table.loc[0, "date_precision"] == "day"


@pytest.mark.unit
def test_override_winner_disagreement_raises(tmp_path: Path) -> None:
    _write_match(
        tmp_path,
        "2018-27-2R16-MS-Alpha One (AAA)-Beta Two (BBB)-Some Open-1080p.json",
        _point(1, 20, 0, 15),  # extraction says T1P1 = Alpha
    )
    overrides = pd.DataFrame(
        {
            "match_uid": ["bdb:2018-27"],
            "date": ["2018-07-05"],
            "winner": ["Beta Two (BBB)"],
            "week_mismatch_ok": [float("nan")],
            "provenance_url": ["https://example.org"],
            "note": [""],
        }
    )
    with pytest.raises(DataContractError, match="disagrees"):
        load_bdb_match_table(tmp_path, {}, overrides)


@pytest.mark.unit
def test_no_override_and_ambiguous_raises(tmp_path: Path) -> None:
    _write_match(
        tmp_path,
        "2018-27-2R16-MS-Alpha One (AAA)-Beta Two (BBB)-Some Open-1080p.json",
        _point(0, 5, 0, 3),
    )
    with pytest.raises(DataContractError, match="no curated override"):
        load_bdb_match_table(tmp_path, {}, load_date_overrides(OVERRIDES_CSV).iloc[0:0])


@pytest.mark.unit
def test_overrides_require_provenance(tmp_path: Path) -> None:
    bad = tmp_path / "overrides.csv"
    bad.write_text(
        "match_uid,date,winner,week_mismatch_ok,provenance_url,note\nx,2020-01-01,W,,,\n"
    )
    with pytest.raises(DataContractError, match="provenance"):
        load_date_overrides(bad)


@pytest.mark.integration
def test_real_bdb_matches() -> None:
    aliases = load_player_aliases(ALIASES_CSV)
    table = load_bdb_match_table(
        paths.BADMINTON_DB_JSON_DIR, aliases, load_date_overrides(OVERRIDES_CSV)
    )
    assert len(table) == 9
    assert (table["date_precision"] == "day").all()
    winners = table["winner"].value_counts()
    # Sourced results (see match_date_overrides.csv): Momota 7, Ginting 2.
    assert winners["Kento MOMOTA"] == 7
    assert winners["Anthony Sinisuka GINTING"] == 2
    names = set(table["winner"]) | set(table["loser"])
    assert names == {"Kento MOMOTA", "Anthony Sinisuka GINTING"}
    assert set(table["n_sets"]) <= {2, 3}
    assert table["match_uid"].is_unique
