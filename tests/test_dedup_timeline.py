"""Dedup and timeline policy: twin detection, strict order, discipline inference."""

import datetime as dt

import pandas as pd
import pytest

from badminton_vision import paths
from badminton_vision.errors import DataContractError
from badminton_vision.ingest.aliases import load_player_aliases
from badminton_vision.ingest.badminton_db import load_bdb_match_table, load_date_overrides
from badminton_vision.ingest.dedup import dedup_matches
from badminton_vision.ingest.shuttleset import (
    MATCH_TABLE_COLUMNS,
    SHUTTLESET22_SPEC,
    SHUTTLESET_SPEC,
    load_match_table,
)
from badminton_vision.ingest.timeline import build_timeline, infer_discipline


def _match(
    uid: str,
    source: str,
    winner: str,
    loser: str,
    date: dt.date,
    *,
    precision: str = "day",
    round_name: str = "Finals",
) -> dict[str, object]:
    return {
        "match_uid": uid,
        "source": source,
        "video": uid,
        "date": date,
        "date_precision": precision,
        "tournament": "T",
        "round_name": round_name,
        "winner": winner,
        "loser": loser,
        "n_sets": 2,
        "duration_min": None,
    }


def _frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=list(MATCH_TABLE_COLUMNS))


D = dt.date(2021, 1, 15)


@pytest.mark.unit
def test_dedup_drops_lower_priority_source() -> None:
    frame = _frame(
        [
            _match("ss:1", "shuttleset", "W", "L", D),
            _match("ss22:9", "shuttleset22", "W", "L", D),
        ]
    )
    unique, dropped, _ = dedup_matches(frame)
    assert list(unique["match_uid"]) == ["ss:1"]
    assert list(dropped["match_uid"]) == ["ss22:9"]
    assert list(dropped["kept_match_uid"]) == ["ss:1"]


@pytest.mark.unit
def test_dedup_conflicting_winner_raises() -> None:
    frame = _frame(
        [
            _match("ss:1", "shuttleset", "W", "L", D),
            _match("ss22:9", "shuttleset22", "L", "W", D),
        ]
    )
    with pytest.raises(DataContractError, match="disagrees on the winner"):
        dedup_matches(frame)


@pytest.mark.unit
def test_week_precision_rows_never_join_exact_key() -> None:
    frame = _frame(
        [
            _match("ss:1", "shuttleset", "W", "L", D),
            _match("bdb:x", "badminton_db", "W", "L", D, precision="week"),
        ]
    )
    unique, dropped, _ = dedup_matches(frame)
    assert len(unique) == 2
    assert dropped.empty


@pytest.mark.unit
def test_near_miss_audit_flags_close_rematches() -> None:
    frame = _frame(
        [
            _match("ss:1", "shuttleset", "W", "L", D),
            _match("ss:2", "shuttleset", "L", "W", D + dt.timedelta(days=2)),
            _match("ss:3", "shuttleset", "W", "L", D + dt.timedelta(days=30)),
        ]
    )
    _, _, near = dedup_matches(frame)
    assert len(near) == 1
    assert near.loc[0, "gap_days"] == 2


@pytest.mark.unit
def test_dedup_unknown_source_raises() -> None:
    with pytest.raises(DataContractError, match="priority"):
        dedup_matches(_frame([_match("x:1", "mystery", "W", "L", D)]))


@pytest.mark.unit
def test_infer_discipline_two_components() -> None:
    frame = _frame(
        [
            _match("a", "shuttleset", "M1", "M2", D),
            _match("b", "shuttleset", "M2", "M3", D + dt.timedelta(days=40)),
            _match("c", "shuttleset", "F1", "F2", D),
        ]
    )
    labels = infer_discipline(frame, seeds={"M1": "MS", "F1": "WS"})
    assert labels == {"M1": "MS", "M2": "MS", "M3": "MS", "F1": "WS", "F2": "WS"}


@pytest.mark.unit
def test_infer_discipline_missing_seed_raises() -> None:
    frame = _frame([_match("a", "shuttleset", "M1", "M2", D)])
    with pytest.raises(DataContractError, match="expected exactly one"):
        infer_discipline(frame, seeds={"X": "MS"})


@pytest.mark.unit
def test_build_timeline_orders_by_date_then_round() -> None:
    frame = _frame(
        [
            _match("final", "shuttleset", "M1", "M2", D, round_name="Finals"),
            _match("semi", "shuttleset", "M1", "M3", D, round_name="Semi-finals"),
            _match("earlier", "shuttleset", "M2", "M3", D - dt.timedelta(days=1)),
        ]
    )
    timeline = build_timeline(frame, seeds={"M1": "MS"})
    assert list(timeline["match_uid"]) == ["earlier", "semi", "final"]
    assert list(timeline["t_idx"]) == [0, 1, 2]
    assert set(timeline["discipline"]) == {"MS"}


@pytest.mark.unit
def test_build_timeline_unknown_round_raises() -> None:
    frame = _frame([_match("a", "shuttleset", "M1", "M2", D, round_name="Round-of-99")])
    with pytest.raises(DataContractError, match="without a rank"):
        build_timeline(frame)


@pytest.mark.unit
def test_build_timeline_duplicate_uid_raises() -> None:
    frame = _frame(
        [_match("a", "shuttleset", "M1", "M2", D), _match("a", "shuttleset", "M1", "M2", D)]
    )
    with pytest.raises(DataContractError, match="duplicate match_uid"):
        build_timeline(frame)


@pytest.mark.integration
def test_real_global_timeline_is_103_matches() -> None:
    aliases = load_player_aliases(paths.CURATED_DIR / "player_aliases.csv")
    overrides = load_date_overrides(paths.CURATED_DIR / "match_date_overrides.csv")
    all_matches = pd.concat(
        [
            load_match_table(SHUTTLESET_SPEC, aliases),
            load_match_table(SHUTTLESET22_SPEC, aliases),
            load_bdb_match_table(paths.BADMINTON_DB_JSON_DIR, aliases, overrides),
        ],
        ignore_index=True,
    )
    assert len(all_matches) == 44 + 58 + 9
    unique, dropped, near = dedup_matches(all_matches)
    # The 8 known ss<->ss22 twin videos; priority keeps the shuttleset row.
    assert len(dropped) == 8
    assert set(dropped["source"]) == {"shuttleset22"}
    assert not (dropped["source"] == "badminton_db").any()
    assert len(unique) == 103

    timeline = build_timeline(unique)
    assert len(timeline) == 103
    assert timeline["date"].min() == dt.date(2018, 6, 29)
    assert timeline["date"].max() == dt.date(2022, 10, 30)
    assert list(timeline["t_idx"]) == list(range(103))
    assert set(timeline["discipline"]) == {"MS", "WS"}
    # Free consistency check: winner and loser always share a discipline (asserted
    # inside build_timeline); near-miss audit exists for manual review.
    assert {"match_uid_a", "gap_days"} <= set(near.columns)
