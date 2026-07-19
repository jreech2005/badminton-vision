"""Events contract: round-trip fidelity, checked-in schema freshness, real-data fit."""

import datetime as dt
import json

import pandas as pd
import pytest

from badminton_vision import paths
from badminton_vision.ingest.events_convert import shuttleset_to_events
from badminton_vision.ingest.labels import LabelMap
from badminton_vision.ingest.preoutcome import assign_sides
from badminton_vision.ingest.shuttleset import (
    SHUTTLESET_SPEC,
    load_match_table,
    load_strokes,
)
from badminton_vision.schemas.events_v1 import MatchEventFile
from tests.conftest import make_loader_frame

LABEL_YAML = paths.CONFIGS_DIR / "shot_labels_v1.yaml"
SCHEMA_JSON = paths.REPO_ROOT / "src/badminton_vision/schemas/events_v1.schema.json"


def _meta_row() -> pd.Series:
    return pd.Series(
        {
            "match_uid": "syn:1",
            "winner": "Alpha",
            "loser": "Beta",
            "date": dt.date(2021, 1, 15),
        }
    )


@pytest.mark.unit
def test_converter_round_trips_through_json() -> None:
    frame = make_loader_frame(n_rallies=2, strokes_per_rally=3)
    frame.loc[0, "type"] = "未知球種"  # unknown stroke must survive as shot_class None
    event_file = shuttleset_to_events(
        _meta_row(), frame, LabelMap.load(LABEL_YAML), seed=7, discipline="MS"
    )
    assert len(event_file.rallies) == 2
    assert sum(len(r.shots) for r in event_file.rallies) == 6
    assert event_file.rallies[0].shots[0].shot_class is None
    assert event_file.rallies[0].shots[0].shot_class_version == "shot_labels_v1"
    round_tripped = MatchEventFile.model_validate_json(event_file.model_dump_json())
    assert round_tripped == event_file


@pytest.mark.unit
def test_converter_outcome_matches_side_assignment() -> None:
    frame = make_loader_frame()
    meta = _meta_row()
    event_file = shuttleset_to_events(
        meta, frame, LabelMap.load(LABEL_YAML), seed=7, discipline="MS"
    )
    side0, _, y = assign_sides("syn:1", "Alpha", "Beta", seed=7)
    assert event_file.side0_player == side0
    assert event_file.outcome is not None
    assert event_file.outcome.winner_side == ("side0" if y == 1 else "side1")


@pytest.mark.unit
def test_checked_in_schema_is_current() -> None:
    checked_in = json.loads(SCHEMA_JSON.read_text())
    assert checked_in == MatchEventFile.model_json_schema(), (
        "events_v1.schema.json is stale; regenerate it from the model"
    )


@pytest.mark.integration
def test_converter_fits_one_real_match() -> None:
    table = load_match_table(SHUTTLESET_SPEC)
    meta = table.iloc[0]
    strokes = load_strokes(
        SHUTTLESET_SPEC, str(meta["match_uid"]), str(meta["video"]), int(meta["n_sets"])
    )
    event_file = shuttleset_to_events(
        meta, strokes, LabelMap.load(LABEL_YAML), seed=7, discipline="MS"
    )
    n_rallies = strokes.groupby(["set_no", "rally"]).ngroups
    assert len(event_file.rallies) == n_rallies
    assert sum(len(r.shots) for r in event_file.rallies) == len(strokes)
    assert MatchEventFile.model_validate_json(event_file.model_dump_json()) == event_file
