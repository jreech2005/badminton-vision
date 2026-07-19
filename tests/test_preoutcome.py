"""Leak-firewall policy: exact strip list, side re-keying, deterministic assignment."""

import pytest

from badminton_vision.errors import DataContractError
from badminton_vision.ingest.preoutcome import (
    KEPT_COLUMNS,
    STRIPPED_COLUMNS,
    assign_sides,
    historical_view,
    pre_outcome_view,
)
from badminton_vision.ingest.shuttleset import PROVENANCE_COLUMNS
from tests.conftest import make_loader_frame

pytestmark = pytest.mark.unit


def test_strip_list_is_exact() -> None:
    view = pre_outcome_view(make_loader_frame(), "Alpha", "Beta", side0="Alpha")
    # Policy test: the output columns are exactly side + kept + provenance.
    assert set(view.columns) == {"side", *KEPT_COLUMNS, *PROVENANCE_COLUMNS}
    assert not (STRIPPED_COLUMNS & set(view.columns))


def test_side_rekey_follows_assignment() -> None:
    frame = make_loader_frame()
    winner_first = pre_outcome_view(frame, "Alpha", "Beta", side0="Alpha")
    loser_first = pre_outcome_view(frame, "Alpha", "Beta", side0="Beta")
    a_rows = frame["player"] == "A"
    assert (winner_first.loc[list(a_rows), "side"] == "side0").all()
    assert (loser_first.loc[list(a_rows), "side"] == "side1").all()


def test_pre_outcome_view_rejects_unknown_side0() -> None:
    with pytest.raises(DataContractError, match="neither winner nor loser"):
        pre_outcome_view(make_loader_frame(), "Alpha", "Beta", side0="Gamma")


def test_views_reject_non_ab_player_values() -> None:
    frame = make_loader_frame()
    frame.loc[0, "player"] = "C"
    with pytest.raises(DataContractError, match="other than A/B"):
        pre_outcome_view(frame, "Alpha", "Beta", side0="Alpha")
    with pytest.raises(DataContractError, match="other than A/B"):
        historical_view(frame, "Alpha", "Beta")


def test_views_reject_non_loader_frames() -> None:
    with pytest.raises(DataContractError, match="loader stroke frame"):
        pre_outcome_view(
            make_loader_frame().drop(columns=["set_no"]), "Alpha", "Beta", side0="Alpha"
        )


def test_historical_view_rekeys_players_to_names() -> None:
    view = historical_view(make_loader_frame(), "Alpha", "Beta")
    assert set(view["player"]) == {"Alpha", "Beta"}
    assert set(view["getpoint_player"].dropna()) == {"Alpha"}


def test_assign_sides_deterministic() -> None:
    first = assign_sides("ss:7", "Alpha", "Beta", seed=11)
    assert first == assign_sides("ss:7", "Alpha", "Beta", seed=11)
    side0, side1, y = first
    assert {side0, side1} == {"Alpha", "Beta"}
    assert y == (1 if side0 == "Alpha" else 0)


def test_assign_sides_is_balanced_and_seed_sensitive() -> None:
    uids = [f"ss:{i}" for i in range(200)]
    ys_seed_a = [assign_sides(u, "W", "L", seed=1)[2] for u in uids]
    ys_seed_b = [assign_sides(u, "W", "L", seed=2)[2] for u in uids]
    # A fair keyed coin over 200 matches: both outcomes occur, no gross skew.
    assert 60 < sum(ys_seed_a) < 140
    assert ys_seed_a != ys_seed_b


def test_assignment_flip_flips_label_and_sides() -> None:
    frame = make_loader_frame()
    for side0, y in (("Alpha", 1), ("Beta", 0)):
        view = pre_outcome_view(frame, "Alpha", "Beta", side0=side0)
        expected = "side0" if side0 == "Alpha" else "side1"
        assert (view.loc[list(frame["player"] == "A"), "side"] == expected).all()
        assert (side0 == "Alpha") == bool(y)
