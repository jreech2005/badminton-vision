"""Club records policy: rulebook gates, name guard, quarantining import, harness fit."""

import datetime as dt
import logging
from pathlib import Path

import pytest

from badminton_vision import paths
from badminton_vision.cli import main
from badminton_vision.errors import ValidationGateError
from badminton_vision.eval.harness import load_harness_config, run_walk_forward
from badminton_vision.eval.predictors import CoinFlip, EloPredictor
from badminton_vision.records.club_db import (
    ClubDB,
    NewMatch,
    validate_game_score,
    validate_match,
)

pytestmark = pytest.mark.unit

TODAY = dt.date(2026, 7, 19)


def _match(**overrides: object) -> NewMatch:
    base: dict[str, object] = {
        "match_type": "singles",
        "played_at": dt.date(2026, 7, 1),
        "side_a": ("Viraj",),
        "side_b": ("Opponent",),
        "winner_side": "A",
        "games": ((21, 15), (21, 18)),
    }
    base.update(overrides)
    return NewMatch(**base)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("side_a", "side_b"),
    [(21, 20), (31, 5), (22, 19), (30, 27), (19, 3), (21, 21), (20, 21)],
)
def test_illegal_game_scores_rejected(side_a: int, side_b: int) -> None:
    with pytest.raises(ValidationGateError, match="illegal game score"):
        validate_game_score(side_a, side_b)


@pytest.mark.parametrize(
    ("side_a", "side_b"),
    [(21, 19), (21, 0), (30, 29), (30, 28), (25, 23), (15, 21), (22, 20), (20, 22)],
)
def test_legal_game_scores_accepted(side_a: int, side_b: int) -> None:
    validate_game_score(side_a, side_b)


def test_self_play_rejected() -> None:
    with pytest.raises(ValidationGateError, match="appears twice"):
        validate_match(_match(side_b=("viraj",)), TODAY)  # case-insensitive twin


def test_future_date_rejected() -> None:
    with pytest.raises(ValidationGateError, match="future"):
        validate_match(_match(played_at=TODAY + dt.timedelta(days=1)), TODAY)


def test_prehistoric_date_rejected() -> None:
    with pytest.raises(ValidationGateError, match="records era"):
        validate_match(_match(played_at=dt.date(2019, 1, 1)), TODAY)


def test_winner_must_take_two_games() -> None:
    with pytest.raises(ValidationGateError, match="won 1 games"):
        validate_match(_match(games=((21, 15), (15, 21)), winner_side="A"), TODAY)


def test_winner_must_take_final_game() -> None:
    with pytest.raises(ValidationGateError, match="final game"):
        validate_match(_match(games=((21, 15), (21, 18), (15, 21)), winner_side="A"), TODAY)


def test_game_count_bounds() -> None:
    with pytest.raises(ValidationGateError, match="2 or 3 games"):
        validate_match(_match(games=((21, 15),)), TODAY)


def test_doubles_requires_two_per_side() -> None:
    with pytest.raises(ValidationGateError, match="2 player"):
        validate_match(_match(match_type="doubles"), TODAY)


def _db(tmp_path: Path) -> ClubDB:
    db = ClubDB(tmp_path / "club.sqlite")
    db.init_schema()
    return db


def test_name_guard_fires_and_can_be_forced(tmp_path: Path) -> None:
    db = _db(tmp_path)
    db.add_player("Samuel Lee")
    with pytest.raises(ValidationGateError, match="suspiciously close"):
        db.add_player("Samuel Le")
    assert db.add_player("Samuel Le", force_new=True) > 0


def test_exact_duplicate_player_rejected(tmp_path: Path) -> None:
    db = _db(tmp_path)
    db.add_player("Viraj")
    with pytest.raises(ValidationGateError, match="already exists"):
        db.add_player("viraj", force_new=True)


def test_unknown_player_in_match_rejected(tmp_path: Path) -> None:
    db = _db(tmp_path)
    db.add_player("Viraj")
    with pytest.raises(ValidationGateError, match="unknown player"):
        db.add_match(_match(), today=TODAY)


def test_same_day_rematch_warns_but_records(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    db = _db(tmp_path)
    db.add_player("Viraj")
    db.add_player("Opponent")
    db.add_match(_match(), today=TODAY)
    with caplog.at_level(logging.WARNING):
        second = db.add_match(_match(), today=TODAY)
    assert second > 0
    assert any("possible duplicate" in r.message for r in caplog.records)


def test_import_csv_quarantines_bad_rows(tmp_path: Path) -> None:
    csv_path = tmp_path / "matches.csv"
    csv_path.write_text(
        "match_type,played_at,side_a,side_b,winner_side,games\n"
        'singles,2026-07-01,Ann,Bea,A,"21-15,21-18"\n'
        'singles,2026-07-02,Ann,Bea,B,"15-21,21-20,21-19"\n'  # 21-20 is illegal
        'singles,2026-07-03,Cid,Dee,A,"21-19,19-21,21-12"\n'
    )
    db = _db(tmp_path)
    report = db.import_csv(csv_path, today=TODAY)
    assert report.added == 2
    assert len(report.rejected) == 1
    assert report.rejected[0][0] == 3  # csv line number of the bad row
    assert "illegal game score" in report.rejected[0][1]


def test_revalidate_catches_external_corruption(tmp_path: Path) -> None:
    db = _db(tmp_path)
    db.add_player("Viraj")
    db.add_player("Opponent")
    match_id = db.add_match(_match(), today=TODAY)
    assert db.revalidate(today=TODAY) == []
    db._conn.execute(
        "UPDATE match_games SET side_a_score = 25, side_b_score = 10 "
        "WHERE match_id = ? AND game_number = 1",
        (match_id,),
    )
    violations = db.revalidate(today=TODAY)
    assert len(violations) == 1
    assert violations[0][0] == match_id


def test_doubles_stored_but_not_in_timeline(tmp_path: Path) -> None:
    db = _db(tmp_path)
    for name in ("A1", "A2", "B1", "B2"):
        db.add_player(name)
    db.add_match(
        _match(
            match_type="doubles",
            side_a=("A1", "A2"),
            side_b=("B1", "B2"),
            games=((21, 15), (21, 18)),
        ),
        today=TODAY,
    )
    assert db.as_timeline().empty


def test_harness_runs_unchanged_on_club_fixture(tmp_path: Path) -> None:
    db = _db(tmp_path)
    players = ["P1", "P2", "P3", "P4", "P5", "P6"]
    for name in players:
        db.add_player(name)
    for i in range(30):
        side_a = players[i % 6]
        side_b = players[(i + 1 + i % 5) % 6]
        if side_a == side_b:
            side_b = players[(i + 2) % 6]
        winner_side = "A" if i % 3 else "B"
        games = ((21, 15), (21, 18)) if winner_side == "A" else ((15, 21), (18, 21))
        db.add_match(
            _match(
                played_at=dt.date(2026, 1, 1) + dt.timedelta(days=i),
                side_a=(side_a,),
                side_b=(side_b,),
                winner_side=winner_side,
                games=games,
            ),
            today=TODAY,
        )
    timeline = db.as_timeline()
    assert len(timeline) == 30
    assert list(timeline["t_idx"]) == list(range(30))
    config = load_harness_config(paths.CONFIGS_DIR / "harness_v1.yaml")
    result = run_walk_forward(timeline, [CoinFlip(), EloPredictor(config.elo)], config)
    coin = result.per_match[result.per_match["predictor"] == "coinflip"]
    assert (coin["brier"] == 0.25).all()
    assert len(result.per_match) == 60


def test_records_cli_end_to_end(tmp_path: Path) -> None:
    db_path = str(tmp_path / "club.sqlite")
    assert main(["records", "init", "--db", db_path]) == 0
    assert main(["records", "add-player", "Viraj", "--db", db_path]) == 0
    assert main(["records", "add-player", "Opponent", "--db", db_path]) == 0
    assert (
        main(
            [
                "records",
                "add-match",
                "--db",
                db_path,
                "--type",
                "singles",
                "--date",
                "2026-07-01",
                "--side-a",
                "Viraj",
                "--side-b",
                "Opponent",
                "--games",
                "21-15,19-21,21-18",
                "--winner",
                "A",
            ]
        )
        == 0
    )
    assert main(["records", "validate", "--db", db_path]) == 0
    # A gate failure surfaces as exit code 1, not a traceback.
    assert (
        main(
            [
                "records",
                "add-match",
                "--db",
                db_path,
                "--type",
                "singles",
                "--date",
                "2026-07-01",
                "--side-a",
                "Viraj",
                "--side-b",
                "Viraj",
                "--games",
                "21-15,21-18",
                "--winner",
                "A",
            ]
        )
        == 1
    )
