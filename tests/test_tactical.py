"""Tactical-feature policy: hand-computed counters, decay, support gates, invariance."""

import datetime as dt
import math
from typing import Any

import pandas as pd
import pytest

from badminton_vision import paths
from badminton_vision.features.tactical import (
    TACTICAL_COLUMNS,
    TacticalParams,
    TacticalState,
    build_tactical_rows,
    match_counters,
)
from badminton_vision.ingest.labels import LabelMap
from badminton_vision.ingest.preoutcome import historical_view
from badminton_vision.ingest.shuttleset import STROKE_COLUMNS
from tests.conftest import cell, make_loader_frame, make_synthetic_timeline

LABELS = LabelMap.load(paths.CONFIGS_DIR / "shot_labels_v1.yaml")
PARAMS = TacticalParams(
    half_life_days=10.0,
    min_weighted_strokes=3.0,
    min_weighted_serves=1.0,
    shrink_strokes=1.0,
    shrink_serves=1.0,
    long_rally_min=9,
    short_rally_max=4,
)
D0 = dt.date(2021, 1, 1)


def _loader_frame(rallies: list[tuple[list[tuple[str, str]], str]]) -> pd.DataFrame:
    """Build a loader stroke frame from [(strokes [(player_ab, type_zh)], winner_ab)]."""
    rows: list[dict[str, object]] = []
    nan = float("nan")
    for rally_no, (strokes, rally_winner) in enumerate(rallies, start=1):
        for ball_round, (player, type_zh) in enumerate(strokes, start=1):
            is_last = ball_round == len(strokes)
            rows.append(
                {
                    "rally": rally_no,
                    "ball_round": ball_round,
                    "time": "00:00:01",
                    "frame_num": rally_no * 100 + ball_round,
                    "roundscore_A": rally_no,
                    "roundscore_B": rally_no,
                    "player": player,
                    "server": 2,
                    "type": type_zh,
                    "aroundhead": 0,
                    "backhand": 0,
                    "hit_height": 1,
                    "hit_area": 1,
                    "hit_x": nan,
                    "hit_y": nan,
                    "landing_height": 1,
                    "landing_area": 1,
                    "landing_x": nan,
                    "landing_y": nan,
                    "lose_reason": nan,
                    "win_reason": nan,
                    "getpoint_player": rally_winner if is_last else nan,
                    "flaw": nan,
                    "player_location_area": 1,
                    "player_location_x": nan,
                    "player_location_y": nan,
                    "opponent_location_area": 1,
                    "opponent_location_x": nan,
                    "opponent_location_y": nan,
                    "db": nan,
                }
            )
    frame = pd.DataFrame(rows, columns=list(STROKE_COLUMNS))
    frame["source"] = "synthetic"
    frame["match_uid"] = "syn:1"
    frame["video"] = "Synthetic_Match"
    frame["set_no"] = 1
    return frame


def _counters(rallies: list[tuple[list[tuple[str, str]], str]]) -> dict[str, Any]:
    view = historical_view(_loader_frame(rallies), "W", "L")
    return dict(match_counters(view, "W", "L", LABELS, PARAMS))


@pytest.mark.unit
def test_match_counters_hand_computed() -> None:
    counters = _counters(
        [
            # W serves short, L nets back, W kills with a smash. W wins the rally.
            ([("A", "發短球"), ("B", "放小球"), ("A", "殺球")], "A"),
            # L serves long, W's net shot fails. L wins the rally.
            ([("B", "發長球"), ("A", "放小球")], "B"),
        ]
    )
    w = counters["W"]
    lo = counters["L"]
    assert w.strokes_total == 3
    assert w.smash_strokes == 1
    assert w.smash_kills == 1
    assert w.serves == 1
    assert w.short_serves == 1
    assert w.serve_rallies == 1
    assert w.serve_rally_wins == 1
    assert w.receive_rallies == 1
    assert w.receive_rally_wins == 0
    assert w.third_stroke_chances == 1
    assert w.third_stroke_attacks == 1
    assert w.last_stroke_losses == 1  # rally 2's failed net shot
    assert w.net_last_losses == 1
    assert w.rally_lengths == [3, 2]
    assert lo.strokes_total == 2
    assert lo.net_strokes == 1
    assert lo.long_serves == 1
    assert lo.serve_rally_wins == 1
    assert lo.last_stroke_losses == 0


@pytest.mark.unit
def test_unknown_strokes_leave_class_rates_but_keep_rallies() -> None:
    counters = _counters([([("A", "發短球"), ("B", "未知球種"), ("A", "未知球種")], "A")])
    w = counters["W"]
    assert w.strokes_total == 2  # unknown still a stroke
    assert w.classified_strokes == 1  # ...but not a classified one
    assert w.rally_lengths == [3]  # rally-level features keep the rally
    assert w.smash_kills == 0  # unknown final stroke cannot be a smash kill


@pytest.mark.unit
def test_single_stroke_rally_is_a_serve_fault() -> None:
    counters = _counters([([("A", "發短球")], "B")])
    assert counters["W"].serve_faults == 1
    assert counters["L"].receive_rally_wins == 1


@pytest.mark.unit
def test_state_supports_gate_then_decays_below_support() -> None:
    state = TacticalState(LABELS, PARAMS)
    frame = _loader_frame(
        [
            ([("A", "發短球"), ("B", "放小球"), ("A", "殺球")], "A"),
            ([("B", "發長球"), ("A", "放小球")], "B"),
        ]
    )
    state.observe_match(D0, "W", "L", frame)
    at_d0 = state.features("W", D0)
    # 3 weighted strokes meets min support; smash_kill_rate = (1 + 1*prior) / (1 + 1)
    # with pooled prior = pooled kills / pooled smashes = 1/1.
    assert at_d0["smash_kill_rate"] == pytest.approx(1.0)
    # stroke_error_rate: W num 1 den 3; pooled num 1 den 5 (both players) -> prior 0.2.
    assert at_d0["stroke_error_rate"] == pytest.approx((1 + 1 * 0.2) / (3 + 1))
    assert at_d0["median_rally_len"] == pytest.approx(2.0)
    # Two half-lives later the 3 strokes weigh 0.75 < min support: everything NaN.
    at_d20 = state.features("W", D0 + dt.timedelta(days=20))
    assert all(math.isnan(v) for v in at_d20.values())


@pytest.mark.unit
def test_unseen_player_is_all_nan() -> None:
    state = TacticalState(LABELS, PARAMS)
    values = state.features("Nobody", D0)
    assert len(values) == 13
    assert all(math.isnan(v) for v in values.values())


def _strokes_for(timeline: pd.DataFrame) -> dict[str, pd.DataFrame]:
    frame = _loader_frame(
        [
            ([("A", "發短球"), ("B", "放小球"), ("A", "殺球")], "A"),
            ([("B", "發長球"), ("A", "放小球")], "B"),
            ([("A", "發短球"), ("B", "挑球"), ("A", "切球"), ("B", "推球")], "B"),
        ]
    )
    return {str(uid): frame for uid in timeline["match_uid"]}


@pytest.mark.unit
def test_build_tactical_rows_prefix_invariance() -> None:
    timeline = make_synthetic_timeline(12)
    strokes = _strokes_for(timeline)
    full = build_tactical_rows(timeline, strokes, LABELS, PARAMS, seed=7)
    prefix = build_tactical_rows(timeline.head(6), strokes, LABELS, PARAMS, seed=7)
    pd.testing.assert_frame_equal(full.head(6).reset_index(drop=True), prefix)


@pytest.mark.unit
def test_build_tactical_rows_antisymmetry() -> None:
    timeline = make_synthetic_timeline(16)
    strokes = _strokes_for(timeline)
    first = build_tactical_rows(timeline, strokes, LABELS, PARAMS, seed=1)
    second = build_tactical_rows(timeline, strokes, LABELS, PARAMS, seed=2)
    flipped = first["side0"] != second["side0"]
    assert flipped.any()
    for i in first.index[flipped]:
        for column in TACTICAL_COLUMNS:
            a, b = cell(first, int(i), column), cell(second, int(i), column)
            if math.isnan(a):
                assert math.isnan(b)
            else:
                assert b == pytest.approx(-a), f"{column} did not negate on side swap"


@pytest.mark.unit
def test_first_row_has_no_tactical_information() -> None:
    timeline = make_synthetic_timeline(6)
    rows = build_tactical_rows(timeline, _strokes_for(timeline), LABELS, PARAMS, seed=7)
    assert all(math.isnan(cell(rows, 0, column)) for column in TACTICAL_COLUMNS)


@pytest.mark.unit
def test_views_are_the_only_stroke_input() -> None:
    # Leak-guard: match_counters must reject a raw loader frame (players still A/B).
    frame = make_loader_frame()
    with pytest.raises(KeyError):
        match_counters(frame, "W", "L", LABELS, PARAMS)


@pytest.mark.integration
def test_run_p4b_real(tmp_path: object) -> None:
    from pathlib import Path

    from badminton_vision.eval.harness import load_harness_config
    from badminton_vision.eval.models import load_models_config
    from badminton_vision.experiments.p4b_rehearsal import load_p4b_config, run_p4b
    from badminton_vision.ingest.pipeline import build_elite_timeline, load_elite_strokes

    assert isinstance(tmp_path, Path)
    timeline = build_elite_timeline()
    assert len(timeline) == 94
    strokes = load_elite_strokes(timeline)
    result, report = run_p4b(
        timeline,
        strokes,
        LABELS,
        load_harness_config(paths.CONFIGS_DIR / "harness_v1.yaml"),
        load_models_config(paths.CONFIGS_DIR / "models_v1.yaml"),
        load_p4b_config(paths.CONFIGS_DIR / "experiment_p4b_v1.yaml"),
        run_dir=tmp_path / "p4b",
    )
    expected = {"coinflip", "elo", "log5", "lr_a", "lr_b", "lr_c", "xgb_a", "xgb_b", "xgb_c"}
    assert set(result.per_match["predictor"]) == expected
    assert len(result.per_match) == 94 * 9
    windows = report["windows"]
    assert isinstance(windows, dict)
    assert "lr_b_vs_lr_a" in windows["full"]["contrasts"]
    assert "calibration_scored" in report
    assert (tmp_path / "p4b" / "p4b_report.json").is_file()
