"""Match-row policy: prefix invariance, antisymmetry, leak-free feature slices."""

import math

import pandas as pd
import pytest

from badminton_vision import paths
from badminton_vision.errors import DataContractError
from badminton_vision.eval.harness import load_harness_config, run_walk_forward
from badminton_vision.eval.metrics import summarize
from badminton_vision.eval.models import (
    load_models_config,
    make_lr_predictor,
    make_xgb_predictor,
)
from badminton_vision.experiments.p4a_records import features_by_uid
from badminton_vision.features.match_row import (
    FEATURE_COLUMNS,
    INVARIANT_COLUMNS,
    build_match_rows,
)
from tests.conftest import cell, make_synthetic_timeline

CONFIG = load_harness_config(paths.CONFIGS_DIR / "harness_v1.yaml")
MODELS = load_models_config(paths.CONFIGS_DIR / "models_v1.yaml")


def _rows(n: int = 24, seed: int | None = None) -> pd.DataFrame:
    timeline = make_synthetic_timeline(n)
    return build_match_rows(
        timeline, CONFIG.elo, CONFIG.log5, CONFIG.seed if seed is None else seed
    )


@pytest.mark.unit
def test_prefix_invariance() -> None:
    # Policy: the row for match k must not change when the future is truncated away.
    timeline = make_synthetic_timeline(24)
    full = build_match_rows(timeline, CONFIG.elo, CONFIG.log5, CONFIG.seed)
    prefix = build_match_rows(timeline.head(8), CONFIG.elo, CONFIG.log5, CONFIG.seed)
    pd.testing.assert_frame_equal(full.head(8).reset_index(drop=True), prefix)


@pytest.mark.unit
def test_antisymmetry_under_seed_flip() -> None:
    # Policy: swapping side assignment negates every diff feature and flips y.
    first, second = _rows(seed=1), _rows(seed=2)
    flipped = first["side0"] != second["side0"]
    assert flipped.any(), "no assignment flipped between seeds; test cannot bite"
    assert (~flipped).any()
    diff_columns = [c for c in FEATURE_COLUMNS if c not in INVARIANT_COLUMNS]
    for i in first.index[flipped]:
        assert second.loc[i, "y"] == 1 - cell(first, int(i), "y")
        for column in diff_columns:
            a, b = cell(first, int(i), column), cell(second, int(i), column)
            if math.isnan(a):
                assert math.isnan(b)
            else:
                assert b == pytest.approx(-a), f"{column} did not negate on side swap"
        for column in INVARIANT_COLUMNS:
            assert cell(second, int(i), column) == cell(first, int(i), column)


@pytest.mark.unit
def test_first_meeting_features_are_neutral() -> None:
    rows = _rows()
    assert cell(rows, 0, "elo_diff") == 0.0
    assert cell(rows, 0, "log5_logit") == 0.0
    assert cell(rows, 0, "h2h_net_wins") == 0.0
    assert cell(rows, 0, "n_matches_diff") == 0.0
    assert math.isnan(cell(rows, 0, "winrate10_diff"))


@pytest.mark.unit
def test_h2h_accumulates_with_sign() -> None:
    # S beats P1 at t=0,3,6,... (period 3 rotation); by their third meeting the
    # head-to-head must be +/-2 depending on which side S was assigned.
    rows = _rows(24)
    meetings = rows[
        ((rows["side0"] == "S") & (rows["side1"] == "P1"))
        | ((rows["side0"] == "P1") & (rows["side1"] == "S"))
    ].reset_index(drop=True)
    third = meetings.iloc[2]
    expected = 2.0 if third["side0"] == "S" else -2.0
    assert third["h2h_net_wins"] == expected


@pytest.mark.unit
def test_streak_resets_on_direction_change() -> None:
    # Match t=11 is S vs P3. S lost match 9 then won match 10, so a correct
    # reset gives S streak +1 (not 0 or -1+1 bookkeeping); P3 has lost all
    # three of its matches, streak -3. The diff must therefore be exactly 4.
    rows = _rows(24)
    row = rows.iloc[11]
    assert {row["side0"], row["side1"]} == {"S", "P3"}
    expected = 4.0 if row["side0"] == "S" else -4.0
    assert row["streak_diff"] == expected


@pytest.mark.unit
def test_feature_slice_excludes_outcome_and_identity() -> None:
    # Leak-guard: the harness-facing slice carries exactly the feature columns.
    rows = _rows(6)
    slices = features_by_uid(rows)
    example = next(iter(slices.values()))
    assert list(example.index) == list(FEATURE_COLUMNS)
    assert "y" not in example.index
    assert "side0" not in example.index


@pytest.mark.unit
def test_model_fallback_before_min_train() -> None:
    timeline = make_synthetic_timeline(30)
    rows = build_match_rows(timeline, CONFIG.elo, CONFIG.log5, CONFIG.seed)
    lr = make_lr_predictor(MODELS, FEATURE_COLUMNS)
    result = run_walk_forward(timeline, [lr], CONFIG, features_by_uid=features_by_uid(rows))
    early = result.per_match.head(MODELS.min_train)
    assert (early["p"] == 0.5).all()


@pytest.mark.unit
def test_lr_learns_on_biased_synthetic() -> None:
    timeline = make_synthetic_timeline(60)
    rows = build_match_rows(timeline, CONFIG.elo, CONFIG.log5, CONFIG.seed)
    lr = make_lr_predictor(MODELS, FEATURE_COLUMNS)
    result = run_walk_forward(timeline, [lr], CONFIG, features_by_uid=features_by_uid(rows))
    scored = result.per_match[result.per_match["scored"]]
    assert cell(summarize(scored).set_index("predictor"), "lr", "brier") < 0.25


@pytest.mark.unit
def test_monotone_constraint_rejects_unknown_feature() -> None:
    bad = MODELS.model_copy(update={"monotone_increasing": ("no_such_feature",)})
    with pytest.raises(DataContractError, match="unknown features"):
        make_xgb_predictor(bad, FEATURE_COLUMNS)


@pytest.mark.integration
def test_run_p4a_real(tmp_path: object) -> None:
    from pathlib import Path

    from badminton_vision.experiments.p4a_records import run_p4a
    from badminton_vision.ingest.pipeline import build_global_timeline

    assert isinstance(tmp_path, Path)
    timeline = build_global_timeline()
    result, report = run_p4a(timeline, CONFIG, MODELS, run_dir=tmp_path / "p4a")
    assert set(result.per_match["predictor"]) == {"coinflip", "elo", "log5", "lr", "xgb"}
    assert len(result.per_match) == 103 * 5
    windows = report["windows"]
    assert isinstance(windows, dict)
    assert "lr_vs_elo" in windows["full"]["contrasts"]
    assert (tmp_path / "p4a" / "p4a_report.json").is_file()
    assert (tmp_path / "p4a" / "manifest.json").is_file()
