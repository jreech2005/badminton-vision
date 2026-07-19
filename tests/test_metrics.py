"""Metric policy: exactness where math guarantees it, hand-computed elsewhere."""

import numpy as np
import pandas as pd
import pytest

from badminton_vision.eval.metrics import (
    accuracy,
    brier,
    calibration_table,
    log_loss,
    summarize,
)

pytestmark = pytest.mark.unit


def test_brier_at_half_is_float_exact_quarter() -> None:
    scores = brier(pd.Series([0.5, 0.5]), pd.Series([0, 1]))
    # 0.5**2 == 0.25 exactly in binary floating point: no tolerance permitted.
    assert (scores == 0.25).all()


def test_brier_hand_computed() -> None:
    assert brier(pd.Series([0.9]), pd.Series([1])).iloc[0] == pytest.approx(0.01)


def test_log_loss_is_clipped_finite() -> None:
    scores = log_loss(pd.Series([0.0, 1.0]), pd.Series([1, 0]))
    assert np.isfinite(scores).all()
    assert (scores > 30).all()  # ln(1e-15) ~ -34.5: catastrophic but finite


def test_accuracy_gives_half_credit_at_exactly_half() -> None:
    p = pd.Series([0.5, 0.9, 0.1])
    y = pd.Series([1, 1, 0])
    # correct: [half, yes, yes] -> (2 + 0.5) / 3
    assert accuracy(p, y) == pytest.approx(5 / 6)


def test_calibration_table_buckets() -> None:
    p = pd.Series([0.05, 0.15, 0.55, 0.95])
    y = pd.Series([0, 1, 1, 1])
    table = calibration_table(p, y, n_buckets=5)
    assert len(table) == 5
    assert table.loc[0, "n"] == 2  # 0.05 and 0.15 share the [0, 0.2) bucket
    assert table.loc[0, "empirical_rate"] == pytest.approx(0.5)
    assert table.loc[4, "mean_p"] == pytest.approx(0.95)


def test_summarize_groups_by_predictor() -> None:
    per_match = pd.DataFrame(
        {
            "predictor": ["a", "a", "b"],
            "p": [0.5, 0.5, 0.9],
            "y": [1, 0, 1],
        }
    )
    per_match["brier"] = brier(per_match["p"], per_match["y"])
    per_match["log_loss"] = log_loss(per_match["p"], per_match["y"])
    summary = summarize(per_match).set_index("predictor")
    assert summary.loc["a", "brier"] == 0.25
    assert summary.loc["b", "n"] == 1
