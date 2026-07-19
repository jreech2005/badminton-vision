"""Bootstrap policy: matches are the cluster unit; results are seeded and exact-null aware."""

import pandas as pd
import pytest

from badminton_vision.errors import DataContractError
from badminton_vision.eval.bootstrap import paired_bootstrap_brier

pytestmark = pytest.mark.unit


def _log(diffs: list[float]) -> pd.DataFrame:
    rows = []
    for i, diff in enumerate(diffs):
        rows.append({"match_uid": f"m{i}", "predictor": "a", "brier": 0.2 + diff})
        rows.append({"match_uid": f"m{i}", "predictor": "b", "brier": 0.2})
    return pd.DataFrame(rows)


def test_constant_positive_diff_excludes_zero() -> None:
    result = paired_bootstrap_brier(_log([0.05] * 20), "a", "b", n_boot=200, seed=3)
    assert result.mean_diff == pytest.approx(0.05)
    assert result.ci_low == pytest.approx(0.05)
    assert result.ci_high == pytest.approx(0.05)
    assert result.p_value == 0.0
    assert result.n_matches == 20


def test_zero_diffs_have_p_value_one() -> None:
    result = paired_bootstrap_brier(_log([0.0] * 20), "a", "b", n_boot=200, seed=3)
    assert result.p_value == 1.0


def test_symmetric_diffs_are_not_significant() -> None:
    result = paired_bootstrap_brier(_log([0.1, -0.1] * 15), "a", "b", n_boot=500, seed=3)
    assert result.ci_low < 0 < result.ci_high
    assert result.p_value > 0.2


def test_same_seed_is_bit_identical() -> None:
    log = _log([0.1, -0.05, 0.02] * 10)
    first = paired_bootstrap_brier(log, "a", "b", n_boot=300, seed=11)
    second = paired_bootstrap_brier(log, "a", "b", n_boot=300, seed=11)
    assert first == second


def test_missing_predictor_raises() -> None:
    with pytest.raises(DataContractError, match="not both present"):
        paired_bootstrap_brier(_log([0.1]), "a", "zzz", n_boot=10, seed=1)
