"""Rating policy: adaptive-K Elo and Beta-shrunk decayed log5, hand-computed."""

import datetime as dt

import pytest

from badminton_vision.ratings.elo import Elo, EloParams
from badminton_vision.ratings.log5 import Log5Beta, Log5Params

pytestmark = pytest.mark.unit

PARAMS = EloParams(
    initial=1500.0, scale=400.0, k_provisional=40.0, k_established=20.0, provisional_matches=2
)
D0 = dt.date(2021, 1, 1)


def test_elo_initial_expectation_is_half() -> None:
    assert Elo(PARAMS).expect("A", "B", "MS") == 0.5


def test_elo_first_update_hand_computed() -> None:
    elo = Elo(PARAMS)
    elo.update("A", "B", "MS")
    # delta = K * (1 - 0.5) = 40 * 0.5 = 20, exactly representable.
    assert elo.rating("A", "MS") == 1520.0
    assert elo.rating("B", "MS") == 1480.0


def test_elo_expect_antisymmetry() -> None:
    elo = Elo(PARAMS)
    elo.update("A", "B", "MS")
    # Logistic curve: e(A,B) + e(B,A) must sum to 1 (float addition of the two forms).
    assert elo.expect("A", "B", "MS") + elo.expect("B", "A", "MS") == pytest.approx(1.0)


def test_elo_k_drops_after_provisional_matches() -> None:
    elo = Elo(PARAMS)
    for _ in range(2):
        elo.update("A", "B", "MS")
    rating_before = elo.rating("A", "MS")
    expected_win = elo.expect("A", "B", "MS")
    elo.update("A", "B", "MS")
    # Third match: A has 2 played, provisional_matches=2, so K is established (20).
    assert elo.rating("A", "MS") == pytest.approx(rating_before + 20.0 * (1 - expected_win))


def test_elo_pools_are_format_scoped() -> None:
    elo = Elo(PARAMS)
    elo.update("A", "B", "MS")
    assert elo.rating("A", "WS") == 1500.0
    assert elo.n_played("A", "WS") == 0


def test_log5_virgin_players_are_even() -> None:
    log5 = Log5Beta(Log5Params(prior_wins=5.0, prior_games=10.0, half_life_days=None))
    assert log5.predict("A", "B", D0) == 0.5


def test_log5_shrunk_rate_hand_computed() -> None:
    log5 = Log5Beta(Log5Params(prior_wins=5.0, prior_games=10.0, half_life_days=None))
    log5.update("A", "B", D0)
    # A is 1-0: (1+5)/(1+10) = 6/11. Against a 0.5 opponent log5 returns A's own rate.
    assert log5.win_rate("A", D0) == pytest.approx(6 / 11)
    assert log5.predict("A", "C", D0) == pytest.approx(6 / 11)


def test_log5_decay_halves_evidence_at_half_life() -> None:
    log5 = Log5Beta(Log5Params(prior_wins=5.0, prior_games=10.0, half_life_days=10.0))
    log5.update("A", "B", D0)
    # Ten days on, the single win counts as half a win over half a game.
    assert log5.win_rate("A", D0 + dt.timedelta(days=10)) == pytest.approx(5.5 / 10.5)


def test_log5_no_decay_when_half_life_none() -> None:
    log5 = Log5Beta(Log5Params(prior_wins=5.0, prior_games=10.0, half_life_days=None))
    log5.update("A", "B", D0)
    assert log5.win_rate("A", D0 + dt.timedelta(days=1000)) == pytest.approx(6 / 11)
