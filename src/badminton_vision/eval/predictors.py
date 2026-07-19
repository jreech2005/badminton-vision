"""The Predictor protocol and the reference rating predictors."""

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from badminton_vision.errors import DataContractError
from badminton_vision.ratings.elo import Elo, EloParams
from badminton_vision.ratings.log5 import Log5Beta, Log5Params


@dataclass(frozen=True)
class MatchContext:
    """Everything a predictor may legally see BEFORE a match is played."""

    match_uid: str
    t_idx: int
    date: dt.date
    discipline: str
    side0: str
    side1: str
    n_prior_side0: int
    n_prior_side1: int
    features: "pd.Series[float] | None" = None


@dataclass(frozen=True)
class MatchResult:
    """A completed match: the context plus the revealed label."""

    ctx: MatchContext
    y: int  # 1 iff side0 won

    @property
    def winner(self) -> str:
        return self.ctx.side0 if self.y == 1 else self.ctx.side1

    @property
    def loser(self) -> str:
        return self.ctx.side1 if self.y == 1 else self.ctx.side0


class Predictor(Protocol):
    """predict() is always called before observe() for the same match."""

    name: str

    def predict(self, ctx: MatchContext, /) -> float: ...

    def observe(self, result: MatchResult, /) -> None: ...


class CoinFlip:
    """H0: exactly 0.5 for every match; per-match Brier is float-exactly 0.25."""

    name = "coinflip"

    def predict(self, _ctx: MatchContext) -> float:
        return 0.5

    def observe(self, _result: MatchResult) -> None:
        return None


class EloPredictor:
    """H1 baseline: adaptive-K Elo expectation, rating pools keyed by discipline."""

    name = "elo"

    def __init__(self, params: EloParams) -> None:
        self._elo = Elo(params)

    def predict(self, ctx: MatchContext) -> float:
        return self._elo.expect(ctx.side0, ctx.side1, ctx.discipline)

    def observe(self, result: MatchResult) -> None:
        self._elo.update(result.winner, result.loser, result.ctx.discipline)


class Log5Predictor:
    """H1 baseline: Beta-shrunk decayed win rates through the log5 formula."""

    name = "log5"

    def __init__(self, params: Log5Params) -> None:
        self._log5 = Log5Beta(params)

    def predict(self, ctx: MatchContext) -> float:
        return self._log5.predict(ctx.side0, ctx.side1, ctx.date)

    def observe(self, result: MatchResult) -> None:
        self._log5.update(result.winner, result.loser, result.ctx.date)


def make_baseline_predictors(
    names: Sequence[str], elo: EloParams, log5: Log5Params
) -> list[Predictor]:
    """Instantiate baseline predictors by name; unknown names fail loud."""
    registry: dict[str, Predictor] = {
        "coinflip": CoinFlip(),
        "elo": EloPredictor(elo),
        "log5": Log5Predictor(log5),
    }
    unknown = set(names) - set(registry)
    if unknown:
        raise DataContractError(f"unknown predictors: {sorted(unknown)}")
    return [registry[name] for name in names]
