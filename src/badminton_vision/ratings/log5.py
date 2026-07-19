"""Beta-shrunk, optionally time-decayed win rates combined with the log5 formula."""

import datetime as dt

from pydantic import BaseModel, ConfigDict


class Log5Params(BaseModel):
    """Shrinkage prior and decay half-life; None half-life disables decay."""

    model_config = ConfigDict(frozen=True)

    prior_wins: float
    prior_games: float
    half_life_days: float | None


class Log5Beta:
    """Mutable per-player decayed win/game sums; feed updates in timeline order."""

    def __init__(self, params: Log5Params) -> None:
        self.params = params
        self._wins: dict[str, float] = {}
        self._games: dict[str, float] = {}
        self._as_of: dict[str, dt.date] = {}

    def _decayed(self, player: str, on: dt.date) -> tuple[float, float]:
        if player not in self._as_of:
            return 0.0, 0.0
        if self.params.half_life_days is None:
            return self._wins[player], self._games[player]
        elapsed = (on - self._as_of[player]).days
        factor = 0.5 ** (max(elapsed, 0) / self.params.half_life_days)
        return self._wins[player] * factor, self._games[player] * factor

    def win_rate(self, player: str, on: dt.date) -> float:
        """Beta-shrunk decayed win rate as of a date."""
        wins, games = self._decayed(player, on)
        return (wins + self.params.prior_wins) / (games + self.params.prior_games)

    def predict(self, p0: str, p1: str, on: dt.date) -> float:
        """log5: P(p0 beats p1) from the two shrunk win rates."""
        r0, r1 = self.win_rate(p0, on), self.win_rate(p1, on)
        numerator = r0 * (1.0 - r1)
        return numerator / (numerator + r1 * (1.0 - r0))

    def update(self, winner: str, loser: str, on: dt.date) -> None:
        """Apply one result, decaying each player's sums up to the match date."""
        for player, won in ((winner, 1.0), (loser, 0.0)):
            wins, games = self._decayed(player, on)
            self._wins[player] = wins + won
            self._games[player] = games + 1.0
            self._as_of[player] = on
