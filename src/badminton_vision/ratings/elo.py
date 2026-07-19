"""Adaptive-K Elo ratings keyed by (player, format)."""

from pydantic import BaseModel, ConfigDict


class EloParams(BaseModel):
    """Elo constants; the harness config YAML is the tuning surface."""

    model_config = ConfigDict(frozen=True)

    initial: float
    scale: float
    k_provisional: float
    k_established: float
    provisional_matches: int


class Elo:
    """Mutable rating state; updates must be fed in strict timeline order."""

    def __init__(self, params: EloParams) -> None:
        self.params = params
        self._rating: dict[tuple[str, str], float] = {}
        self._n_played: dict[tuple[str, str], int] = {}

    def rating(self, player: str, fmt: str) -> float:
        """Current rating; unseen players sit at the initial rating."""
        return self._rating.get((player, fmt), self.params.initial)

    def n_played(self, player: str, fmt: str) -> int:
        """Matches this player has completed in this format."""
        return self._n_played.get((player, fmt), 0)

    def _k(self, player: str, fmt: str) -> float:
        provisional = self.n_played(player, fmt) < self.params.provisional_matches
        return self.params.k_provisional if provisional else self.params.k_established

    def expect(self, p0: str, p1: str, fmt: str) -> float:
        """P(p0 beats p1) under the logistic Elo curve."""
        diff = self.rating(p0, fmt) - self.rating(p1, fmt)
        return float(1.0 / (1.0 + 10.0 ** (-diff / self.params.scale)))

    def update(self, winner: str, loser: str, fmt: str) -> None:
        """Apply one result; each player's delta uses their own K-factor."""
        expected_win = self.expect(winner, loser, fmt)
        surprise = 1.0 - expected_win
        self._rating[(winner, fmt)] = self.rating(winner, fmt) + self._k(winner, fmt) * surprise
        self._rating[(loser, fmt)] = self.rating(loser, fmt) - self._k(loser, fmt) * surprise
        self._n_played[(winner, fmt)] = self.n_played(winner, fmt) + 1
        self._n_played[(loser, fmt)] = self.n_played(loser, fmt) + 1
