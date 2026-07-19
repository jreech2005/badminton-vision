"""One row per match: Tier 0-2 records features from a single chronological pass.

Every row is emitted from state BEFORE that match's outcome touches any
accumulator — the prefix-invariance test proves it. Pairwise features are
side0 - side1 differences (they negate under a side swap); log5 enters as a
logit so it negates too; round_rank and is_ws are swap-invariant context.
"""

import datetime as dt
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Final, cast

import pandas as pd

from badminton_vision.ingest.preoutcome import assign_sides
from badminton_vision.ratings.elo import Elo, EloParams
from badminton_vision.ratings.log5 import Log5Beta, Log5Params

FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "elo_diff",
    "log5_logit",
    "h2h_net_wins",
    "n_matches_diff",
    "days_since_last_diff",
    "winrate5_diff",
    "winrate10_diff",
    "streak_diff",
    "game_win_share_diff",
    "opp_quality_diff",
    "round_rank",
    "is_ws",
)
# Swap-invariant context columns; everything else negates under a side swap.
INVARIANT_COLUMNS: Final[frozenset[str]] = frozenset({"round_rank", "is_ws"})

_FORM_PRIOR_WINS: Final[float] = 2.0  # Beta(2,2) shrinkage for small-window rates
_FORM_PRIOR_GAMES: Final[float] = 4.0
_OPP_QUALITY_WINDOW: Final[int] = 10


@dataclass
class _PlayerHistory:
    results: deque[int] = field(default_factory=lambda: deque(maxlen=10))
    last_date: dt.date | None = None
    streak: int = 0
    games_won: int = 0
    games_played: int = 0
    opp_elos: deque[float] = field(default_factory=lambda: deque(maxlen=_OPP_QUALITY_WINDOW))
    n_matches: int = 0


def _shrunk_rate(wins: float, games: float) -> float:
    return (wins + _FORM_PRIOR_WINS) / (games + _FORM_PRIOR_GAMES)


def _per_player(history: _PlayerHistory, on: dt.date) -> dict[str, float]:
    if history.n_matches == 0:
        nan = float("nan")
        return {
            "days_since_last": nan,
            "winrate5": nan,
            "winrate10": nan,
            "streak": nan,
            "game_win_share": nan,
            "opp_quality": nan,
        }
    last5 = list(history.results)[-5:]
    last10 = list(history.results)
    assert history.last_date is not None
    return {
        "days_since_last": float((on - history.last_date).days),
        "winrate5": _shrunk_rate(sum(last5), len(last5)),
        "winrate10": _shrunk_rate(sum(last10), len(last10)),
        "streak": float(history.streak),
        "game_win_share": _shrunk_rate(history.games_won, history.games_played),
        "opp_quality": sum(history.opp_elos) / len(history.opp_elos),
    }


def build_match_rows(
    timeline: pd.DataFrame, elo_params: EloParams, log5_params: Log5Params, seed: int
) -> pd.DataFrame:
    """Emit the symmetrized feature row for every timeline match, strictly pre-outcome."""
    elo = Elo(elo_params)
    log5 = Log5Beta(log5_params)
    histories: dict[str, _PlayerHistory] = {}
    h2h: dict[tuple[str, str], dict[str, int]] = {}
    rows: list[dict[str, object]] = []

    for match in timeline.itertuples(index=False):
        match_uid = str(match.match_uid)
        winner, loser = str(match.winner), str(match.loser)
        date = cast(dt.date, match.date)
        discipline = str(match.discipline)
        side0, side1, y = assign_sides(match_uid, winner, loser, seed)
        h0 = histories.setdefault(side0, _PlayerHistory())
        h1 = histories.setdefault(side1, _PlayerHistory())
        f0, f1 = _per_player(h0, date), _per_player(h1, date)
        meetings = h2h.get((min(side0, side1), max(side0, side1)), {})
        log5_p = log5.predict(side0, side1, date)

        rows.append(
            {
                "match_uid": match_uid,
                "t_idx": int(cast(int, match.t_idx)),
                "date": date,
                "side0": side0,
                "side1": side1,
                "y": y,
                "elo_diff": elo.rating(side0, discipline) - elo.rating(side1, discipline),
                "log5_logit": math.log(log5_p / (1.0 - log5_p)),
                "h2h_net_wins": float(meetings.get(side0, 0) - meetings.get(side1, 0)),
                "n_matches_diff": float(h0.n_matches - h1.n_matches),
                "days_since_last_diff": f0["days_since_last"] - f1["days_since_last"],
                "winrate5_diff": f0["winrate5"] - f1["winrate5"],
                "winrate10_diff": f0["winrate10"] - f1["winrate10"],
                "streak_diff": f0["streak"] - f1["streak"],
                "game_win_share_diff": f0["game_win_share"] - f1["game_win_share"],
                "opp_quality_diff": f0["opp_quality"] - f1["opp_quality"],
                "round_rank": float(cast(int, match.round_rank)),
                "is_ws": 1.0 if discipline == "WS" else 0.0,
            }
        )

        # Reveal the outcome to every accumulator only after the row is emitted.
        winner_history, loser_history = histories[winner], histories[loser]
        n_sets = int(cast(int, match.n_sets))
        for history, won, opponent in (
            (winner_history, 1, loser),
            (loser_history, 0, winner),
        ):
            history.opp_elos.append(elo.rating(opponent, discipline))
            history.results.append(won)
            history.last_date = date
            if won:
                history.streak = history.streak + 1 if history.streak > 0 else 1
            else:
                history.streak = history.streak - 1 if history.streak < 0 else -1
            history.games_won += 2 if won else n_sets - 2
            history.games_played += n_sets
            history.n_matches += 1
        key = (min(winner, loser), max(winner, loser))
        h2h.setdefault(key, {})[winner] = h2h.setdefault(key, {}).get(winner, 0) + 1
        elo.update(winner, loser, discipline)
        log5.update(winner, loser, date)

    return pd.DataFrame(rows)
