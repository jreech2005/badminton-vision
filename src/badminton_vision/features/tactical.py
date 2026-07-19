"""P4b tactical features: per-player stroke-derived rates from strictly earlier matches.

Consumes HistoricalView frames only (the leak firewall's past-matches channel).
Rates are exponentially time-decayed, Beta-shrunk toward the pooled elite rate,
and NaN-gated below minimum weighted support. Unknown-label policy per
shot_labels_v1: unknown strokes leave class-conditional rates but their rallies
still count for rally-level features.
"""

import datetime as dt
from dataclasses import dataclass, field
from typing import Final, cast

import pandas as pd
from pydantic import BaseModel, ConfigDict

from badminton_vision.errors import DataContractError
from badminton_vision.ingest.labels import LabelMap, apply_labels
from badminton_vision.ingest.preoutcome import assign_sides, historical_view

TACTICAL_COLUMNS: Final[tuple[str, ...]] = (
    "smash_kill_rate_diff",
    "stroke_error_rate_diff",
    "net_error_rate_diff",
    "serve_fault_rate_diff",
    "short_serve_share_diff",
    "attack_share_diff",
    "backhand_share_diff",
    "aroundhead_share_diff",
    "median_rally_len_diff",
    "long_rally_win_rate_diff",
    "short_rally_win_rate_diff",
    "serve_advantage_diff",
    "third_stroke_attack_rate_diff",
)

_SMASH_CLASSES: Final[frozenset[str]] = frozenset({"smash", "wrist_smash"})
# Rates shrunk with the serve pseudo-count and gated on serve support.
_SERVE_FEATURES: Final[frozenset[str]] = frozenset(
    {"serve_fault_rate", "short_serve_share", "serve_advantage", "third_stroke_attack_rate"}
)


class TacticalParams(BaseModel):
    """Decay, support, and shrinkage constants (configs/experiment_p4b_v1.yaml)."""

    model_config = ConfigDict(frozen=True)

    half_life_days: float
    min_weighted_strokes: float
    min_weighted_serves: float
    shrink_strokes: float
    shrink_serves: float
    long_rally_min: int
    short_rally_max: int


@dataclass
class _MatchCounters:
    """One player's raw tactical counts from one match."""

    date: dt.date
    strokes_total: float = 0.0
    last_stroke_losses: float = 0.0
    smash_strokes: float = 0.0
    smash_kills: float = 0.0
    net_strokes: float = 0.0
    net_last_losses: float = 0.0
    serves: float = 0.0
    serve_faults: float = 0.0
    short_serves: float = 0.0
    long_serves: float = 0.0
    attack_strokes: float = 0.0
    classified_strokes: float = 0.0
    backhand_strokes: float = 0.0
    aroundhead_strokes: float = 0.0
    flagged_strokes: float = 0.0
    rally_lengths: list[int] = field(default_factory=list)
    long_played: float = 0.0
    long_won: float = 0.0
    short_played: float = 0.0
    short_won: float = 0.0
    serve_rallies: float = 0.0
    serve_rally_wins: float = 0.0
    receive_rallies: float = 0.0
    receive_rally_wins: float = 0.0
    third_stroke_chances: float = 0.0
    third_stroke_attacks: float = 0.0


_RATE_DEFS: Final[tuple[tuple[str, str, str], ...]] = (
    # (feature, numerator attr, denominator attr)
    ("smash_kill_rate", "smash_kills", "smash_strokes"),
    ("stroke_error_rate", "last_stroke_losses", "strokes_total"),
    ("net_error_rate", "net_last_losses", "net_strokes"),
    ("serve_fault_rate", "serve_faults", "serves"),
    ("short_serve_share", "short_serves", "serves"),
    ("attack_share", "attack_strokes", "classified_strokes"),
    ("backhand_share", "backhand_strokes", "flagged_strokes"),
    ("aroundhead_share", "aroundhead_strokes", "flagged_strokes"),
    ("long_rally_win_rate", "long_won", "long_played"),
    ("short_rally_win_rate", "short_won", "short_played"),
    ("third_stroke_attack_rate", "third_stroke_attacks", "third_stroke_chances"),
)


def match_counters(
    view: pd.DataFrame, winner: str, loser: str, label_map: LabelMap, params: TacticalParams
) -> dict[str, _MatchCounters]:
    """Extract both players' raw counters from one match's HistoricalView."""
    labelled = apply_labels(view, label_map)
    date = dt.date.min  # placeholder; caller stamps the real date
    counters = {winner: _MatchCounters(date), loser: _MatchCounters(date)}
    for _, rally_rows in labelled.groupby(["set_no", "rally"], sort=True):
        rally = rally_rows.sort_values("ball_round")
        length = len(rally)
        last = rally.iloc[-1]
        first = rally.iloc[0]
        server = str(first["player"])
        rally_winner = None if pd.isna(last["getpoint_player"]) else str(last["getpoint_player"])

        for player, counter in counters.items():
            mine = rally[rally["player"] == player]
            counter.strokes_total += len(mine)
            counter.classified_strokes += int((~mine["is_unknown"]).sum())
            counter.smash_strokes += int(mine["shot_class"].isin(_SMASH_CLASSES).sum())
            counter.net_strokes += int((mine["shot_group"] == "net").sum())
            counter.short_serves += int((mine["shot_class"] == "short_service").sum())
            counter.long_serves += int((mine["shot_class"] == "long_service").sum())
            counter.attack_strokes += int((mine["shot_group"] == "attack").sum())
            flagged = mine[mine["backhand"].notna() & mine["aroundhead"].notna()]
            counter.flagged_strokes += len(flagged)
            counter.backhand_strokes += int((flagged["backhand"] == 1).sum())
            counter.aroundhead_strokes += int((flagged["aroundhead"] == 1).sum())
            counter.rally_lengths.append(length)

        counters[server].serves += 1
        striker_last = str(last["player"])
        if rally_winner is not None:
            loser_of_rally = winner if rally_winner == loser else loser
            if striker_last == loser_of_rally:
                counters[loser_of_rally].last_stroke_losses += 1
                if str(last["shot_group"]) == "net":
                    counters[loser_of_rally].net_last_losses += 1
            if striker_last == rally_winner and str(last["shot_class"]) in _SMASH_CLASSES:
                counters[rally_winner].smash_kills += 1
            if length == 1 and rally_winner != server:
                counters[server].serve_faults += 1
            receiver = winner if server == loser else loser
            counters[server].serve_rallies += 1
            counters[receiver].receive_rallies += 1
            if rally_winner == server:
                counters[server].serve_rally_wins += 1
            else:
                counters[receiver].receive_rally_wins += 1
            for player, counter in counters.items():
                if length >= params.long_rally_min:
                    counter.long_played += 1
                    counter.long_won += 1 if rally_winner == player else 0
                if length <= params.short_rally_max:
                    counter.short_played += 1
                    counter.short_won += 1 if rally_winner == player else 0
        if length >= 3:
            third = rally.iloc[2]
            if str(third["player"]) == server:
                counters[server].third_stroke_chances += 1
                if str(third["shot_group"]) == "attack":
                    counters[server].third_stroke_attacks += 1
    return counters


class TacticalState:
    """Per-player decayed tactical accumulators plus the pooled elite prior."""

    def __init__(self, label_map: LabelMap, params: TacticalParams) -> None:
        self._label_map = label_map
        self.params = params
        self._history: dict[str, list[_MatchCounters]] = {}
        self._pooled: dict[str, float] = {}

    def observe_match(self, date: dt.date, winner: str, loser: str, strokes: pd.DataFrame) -> None:
        """Fold one completed match's strokes into both players' histories."""
        view = historical_view(strokes, winner, loser)
        for player, counter in match_counters(
            view, winner, loser, self._label_map, self.params
        ).items():
            counter.date = date
            self._history.setdefault(player, []).append(counter)
            for feature, num_attr, den_attr in _RATE_DEFS:
                self._pooled[f"{feature}_num"] = self._pooled.get(f"{feature}_num", 0.0) + getattr(
                    counter, num_attr
                )
                self._pooled[f"{feature}_den"] = self._pooled.get(f"{feature}_den", 0.0) + getattr(
                    counter, den_attr
                )

    def _prior(self, feature: str) -> float:
        denominator = self._pooled.get(f"{feature}_den", 0.0)
        if denominator == 0.0:
            return 0.5
        return self._pooled[f"{feature}_num"] / denominator

    def features(self, player: str, on: dt.date) -> dict[str, float]:
        """Decayed, shrunk, support-gated tactical vector as of a date."""
        nan = float("nan")
        result = dict.fromkeys(
            (
                *(name for name, _, _ in _RATE_DEFS),
                "median_rally_len",
                "serve_advantage",
            ),
            nan,
        )
        entries = self._history.get(player, [])
        if not entries:
            return result
        weights = [
            0.5 ** (max((on - entry.date).days, 0) / self.params.half_life_days)
            for entry in entries
        ]

        def weighted(attr: str) -> float:
            return float(
                sum(w * getattr(entry, attr) for w, entry in zip(weights, entries, strict=True))
            )

        if weighted("strokes_total") < self.params.min_weighted_strokes:
            return result
        serve_support = weighted("serves") >= self.params.min_weighted_serves

        for feature, num_attr, den_attr in _RATE_DEFS:
            is_serve_feature = feature in _SERVE_FEATURES
            if is_serve_feature and not serve_support:
                continue
            shrink = self.params.shrink_serves if is_serve_feature else self.params.shrink_strokes
            denominator = weighted(den_attr)
            result[feature] = (weighted(num_attr) + shrink * self._prior(feature)) / (
                denominator + shrink
            )

        lengths_with_weights = sorted(
            (length, w)
            for w, entry in zip(weights, entries, strict=True)
            for length in entry.rally_lengths
        )
        total_weight = sum(w for _, w in lengths_with_weights)
        if total_weight > 0:
            cumulative = 0.0
            for length, w in lengths_with_weights:
                cumulative += w
                if cumulative >= total_weight / 2:
                    result["median_rally_len"] = float(length)
                    break

        if serve_support:
            serve_played, receive_played = weighted("serve_rallies"), weighted("receive_rallies")
            if serve_played > 0 and receive_played > 0:
                result["serve_advantage"] = (
                    weighted("serve_rally_wins") / serve_played
                    - weighted("receive_rally_wins") / receive_played
                )
        return result


def build_tactical_rows(
    timeline: pd.DataFrame,
    strokes_by_uid: dict[str, pd.DataFrame],
    label_map: LabelMap,
    params: TacticalParams,
    seed: int,
) -> pd.DataFrame:
    """One tactical-diff row per timeline match, strictly pre-outcome."""
    state = TacticalState(label_map, params)
    rows: list[dict[str, object]] = []
    for match in timeline.itertuples(index=False):
        match_uid = str(match.match_uid)
        winner, loser = str(match.winner), str(match.loser)
        date = cast(dt.date, match.date)
        side0, side1, y = assign_sides(match_uid, winner, loser, seed)
        f0, f1 = state.features(side0, date), state.features(side1, date)
        row: dict[str, object] = {
            "match_uid": match_uid,
            "t_idx": int(cast(int, match.t_idx)),
            "date": date,
            "side0": side0,
            "side1": side1,
            "y": y,
        }
        for column in TACTICAL_COLUMNS:
            base = column.removesuffix("_diff")
            row[column] = f0[base] - f1[base]
        rows.append(row)
        strokes = strokes_by_uid.get(match_uid)
        if strokes is not None:
            state.observe_match(date, winner, loser, strokes)
    if not rows:
        raise DataContractError("empty timeline passed to build_tactical_rows")
    return pd.DataFrame(rows)
