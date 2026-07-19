"""The leak firewall: outcome-stripped views and deterministic side assignment.

ShuttleSet stroke files encode "player A" as the match winner, so several
columns leak the outcome by construction. Feature code consumes exactly two
views built here — HistoricalView (full columns, players re-keyed to names;
legal only for matches strictly earlier than the prediction target) and
PreOutcomeView (the target match with every outcome-bearing column stripped).
Changing STRIPPED_COLUMNS requires updating the strip-table justification in
the increment plan and tests/test_preoutcome.py in the same commit.
"""

from hashlib import blake2b
from typing import Final

import pandas as pd

from badminton_vision.errors import DataContractError
from badminton_vision.ingest.shuttleset import PROVENANCE_COLUMNS, STROKE_COLUMNS

# Justification per column lives in the increment plan's strip table.
STRIPPED_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "roundscore_A",  # winner-encoded column name leaks the match outcome
        "roundscore_B",
        "player",  # A = match winner by construction; re-keyed to side, then dropped
        "getpoint_player",  # literal rally outcome, winner-encoded
        "win_reason",  # post-rally outcome annotation
        "lose_reason",
        "flaw",  # post-hoc annotator QA flag, winner-referenced
        "server",  # undocumented domain {1,2,3}; serve identity = ball_round 1 striker
        "db",  # undocumented annotation artifact
    }
)
KEPT_COLUMNS: Final[tuple[str, ...]] = tuple(c for c in STROKE_COLUMNS if c not in STRIPPED_COLUMNS)


def assign_sides(match_uid: str, winner: str, loser: str, seed: int) -> tuple[str, str, int]:
    """Deterministic per-match coin flip; returns (side0, side1, y) with y=1 iff side0 won.

    Keyed with blake2b, not built-in hash(): hash() is salted per process and
    would silently break bit-identical reruns.
    """
    digest = blake2b(f"{seed}:{match_uid}".encode(), digest_size=8).digest()
    if int.from_bytes(digest, "big") & 1:
        return winner, loser, 1
    return loser, winner, 0


def _require_loader_frame(strokes: pd.DataFrame) -> None:
    expected = set(STROKE_COLUMNS) | set(PROVENANCE_COLUMNS)
    if set(strokes.columns) != expected:
        raise DataContractError(
            f"expected a loader stroke frame with columns {sorted(expected)}, "
            f"got {sorted(strokes.columns)}"
        )


def historical_view(strokes: pd.DataFrame, winner: str, loser: str) -> pd.DataFrame:
    """Full-column view with A/B re-keyed to player names; PAST MATCHES ONLY.

    The caller (timeline/feature code) is responsible for passing only matches
    strictly earlier than the prediction target.
    """
    view = strokes.copy()
    _require_loader_frame(view)
    ab_to_name = {"A": winner, "B": loser}
    view["player"] = view["player"].map(ab_to_name)
    view["getpoint_player"] = view["getpoint_player"].map(ab_to_name)
    if view["player"].isna().any():
        raise DataContractError("player column contains values other than A/B")
    return view


def pre_outcome_view(strokes: pd.DataFrame, winner: str, loser: str, side0: str) -> pd.DataFrame:
    """The target match as a camera would see it: outcome columns stripped, A/B -> sides."""
    _require_loader_frame(strokes)
    if side0 == winner:
        ab_to_side = {"A": "side0", "B": "side1"}
    elif side0 == loser:
        ab_to_side = {"A": "side1", "B": "side0"}
    else:
        raise DataContractError(f"side0 {side0!r} is neither winner nor loser of this match")
    side = strokes["player"].map(ab_to_side)
    if side.isna().any():
        raise DataContractError("player column contains values other than A/B")
    view = strokes.drop(columns=list(STRIPPED_COLUMNS))
    view.insert(0, "side", side)
    return view
