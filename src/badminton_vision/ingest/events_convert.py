"""Convert one ShuttleSet match into the events_v1 contract.

Exists to prove the contract fits real data before any CV code does; the
observation stream is built from the outcome-stripped PreOutcomeView, and the
outcome channel from the match record — never the other way around.
"""

import datetime as dt

import pandas as pd

from badminton_vision.errors import DataContractError
from badminton_vision.ingest.labels import LabelMap
from badminton_vision.ingest.preoutcome import assign_sides, pre_outcome_view
from badminton_vision.schemas.events_v1 import (
    MatchEventFile,
    MatchOutcome,
    RallyEvent,
    ShotEvent,
    SourceKind,
)


def _xy(row: pd.Series, x_col: str, y_col: str) -> tuple[float, float] | None:
    if pd.isna(row[x_col]) or pd.isna(row[y_col]):
        return None
    return float(row[x_col]), float(row[y_col])


def shuttleset_to_events(
    meta: pd.Series,
    strokes: pd.DataFrame,
    label_map: LabelMap,
    seed: int,
    discipline: str,
) -> MatchEventFile:
    """Build a MatchEventFile for one match from its loader stroke frame."""
    match_uid = str(meta["match_uid"])
    winner, loser = str(meta["winner"]), str(meta["loser"])
    side0, side1, y = assign_sides(match_uid, winner, loser, seed)
    view = pre_outcome_view(strokes, winner, loser, side0=side0)
    view = view.sort_values(["set_no", "rally", "ball_round"])

    rallies: list[RallyEvent] = []
    for rally_idx, (_, rally_rows) in enumerate(view.groupby(["set_no", "rally"], sort=True)):
        shots: list[ShotEvent] = []
        for shot_idx, (_, row) in enumerate(rally_rows.iterrows()):
            striker = str(row["side"])
            if striker not in ("side0", "side1"):
                raise DataContractError(f"{match_uid}: bad striker value {striker!r}")
            shots.append(
                ShotEvent(
                    rally_idx=rally_idx,
                    shot_idx=shot_idx,
                    frame_hit=int(row["frame_num"]),
                    striker=striker,
                    shot_class=label_map.canonical(str(row["type"])),
                    shot_class_version=label_map.version,
                    around_head=bool(row["aroundhead"]) if pd.notna(row["aroundhead"]) else None,
                    backhand=bool(row["backhand"]) if pd.notna(row["backhand"]) else None,
                    striker_xy=_xy(row, "player_location_x", "player_location_y"),
                    receiver_xy=_xy(row, "opponent_location_x", "opponent_location_y"),
                    landing_xy=_xy(row, "landing_x", "landing_y"),
                    source=SourceKind.human_annotation,
                )
            )
        rallies.append(RallyEvent(rally_idx=rally_idx, shots=shots))

    date = meta["date"]
    if not isinstance(date, dt.date):
        raise DataContractError(f"{match_uid}: match table date is not a date: {date!r}")
    return MatchEventFile(
        match_uid=match_uid,
        date=date,
        discipline=discipline,
        side0_player=side0,
        side1_player=side1,
        rallies=rallies,
        outcome=MatchOutcome(winner_side="side0" if y == 1 else "side1"),
    )
