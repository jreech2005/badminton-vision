"""One loader for both ShuttleSet variants: match tables, stroke frames, homographies.

match.csv is authoritative for identity and dates; video folder names are
opaque verbatim keys (they contain stray spaces and are not winner-first).
"""

import datetime as dt
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from badminton_vision import paths
from badminton_vision.errors import DataContractError

STROKE_COLUMNS: Final[tuple[str, ...]] = (
    "rally",
    "ball_round",
    "time",
    "frame_num",
    "roundscore_A",
    "roundscore_B",
    "player",
    "server",
    "type",
    "aroundhead",
    "backhand",
    "hit_height",
    "hit_area",
    "hit_x",
    "hit_y",
    "landing_height",
    "landing_area",
    "landing_x",
    "landing_y",
    "lose_reason",
    "win_reason",
    "getpoint_player",
    "flaw",
    "player_location_area",
    "player_location_x",
    "player_location_y",
    "opponent_location_area",
    "opponent_location_x",
    "opponent_location_y",
    "db",
)
# Columns the loader stamps onto every stroke row (provenance, never features).
PROVENANCE_COLUMNS: Final[tuple[str, ...]] = ("source", "match_uid", "video", "set_no")

MATCH_TABLE_COLUMNS: Final[tuple[str, ...]] = (
    "match_uid",
    "source",
    "video",
    "date",
    "date_precision",
    "tournament",
    "round_name",
    "winner",
    "loser",
    "n_sets",
    "duration_min",
)


@dataclass(frozen=True)
class SourceSpec:
    """Where one ShuttleSet-schema dataset lives and how its ids are prefixed."""

    name: str
    set_dir: Path
    uid_prefix: str


SHUTTLESET_SPEC: Final[SourceSpec] = SourceSpec(
    name="shuttleset", set_dir=paths.SHUTTLESET_SET_DIR, uid_prefix="ss"
)
SHUTTLESET22_SPEC: Final[SourceSpec] = SourceSpec(
    name="shuttleset22", set_dir=paths.SHUTTLESET22_SET_DIR, uid_prefix="ss22"
)


@dataclass(frozen=True)
class MatchMeta:
    """One match as recorded in match.csv, dates coerced, names optionally aliased."""

    match_uid: str
    source: str
    video: str
    date: dt.date
    date_precision: str
    tournament: str
    round_name: str
    winner: str
    loser: str
    n_sets: int
    duration_min: int | None


def _coerce_int(value: object, context: str) -> int:
    # ShuttleSet22 stores integers as floats ("2022.0"); accept exactly-integral only.
    try:
        as_float = float(str(value))
    except ValueError:
        raise DataContractError(f"{context}: unparseable integer {value!r}") from None
    if not as_float.is_integer():
        raise DataContractError(f"{context}: non-integral value {value!r}")
    return int(as_float)


def load_match_table(spec: SourceSpec, aliases: Mapping[str, str] | None = None) -> pd.DataFrame:
    """Load match.csv as one MatchMeta row per match; verify every video dir exists."""
    table = pd.read_csv(spec.set_dir / "match.csv")
    metas: list[MatchMeta] = []
    for row in table.itertuples(index=False):
        context = f"{spec.name} match id {row.id}"
        video = str(row.video)
        if not (spec.set_dir / video).is_dir():
            raise DataContractError(f"{context}: video dir {video!r} not found verbatim")
        try:
            date = dt.date(
                _coerce_int(row.year, context),
                _coerce_int(row.month, context),
                _coerce_int(row.day, context),
            )
        except ValueError:
            raise DataContractError(
                f"{context}: invalid date {row.year!r}-{row.month!r}-{row.day!r}"
            ) from None
        winner, loser = str(row.winner), str(row.loser)
        if aliases is not None:
            winner = aliases.get(winner, winner)
            loser = aliases.get(loser, loser)
        metas.append(
            MatchMeta(
                match_uid=f"{spec.uid_prefix}:{_coerce_int(row.id, context)}",
                source=spec.name,
                video=video,
                date=date,
                date_precision="day",
                tournament=str(row.tournament),
                round_name=str(row.round),
                winner=winner,
                loser=loser,
                n_sets=_coerce_int(row.set, context),
                duration_min=None if pd.isna(row.duration) else _coerce_int(row.duration, context),
            )
        )
    return pd.DataFrame([asdict(m) for m in metas], columns=list(MATCH_TABLE_COLUMNS))


def load_strokes(
    spec: SourceSpec, match_uid: str, video: str, expected_sets: int | None = None
) -> pd.DataFrame:
    """Concatenate a match's set CSVs; enforce the 30-column contract; stamp provenance."""
    match_dir = spec.set_dir / video
    set_files = sorted(match_dir.glob("set*.csv"))
    if not set_files:
        raise DataContractError(f"{match_uid}: no set CSVs under {match_dir}")
    if expected_sets is not None:
        expected_names = [f"set{i}.csv" for i in range(1, expected_sets + 1)]
        if [f.name for f in set_files] != expected_names:
            raise DataContractError(
                f"{match_uid}: set files {[f.name for f in set_files]} != {expected_names}"
            )
    frames: list[pd.DataFrame] = []
    for set_file in set_files:
        strokes = pd.read_csv(set_file)
        if tuple(strokes.columns) != STROKE_COLUMNS:
            raise DataContractError(
                f"{match_uid}: {set_file.name} columns deviate from the 30-column contract: "
                f"{list(strokes.columns)}"
            )
        strokes["source"] = spec.name
        strokes["match_uid"] = match_uid
        strokes["video"] = video
        strokes["set_no"] = int(set_file.stem.removeprefix("set"))
        frames.append(strokes)
    return pd.concat(frames, ignore_index=True)


def load_all_strokes(spec: SourceSpec, match_table: pd.DataFrame) -> pd.DataFrame:
    """Load stroke frames for every match in a match table, cross-checking set counts."""
    frames = [
        load_strokes(spec, str(uid), str(video), expected_sets=int(n_sets))
        for uid, video, n_sets in zip(
            match_table["match_uid"], match_table["video"], match_table["n_sets"], strict=True
        )
    ]
    return pd.concat(frames, ignore_index=True)


def load_homography(spec: SourceSpec) -> dict[str, np.ndarray]:
    """Parse per-video 3x3 homography matrices; stored for later CV increments."""
    table = pd.read_csv(spec.set_dir / "homography.csv")
    matrices: dict[str, np.ndarray] = {}
    for video, matrix_json in zip(table["video"], table["homography_matrix"], strict=True):
        matrix = np.asarray(json.loads(str(matrix_json)), dtype=np.float64)
        if matrix.shape != (3, 3):
            raise DataContractError(
                f"{spec.name} homography for {video!r} has shape {matrix.shape}, not 3x3"
            )
        matrices[str(video)] = matrix
    return matrices
