"""BadmintonDB loader: nine Ginting-Momota matches, match results only this increment.

Filenames encode YYYY-WW-round; per-point JSON records the pre-point state
(Score = games won, Point = points in the current game). The match winner is
the unique player who can end the match on the final recorded point; the
curated override table cross-checks that extraction and supplies day-precision
dates with provenance.
"""

import datetime as dt
import json
import re
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any, Final

import pandas as pd

from badminton_vision.errors import DataContractError
from badminton_vision.ingest.shuttleset import MATCH_TABLE_COLUMNS, MatchMeta

_FILENAME: Final[re.Pattern[str]] = re.compile(
    r"^(?P<year>\d{4})-(?P<week>\d{2})-\d(?P<round>R32|R16|QF|SF|F)-MS-"
    r"(?P<t1>[^-]+)-(?P<t2>[^-]+)-(?P<tournament>[^-]+)-1080p(?:-50fps)?\.json$"
)
_GAMES_TO_WIN_MATCH: Final[int] = 2  # best-of-three singles


def _would_win_game(own_points: int, opp_points: int) -> bool:
    after = own_points + 1
    if after == 30 and opp_points == 29:
        return True
    return after >= 21 and after - opp_points >= 2 and after <= 30


def _ordered_points(doc: dict[str, Any], context: str) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for i in range(1, len(doc) + 1):
        key = f"Point{i}"
        if key not in doc:
            raise DataContractError(f"{context}: point keys not contiguous at {key}")
        points.append(doc[key]["PointInfo"])
    return points


def extract_winner(points: list[dict[str, Any]]) -> str | None:
    """Return 'T1P1'/'T2P1' if exactly one player can end the match on the final point."""
    last = points[-1]
    candidates = [
        team
        for team, opponent in (("T1P1", "T2P1"), ("T2P1", "T1P1"))
        if _would_win_game(last[team]["Point"], last[opponent]["Point"])
        and last[team]["Score"] + 1 == _GAMES_TO_WIN_MATCH
    ]
    if len(candidates) == 1:
        return candidates[0]
    return None


def load_date_overrides(path: Path) -> pd.DataFrame:
    """Load the curated override table; every row must carry a provenance URL."""
    table = pd.read_csv(path, comment="#")
    if table["provenance_url"].isna().any():
        raise DataContractError(f"{path.name}: override row without provenance URL")
    return table


def load_bdb_match_table(
    json_dir: Path, aliases: Mapping[str, str], overrides: pd.DataFrame
) -> pd.DataFrame:
    """Load the nine BadmintonDB matches as MatchMeta rows (records only)."""
    override_by_uid = {str(r.match_uid): r for r in overrides.itertuples(index=False)}
    metas: list[MatchMeta] = []
    for json_file in sorted(json_dir.glob("*.json")):
        parsed = _FILENAME.match(json_file.name)
        if parsed is None:
            raise DataContractError(f"unparseable BadmintonDB filename: {json_file.name}")
        year, week = int(parsed["year"]), int(parsed["week"])
        match_uid = f"bdb:{year}-{week:02d}"
        context = f"{match_uid} ({json_file.name})"
        names = {
            "T1P1": aliases.get(parsed["t1"], parsed["t1"]),
            "T2P1": aliases.get(parsed["t2"], parsed["t2"]),
        }
        points = _ordered_points(json.loads(json_file.read_text()), context)
        extracted_key = extract_winner(points)
        extracted = None if extracted_key is None else names[extracted_key]

        override = override_by_uid.get(match_uid)
        if override is not None:
            winner = str(override.winner)
            if extracted is not None and extracted != winner:
                raise DataContractError(
                    f"{context}: extracted winner {extracted!r} disagrees with "
                    f"curated override {winner!r}"
                )
            date = dt.date.fromisoformat(str(override.date))
            mismatch_ok = str(override.week_mismatch_ok) == "yes"
            if date.isocalendar()[:2] != (year, week) and not mismatch_ok:
                raise DataContractError(
                    f"{context}: override date {date} is outside filename ISO week "
                    f"{year}-W{week:02d} and week_mismatch_ok is not set"
                )
            precision = "day"
        elif extracted is not None:
            winner = extracted
            date = dt.date.fromisocalendar(year, week, 1)
            precision = "week"
        else:
            raise DataContractError(f"{context}: winner not derivable and no curated override")

        loser = names["T2P1"] if winner == names["T1P1"] else names["T1P1"]
        if winner not in names.values():
            raise DataContractError(f"{context}: winner {winner!r} is not one of {names}")
        last = points[-1]
        metas.append(
            MatchMeta(
                match_uid=match_uid,
                source="badminton_db",
                video=json_file.stem,
                date=date,
                date_precision=precision,
                tournament=str(parsed["tournament"]),
                round_name=str(parsed["round"]),
                winner=winner,
                loser=loser,
                n_sets=int(last["T1P1"]["Score"]) + int(last["T2P1"]["Score"]) + 1,
                duration_min=None,
            )
        )
    return pd.DataFrame([asdict(m) for m in metas], columns=list(MATCH_TABLE_COLUMNS))
