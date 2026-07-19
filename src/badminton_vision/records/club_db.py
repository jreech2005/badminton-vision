"""The club records store: SQLite ground truth behind rulebook validation gates.

Quarantine principle: a record that fails a gate is rejected with a reason,
never silently fixed. The harness consumes club data through as_timeline(),
which emits the same match-table contract as the elite loaders.
"""

import datetime as dt
import difflib
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd

from badminton_vision.errors import ValidationGateError
from badminton_vision.ingest.shuttleset import MATCH_TABLE_COLUMNS

logger = logging.getLogger(__name__)

SCHEMA_SQL: Final[Path] = Path(__file__).parent / "schema.sql"
EARLIEST_PLAUSIBLE_DATE: Final[dt.date] = dt.date(2020, 1, 1)
# difflib ratio on normalized names; 0.85 catches "Sam"/"Sam L"-style near-twins.
NAME_SIMILARITY_THRESHOLD: Final[float] = 0.85
GAMES_TO_WIN: Final[int] = 2


@dataclass(frozen=True)
class NewMatch:
    """One match entry as submitted, before any gate has passed it."""

    match_type: str
    played_at: dt.date
    side_a: tuple[str, ...]
    side_b: tuple[str, ...]
    winner_side: str
    games: tuple[tuple[int, int], ...]
    source: str = "manual"


@dataclass(frozen=True)
class ImportReport:
    """Row-level import outcome: quarantined rows carry their gate reason."""

    added: int
    rejected: tuple[tuple[int, str], ...]  # (csv row number, reason)


def normalize_name(name: str) -> str:
    """Casefold, strip, and collapse internal whitespace."""
    return " ".join(name.split()).casefold()


def validate_game_score(side_a: int, side_b: int) -> None:
    """Reject any score pair that is not a legal completed badminton game."""
    winner, loser = max(side_a, side_b), min(side_a, side_b)
    legal = (
        (winner == 21 and loser <= 19)
        or (22 <= winner <= 30 and winner - loser == 2)
        or (winner == 30 and loser == 29)
    )
    if not legal:
        raise ValidationGateError(f"illegal game score {side_a}-{side_b}")


def validate_match(match: NewMatch, today: dt.date) -> None:
    """Run every rulebook gate; raise ValidationGateError with the first violation."""
    if match.match_type not in ("singles", "doubles"):
        raise ValidationGateError(f"unknown match type {match.match_type!r}")
    per_side = 1 if match.match_type == "singles" else 2
    if len(match.side_a) != per_side or len(match.side_b) != per_side:
        raise ValidationGateError(
            f"{match.match_type} needs {per_side} player(s) per side, "
            f"got {len(match.side_a)} vs {len(match.side_b)}"
        )
    if match.winner_side not in ("A", "B"):
        raise ValidationGateError(f"winner side must be A or B, got {match.winner_side!r}")
    names = [normalize_name(n) for n in (*match.side_a, *match.side_b)]
    if len(set(names)) != len(names):
        raise ValidationGateError(f"a player appears twice in {match.side_a} vs {match.side_b}")
    if match.played_at > today:
        raise ValidationGateError(f"match date {match.played_at} is in the future")
    if match.played_at < EARLIEST_PLAUSIBLE_DATE:
        raise ValidationGateError(f"match date {match.played_at} predates the club records era")
    if not 2 <= len(match.games) <= 3:
        raise ValidationGateError(f"a best-of-three match has 2 or 3 games, got {len(match.games)}")
    wins_a = 0
    for side_a_score, side_b_score in match.games:
        validate_game_score(side_a_score, side_b_score)
        wins_a += 1 if side_a_score > side_b_score else 0
    wins_winner = wins_a if match.winner_side == "A" else len(match.games) - wins_a
    if wins_winner != GAMES_TO_WIN:
        raise ValidationGateError(
            f"winner side {match.winner_side} won {wins_winner} games, needs {GAMES_TO_WIN}"
        )
    last_a, last_b = match.games[-1]
    last_winner = "A" if last_a > last_b else "B"
    if last_winner != match.winner_side:
        raise ValidationGateError("the match winner must win the final game")


class ClubDB:
    """One SQLite file of club records; all writes pass through the gates."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._conn = sqlite3.connect(path)
        self._conn.execute("PRAGMA foreign_keys = ON")

    def init_schema(self) -> None:
        """Create the v1 tables if absent."""
        self._conn.executescript(SCHEMA_SQL.read_text())
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def _similar_existing(self, name: str) -> str | None:
        normalized = normalize_name(name)
        for (existing,) in self._conn.execute("SELECT canonical_name FROM players"):
            existing_norm = normalize_name(existing)
            if existing_norm == normalized:
                continue  # exact duplicates are handled by the UNIQUE constraint
            ratio = difflib.SequenceMatcher(None, normalized, existing_norm).ratio()
            if ratio >= NAME_SIMILARITY_THRESHOLD:
                return str(existing)
        return None

    def add_player(self, name: str, *, force_new: bool = False) -> int:
        """Insert a player; near-duplicate names are rejected unless forced."""
        cleaned = " ".join(name.split())
        if not cleaned:
            raise ValidationGateError("empty player name")
        if not force_new:
            twin = self._similar_existing(cleaned)
            if twin is not None:
                raise ValidationGateError(
                    f"name {cleaned!r} is suspiciously close to existing {twin!r}; "
                    f"use force_new if they really are different people"
                )
        try:
            cursor = self._conn.execute(
                "INSERT INTO players (canonical_name) VALUES (?)", (cleaned,)
            )
        except sqlite3.IntegrityError:
            raise ValidationGateError(f"player {cleaned!r} already exists") from None
        self._conn.commit()
        return int(cursor.lastrowid or 0)

    def player_id(self, name: str) -> int:
        """Exact (case-insensitive) lookup; absence is an error, never auto-created."""
        row = self._conn.execute(
            "SELECT id FROM players WHERE canonical_name = ?", (" ".join(name.split()),)
        ).fetchone()
        if row is None:
            raise ValidationGateError(f"unknown player {name!r}; add them first")
        return int(row[0])

    def add_match(self, match: NewMatch, today: dt.date | None = None) -> int:
        """Gate, then insert match + games in one transaction."""
        validate_match(match, today if today is not None else dt.date.today())
        ids_a = [self.player_id(n) for n in match.side_a]
        ids_b = [self.player_id(n) for n in match.side_b]
        duplicate = self._conn.execute(
            "SELECT id FROM matches WHERE played_at = ? AND side_a_p1 IN (?, ?) "
            "AND side_b_p1 IN (?, ?)",
            (match.played_at.isoformat(), ids_a[0], ids_b[0], ids_a[0], ids_b[0]),
        ).fetchone()
        if duplicate is not None:
            # Same-day rematches are legitimate in a club; flag, never block.
            logger.warning(
                "possible duplicate: %s vs %s on %s already recorded (match id %s)",
                match.side_a,
                match.side_b,
                match.played_at,
                duplicate[0],
            )
        with self._conn:
            cursor = self._conn.execute(
                "INSERT INTO matches (match_type, played_at, side_a_p1, side_a_p2, "
                "side_b_p1, side_b_p2, winner_side, source) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    match.match_type,
                    match.played_at.isoformat(),
                    ids_a[0],
                    ids_a[1] if len(ids_a) == 2 else None,
                    ids_b[0],
                    ids_b[1] if len(ids_b) == 2 else None,
                    match.winner_side,
                    match.source,
                ),
            )
            match_id = int(cursor.lastrowid or 0)
            for game_number, (score_a, score_b) in enumerate(match.games, start=1):
                self._conn.execute(
                    "INSERT INTO match_games (match_id, game_number, side_a_score, "
                    "side_b_score) VALUES (?, ?, ?, ?)",
                    (match_id, game_number, score_a, score_b),
                )
        return match_id

    def import_csv(self, csv_path: Path, today: dt.date | None = None) -> ImportReport:
        """Row-level import: gate failures quarantine the row, never abort the file."""
        table = pd.read_csv(csv_path, comment="#")
        added = 0
        rejected: list[tuple[int, str]] = []
        for offset, row in enumerate(table.itertuples(index=False)):
            try:
                games = tuple(
                    (int(a), int(b)) for a, b in (g.split("-") for g in str(row.games).split(","))
                )
                match = NewMatch(
                    match_type=str(row.match_type),
                    played_at=dt.date.fromisoformat(str(row.played_at)),
                    side_a=tuple(str(row.side_a).split("|")),
                    side_b=tuple(str(row.side_b).split("|")),
                    winner_side=str(row.winner_side),
                    games=games,
                    source=f"import:{csv_path.name}",
                )
                for name in (*match.side_a, *match.side_b):
                    try:
                        self.player_id(name)
                    except ValidationGateError:
                        self.add_player(name)
                self.add_match(match, today)
                added += 1
            except (ValidationGateError, ValueError) as exc:
                rejected.append((offset + 2, str(exc)))  # +2: header line + 1-indexing
        return ImportReport(added=added, rejected=tuple(rejected))

    def revalidate(self, today: dt.date | None = None) -> list[tuple[int, str]]:
        """Re-run every gate over stored matches; external edits surface here."""
        query = """
            SELECT m.id, m.match_type, m.played_at, m.winner_side,
                   pa1.canonical_name, pa2.canonical_name,
                   pb1.canonical_name, pb2.canonical_name
            FROM matches m
            JOIN players pa1 ON pa1.id = m.side_a_p1
            LEFT JOIN players pa2 ON pa2.id = m.side_a_p2
            JOIN players pb1 ON pb1.id = m.side_b_p1
            LEFT JOIN players pb2 ON pb2.id = m.side_b_p2
            ORDER BY m.id
        """
        violations: list[tuple[int, str]] = []
        for mid, mtype, played_at, winner_side, a1, a2, b1, b2 in self._conn.execute(query):
            games = tuple(
                (int(a), int(b))
                for a, b in self._conn.execute(
                    "SELECT side_a_score, side_b_score FROM match_games "
                    "WHERE match_id = ? ORDER BY game_number",
                    (mid,),
                )
            )
            match = NewMatch(
                match_type=str(mtype),
                played_at=dt.date.fromisoformat(played_at),
                side_a=(a1,) if a2 is None else (a1, a2),
                side_b=(b1,) if b2 is None else (b1, b2),
                winner_side=str(winner_side),
                games=games,
            )
            try:
                validate_match(match, today if today is not None else dt.date.today())
            except ValidationGateError as exc:
                violations.append((int(mid), str(exc)))
        return violations

    def as_match_table(self) -> pd.DataFrame:
        """All matches in the shared MatchMeta frame contract (winner/loser as names)."""
        query = """
            SELECT m.id, m.match_type, m.played_at, m.winner_side,
                   pa.canonical_name AS a1, pb.canonical_name AS b1,
                   (SELECT COUNT(*) FROM match_games g WHERE g.match_id = m.id) AS n_games
            FROM matches m
            JOIN players pa ON pa.id = m.side_a_p1
            JOIN players pb ON pb.id = m.side_b_p1
            WHERE m.match_type = 'singles'
            ORDER BY m.played_at, m.id
        """
        rows = []
        for mid, _mtype, played_at, winner_side, a1, b1, n_games in self._conn.execute(query):
            winner, loser = (a1, b1) if winner_side == "A" else (b1, a1)
            rows.append(
                {
                    "match_uid": f"club:{mid}",
                    "source": "club",
                    "video": "",
                    "date": dt.date.fromisoformat(played_at),
                    "date_precision": "day",
                    "tournament": "club",
                    "round_name": "Club",
                    "winner": winner,
                    "loser": loser,
                    "n_sets": int(n_games),
                    "duration_min": None,
                }
            )
        return pd.DataFrame(rows, columns=list(MATCH_TABLE_COLUMNS))

    def as_timeline(self) -> pd.DataFrame:
        """Harness-ready singles timeline: t_idx stamped, discipline = singles.

        Doubles prediction arrives via pair composition in a later increment;
        doubles rows stay in the DB but are not emitted here (DECISIONS.md).
        """
        frame = self.as_match_table()
        frame = frame.sort_values(["date", "match_uid"]).reset_index(drop=True)
        frame["round_rank"] = 0
        frame["t_idx"] = range(len(frame))
        frame["discipline"] = "singles"
        return frame
