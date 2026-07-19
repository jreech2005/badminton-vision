"""bviz records subcommands: thin argument parsing over ClubDB."""

import argparse
import datetime as dt
from pathlib import Path

from badminton_vision import paths
from badminton_vision.records.club_db import ClubDB, NewMatch


def add_records_parser(sub: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    """Attach the records subcommand tree to the main bviz parser."""
    records = sub.add_parser("records", help="club match records")
    records_sub = records.add_subparsers(dest="records_command", required=True)
    default_db = paths.DB_DIR / "club.sqlite"

    init = records_sub.add_parser("init", help="create the club database")
    init.add_argument("--db", type=Path, default=default_db)

    add_player = records_sub.add_parser("add-player", help="register a player")
    add_player.add_argument("name")
    add_player.add_argument("--db", type=Path, default=default_db)
    add_player.add_argument("--force-new", action="store_true")

    add_match = records_sub.add_parser("add-match", help="record a completed match")
    add_match.add_argument("--db", type=Path, default=default_db)
    add_match.add_argument("--type", choices=["singles", "doubles"], required=True)
    add_match.add_argument("--date", required=True, help="ISO date, e.g. 2026-07-19")
    add_match.add_argument("--side-a", required=True, help="player name, doubles: 'P1|P2'")
    add_match.add_argument("--side-b", required=True)
    add_match.add_argument("--games", required=True, help="e.g. 21-15,19-21,21-18")
    add_match.add_argument("--winner", choices=["A", "B"], required=True)

    import_csv = records_sub.add_parser("import-csv", help="bulk import; bad rows quarantine")
    import_csv.add_argument("csv_path", type=Path)
    import_csv.add_argument("--db", type=Path, default=default_db)

    validate = records_sub.add_parser("validate", help="re-run gates over stored matches")
    validate.add_argument("--db", type=Path, default=default_db)


def handle_records(args: argparse.Namespace) -> None:
    """Dispatch a records subcommand."""
    args.db.parent.mkdir(parents=True, exist_ok=True)
    db = ClubDB(args.db)
    try:
        if args.records_command == "init":
            db.init_schema()
            print(f"initialized {args.db}")
        elif args.records_command == "add-player":
            player_id = db.add_player(args.name, force_new=args.force_new)
            print(f"player {args.name} -> id {player_id}")
        elif args.records_command == "add-match":
            games = tuple(
                (int(a), int(b)) for a, b in (g.split("-") for g in args.games.split(","))
            )
            match_id = db.add_match(
                NewMatch(
                    match_type=args.type,
                    played_at=dt.date.fromisoformat(args.date),
                    side_a=tuple(args.side_a.split("|")),
                    side_b=tuple(args.side_b.split("|")),
                    winner_side=args.winner,
                    games=games,
                )
            )
            print(f"match recorded -> id {match_id}")
        elif args.records_command == "import-csv":
            report = db.import_csv(args.csv_path)
            print(f"added {report.added}, rejected {len(report.rejected)}")
            for line, reason in report.rejected:
                print(f"  line {line}: {reason}")
        elif args.records_command == "validate":
            violations = db.revalidate()
            if violations:
                for match_id, reason in violations:
                    print(f"match {match_id}: {reason}")
            print(f"{len(violations)} violations")
    finally:
        db.close()
