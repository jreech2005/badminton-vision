"""bviz command-line entrypoint: parse args, call library functions, print."""

import argparse
import sys

from badminton_vision import paths
from badminton_vision.errors import BadmintonVisionError
from badminton_vision.ingest.extract_check import check_raw


def main(argv: list[str] | None = None) -> int:
    """Dispatch a bviz subcommand; return the process exit code."""
    parser = argparse.ArgumentParser(prog="bviz")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("extract-check", help="verify raw-data counts and read-only lockdown")
    args = parser.parse_args(argv)

    try:
        if args.command == "extract-check":
            report = check_raw(
                paths.RAW_DIR,
                paths.SHUTTLESET_SET_DIR,
                paths.SHUTTLESET22_SET_DIR,
                paths.BADMINTON_DB_JSON_DIR,
            )
            print(f"shuttleset: {report.shuttleset[0]} match dirs, {report.shuttleset[1]} set csvs")
            print(
                f"shuttleset22: {report.shuttleset22[0]} match dirs, "
                f"{report.shuttleset22[1]} set csvs"
            )
            print(f"badminton_db: {report.bdb_jsons} match jsons")
            print("data/raw is read-only")
    except BadmintonVisionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0
