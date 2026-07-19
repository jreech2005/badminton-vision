"""bviz command-line entrypoint: parse args, call library functions, print."""

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import pandas as pd

from badminton_vision import paths
from badminton_vision.errors import BadmintonVisionError
from badminton_vision.eval.harness import load_harness_config, run_walk_forward
from badminton_vision.eval.metrics import calibration_table, summarize
from badminton_vision.eval.predictors import make_baseline_predictors
from badminton_vision.ingest.extract_check import check_raw
from badminton_vision.ingest.pipeline import build_global_timeline


def _cmd_extract_check() -> None:
    report = check_raw(
        paths.RAW_DIR,
        paths.SHUTTLESET_SET_DIR,
        paths.SHUTTLESET22_SET_DIR,
        paths.BADMINTON_DB_JSON_DIR,
    )
    print(f"shuttleset: {report.shuttleset[0]} match dirs, {report.shuttleset[1]} set csvs")
    print(f"shuttleset22: {report.shuttleset22[0]} match dirs, {report.shuttleset22[1]} set csvs")
    print(f"badminton_db: {report.bdb_jsons} match jsons")
    print("data/raw is read-only")


def _cmd_timeline_show(tail: int) -> None:
    timeline = build_global_timeline()
    columns = ["t_idx", "date", "match_uid", "winner", "loser", "discipline", "round_name"]
    print(timeline[columns].tail(tail).to_string(index=False))


def _cmd_run(predictor_names: list[str], config_path: Path) -> None:
    config = load_harness_config(config_path)
    predictors = make_baseline_predictors(predictor_names, config.elo, config.log5)
    timeline = build_global_timeline()
    run_name = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    result = run_walk_forward(timeline, predictors, config, run_dir=paths.RUNS_DIR / run_name)
    print(f"run: {run_name} ({len(timeline)} matches)")
    print("full window:")
    print(result.summary.to_string(index=False))
    scored = result.per_match[result.per_match["scored"]]
    print(f"scored window (both players >= {config.scored_min_history} prior matches):")
    print(summarize(scored).to_string(index=False))


def _cmd_report(run_name: str, config_path: Path) -> None:
    config = load_harness_config(config_path)
    if run_name == "latest":
        runs = sorted(p.name for p in paths.RUNS_DIR.iterdir() if p.is_dir())
        if not runs:
            raise BadmintonVisionError(f"no runs under {paths.RUNS_DIR}")
        run_name = runs[-1]
    run_dir = paths.RUNS_DIR / run_name
    per_match = pd.read_parquet(run_dir / "predictions.parquet")
    manifest = json.loads((run_dir / "manifest.json").read_text())
    print(f"run: {run_name} (git {manifest['git_sha'][:12]}, config {manifest['config_version']})")
    print("full window:")
    print(summarize(per_match).to_string(index=False))
    scored = per_match[per_match["scored"]]
    print("scored window:")
    print(summarize(scored).to_string(index=False))
    for name, group in scored.groupby("predictor", sort=True):
        print(f"calibration ({name}, scored window):")
        table = calibration_table(group["p"], group["y"], config.calibration_buckets)
        print(table.to_string(index=False))


def main(argv: list[str] | None = None) -> int:
    """Dispatch a bviz subcommand; return the process exit code."""
    parser = argparse.ArgumentParser(prog="bviz")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("extract-check", help="verify raw-data counts and read-only lockdown")

    timeline_parser = sub.add_parser("timeline", help="inspect the global timeline")
    timeline_sub = timeline_parser.add_subparsers(dest="timeline_command", required=True)
    show = timeline_sub.add_parser("show", help="print the tail of the timeline")
    show.add_argument("--tail", type=int, default=10)

    run = sub.add_parser("run", help="walk-forward run over the global timeline")
    run.add_argument("--predictors", default="coinflip,elo,log5")
    run.add_argument("--config", type=Path, default=paths.CONFIGS_DIR / "harness_v1.yaml")

    report = sub.add_parser("report", help="summarize a stored run")
    report.add_argument("--run", default="latest")
    report.add_argument("--config", type=Path, default=paths.CONFIGS_DIR / "harness_v1.yaml")

    args = parser.parse_args(argv)
    try:
        if args.command == "extract-check":
            _cmd_extract_check()
        elif args.command == "timeline":
            _cmd_timeline_show(args.tail)
        elif args.command == "run":
            _cmd_run([n.strip() for n in args.predictors.split(",") if n.strip()], args.config)
        elif args.command == "report":
            _cmd_report(args.run, args.config)
    except BadmintonVisionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0
