"""The walk-forward engine: predict, log, then reveal — in strict t_idx order.

Owns run I/O: the per-match prediction log (parquet) and the run manifest.
No manifest, no run.
"""

import datetime as dt
import hashlib
import json
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import cast

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict

from badminton_vision import paths
from badminton_vision.errors import DataContractError
from badminton_vision.eval.metrics import brier, log_loss, summarize
from badminton_vision.eval.predictors import MatchContext, MatchResult, Predictor
from badminton_vision.ingest.preoutcome import assign_sides
from badminton_vision.ratings.elo import EloParams
from badminton_vision.ratings.log5 import Log5Params


class BootstrapParams(BaseModel):
    model_config = ConfigDict(frozen=True)

    n_boot: int
    seed: int


class HarnessConfig(BaseModel):
    """Everything a run's numbers depend on; hashed into the manifest."""

    model_config = ConfigDict(frozen=True)

    version: str
    seed: int
    elo: EloParams
    log5: Log5Params
    scored_min_history: int
    calibration_buckets: int
    bootstrap: BootstrapParams


def load_harness_config(path: Path) -> HarnessConfig:
    """Load and validate the harness YAML."""
    with path.open(encoding="utf-8") as fh:
        return HarnessConfig.model_validate(yaml.safe_load(fh))


@dataclass(frozen=True)
class RunResult:
    """A completed walk-forward run: per-match log, summary, and provenance."""

    per_match: pd.DataFrame
    summary: pd.DataFrame
    manifest: dict[str, object]
    run_dir: Path | None = field(default=None)


def _timeline_hash(timeline: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for row in timeline[["match_uid", "date", "winner", "loser"]].itertuples(index=False):
        digest.update(f"{row.match_uid}|{row.date}|{row.winner}|{row.loser}".encode())
    return digest.hexdigest()


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=paths.REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"  # policy: a missing git identity is recorded, never fatal
    return out.stdout.strip()


def build_manifest(
    config: HarnessConfig,
    timeline: pd.DataFrame,
    predictors: list[str],
    extra_versions: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Assemble the provenance record without which a run does not exist."""
    config_json = config.model_dump_json()
    return {
        "git_sha": _git_sha(),
        "config_version": config.version,
        "config_sha256": hashlib.sha256(config_json.encode()).hexdigest(),
        "seed": config.seed,
        "timeline_sha256": _timeline_hash(timeline),
        "n_matches": len(timeline),
        "predictors": predictors,
        "packages": {
            name: metadata.version(name)
            for name in ("pandas", "numpy", "scipy", "scikit-learn", "xgboost", "pydantic")
        },
        "extra_versions": dict(extra_versions or {}),
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
    }


def run_walk_forward(
    timeline: pd.DataFrame,
    predictors: list[Predictor],
    config: HarnessConfig,
    features_by_uid: Mapping[str, "pd.Series[float]"] | None = None,
    run_dir: Path | None = None,
    extra_versions: Mapping[str, str] | None = None,
) -> RunResult:
    """Walk the timeline once; every predictor predicts before any outcome is revealed."""
    names = [p.name for p in predictors]
    if len(set(names)) != len(names):
        raise DataContractError(f"duplicate predictor names: {names}")
    if list(timeline["t_idx"]) != list(range(len(timeline))):
        raise DataContractError("timeline t_idx is not contiguous from 0; rebuild it")

    n_played: dict[str, int] = {}
    rows: list[dict[str, object]] = []
    for match in timeline.itertuples(index=False):
        match_uid = str(match.match_uid)
        winner, loser = str(match.winner), str(match.loser)
        side0, side1, y = assign_sides(match_uid, winner, loser, config.seed)
        ctx = MatchContext(
            match_uid=match_uid,
            t_idx=cast(int, match.t_idx),
            date=cast(dt.date, match.date),
            discipline=str(match.discipline),
            side0=side0,
            side1=side1,
            n_prior_side0=n_played.get(side0, 0),
            n_prior_side1=n_played.get(side1, 0),
            features=None if features_by_uid is None else features_by_uid[match_uid],
        )
        for predictor in predictors:
            p = predictor.predict(ctx)
            if not 0.0 < p < 1.0:
                raise DataContractError(f"{predictor.name} predicted {p} for {match_uid}")
            rows.append(
                {
                    "match_uid": match_uid,
                    "t_idx": ctx.t_idx,
                    "date": ctx.date.isoformat(),
                    "discipline": ctx.discipline,
                    "side0": side0,
                    "side1": side1,
                    "predictor": predictor.name,
                    "p": p,
                    "y": y,
                    "n_prior_side0": ctx.n_prior_side0,
                    "n_prior_side1": ctx.n_prior_side1,
                }
            )
        result = MatchResult(ctx=ctx, y=y)
        for predictor in predictors:
            predictor.observe(result)
        n_played[winner] = n_played.get(winner, 0) + 1
        n_played[loser] = n_played.get(loser, 0) + 1

    per_match = pd.DataFrame(rows)
    per_match["brier"] = brier(per_match["p"], per_match["y"])
    per_match["log_loss"] = log_loss(per_match["p"], per_match["y"])
    per_match["scored"] = (per_match["n_prior_side0"] >= config.scored_min_history) & (
        per_match["n_prior_side1"] >= config.scored_min_history
    )
    summary = summarize(per_match)
    manifest = build_manifest(config, timeline, names, extra_versions)
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        per_match.to_parquet(run_dir / "predictions.parquet", index=False)
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return RunResult(per_match=per_match, summary=summary, manifest=manifest, run_dir=run_dir)
