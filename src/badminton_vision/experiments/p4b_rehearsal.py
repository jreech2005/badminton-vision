"""P4b rehearsal: A records-only vs B records+tactical vs C tactical-only.

The perfect-label upper bound: if clean-label tactical features add no lift
here, noisy CV-derived features will not either. Feature sets and contrasts
are preregistered in configs/experiment_p4b_v1.yaml.
"""

import dataclasses
import json
from pathlib import Path

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict

from badminton_vision.errors import DataContractError
from badminton_vision.eval.bootstrap import paired_bootstrap_brier
from badminton_vision.eval.harness import HarnessConfig, RunResult, run_walk_forward
from badminton_vision.eval.metrics import calibration_table, summarize
from badminton_vision.eval.models import ModelsConfig, make_lr_predictor, make_xgb_predictor
from badminton_vision.eval.predictors import Predictor, make_baseline_predictors
from badminton_vision.features.match_row import FEATURE_COLUMNS, build_match_rows
from badminton_vision.features.tactical import (
    TACTICAL_COLUMNS,
    TacticalParams,
    build_tactical_rows,
)
from badminton_vision.ingest.labels import LabelMap


class P4bConfig(BaseModel):
    """Preregistered rehearsal configuration."""

    model_config = ConfigDict(frozen=True)

    version: str
    tactical: TacticalParams
    contrasts: tuple[tuple[str, str], ...]


def load_p4b_config(path: Path) -> P4bConfig:
    """Load and validate the preregistered P4b YAML."""
    with path.open(encoding="utf-8") as fh:
        return P4bConfig.model_validate(yaml.safe_load(fh))


FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "a": FEATURE_COLUMNS,
    "b": (*FEATURE_COLUMNS, *TACTICAL_COLUMNS),
    "c": TACTICAL_COLUMNS,
}


def run_p4b(
    timeline: pd.DataFrame,
    strokes_by_uid: dict[str, pd.DataFrame],
    label_map: LabelMap,
    harness_config: HarnessConfig,
    models_config: ModelsConfig,
    p4b_config: P4bConfig,
    run_dir: Path | None = None,
) -> tuple[RunResult, dict[str, object]]:
    """Run the A/B/C walk-forward and the preregistered contrast report."""
    missing = {str(uid) for uid in timeline["match_uid"]} - set(strokes_by_uid)
    if missing:
        raise DataContractError(f"{len(missing)} timeline matches lack stroke frames")
    records_rows = build_match_rows(
        timeline, harness_config.elo, harness_config.log5, harness_config.seed
    )
    tactical_rows = build_tactical_rows(
        timeline, strokes_by_uid, label_map, p4b_config.tactical, harness_config.seed
    )
    merged = records_rows.merge(
        tactical_rows[["match_uid", *TACTICAL_COLUMNS]], on="match_uid", validate="1:1"
    )
    all_columns = FEATURE_SETS["b"]
    frame = merged.set_index("match_uid")[list(all_columns)].astype(float)
    features = {str(uid): frame.loc[uid] for uid in frame.index}

    predictors: list[Predictor] = list(
        make_baseline_predictors(
            ["coinflip", "elo", "log5"], harness_config.elo, harness_config.log5
        )
    )
    for set_name, columns in FEATURE_SETS.items():
        predictors.append(make_lr_predictor(models_config, columns, name=f"lr_{set_name}"))
        predictors.append(
            make_xgb_predictor(
                models_config, columns, name=f"xgb_{set_name}", strict_monotone=set_name != "c"
            )
        )
    result = run_walk_forward(
        timeline,
        predictors,
        harness_config,
        features_by_uid=features,
        run_dir=run_dir,
        extra_versions={
            "features": "match_row_v1+tactical_v1",
            "models": models_config.version,
            "experiment": p4b_config.version,
            "label_map": label_map.version,
        },
    )
    report = build_p4b_report(result, harness_config, p4b_config)
    if run_dir is not None:
        (run_dir / "p4b_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return result, report


def build_p4b_report(
    result: RunResult, harness_config: HarnessConfig, p4b_config: P4bConfig
) -> dict[str, object]:
    """Summaries, preregistered contrasts, and scored-window calibration tables."""
    report: dict[str, object] = {"experiment": p4b_config.version, "windows": {}}
    windows = {
        "full": result.per_match,
        "scored": result.per_match[result.per_match["scored"]],
    }
    for window_name, frame in windows.items():
        contrasts = {}
        for challenger, champion in p4b_config.contrasts:
            boot = paired_bootstrap_brier(
                frame,
                challenger,
                champion,
                n_boot=harness_config.bootstrap.n_boot,
                seed=harness_config.bootstrap.seed,
            )
            contrasts[f"{challenger}_vs_{champion}"] = dataclasses.asdict(boot)
        report["windows"][window_name] = {  # type: ignore[index]
            "summary": summarize(frame).to_dict(orient="records"),
            "contrasts": contrasts,
        }
    scored = windows["scored"]
    calibration = {}
    for name, group in scored.groupby("predictor", sort=True):
        calibration[str(name)] = calibration_table(
            group["p"], group["y"], harness_config.calibration_buckets
        ).to_dict(orient="records")
    report["calibration_scored"] = calibration
    return report
