"""P4a: do records-only models add lift over the rating baselines?

H2 (records models) challenges H1 (Elo/log5) with paired-bootstrap Brier
contrasts on both the full and scored windows. LR carries the claims at this
sample size; XGB is rehearsal machinery (DECISIONS.md).
"""

import dataclasses
import json
from pathlib import Path

import pandas as pd

from badminton_vision.eval.bootstrap import paired_bootstrap_brier
from badminton_vision.eval.harness import HarnessConfig, RunResult, run_walk_forward
from badminton_vision.eval.metrics import summarize
from badminton_vision.eval.models import ModelsConfig, make_lr_predictor, make_xgb_predictor
from badminton_vision.eval.predictors import Predictor, make_baseline_predictors
from badminton_vision.features.match_row import FEATURE_COLUMNS, build_match_rows

# Preregistered contrasts: challenger vs champion, most important first.
CONTRASTS: tuple[tuple[str, str], ...] = (
    ("lr", "elo"),
    ("lr", "log5"),
    ("xgb", "elo"),
    ("xgb", "lr"),
    ("elo", "coinflip"),
    ("log5", "coinflip"),
)


def features_by_uid(rows: pd.DataFrame) -> dict[str, "pd.Series[float]"]:
    """Index the feature slice (never y or identities) by match_uid for the harness."""
    frame = rows.set_index("match_uid")[list(FEATURE_COLUMNS)].astype(float)
    return {str(uid): frame.loc[uid] for uid in frame.index}


def run_p4a(
    timeline: pd.DataFrame,
    harness_config: HarnessConfig,
    models_config: ModelsConfig,
    run_dir: Path | None = None,
) -> tuple[RunResult, dict[str, object]]:
    """Run the P4a walk-forward and bootstrap report; returns (run, report)."""
    rows = build_match_rows(timeline, harness_config.elo, harness_config.log5, harness_config.seed)
    features = features_by_uid(rows)
    predictors: list[Predictor] = [
        *make_baseline_predictors(
            ["coinflip", "elo", "log5"], harness_config.elo, harness_config.log5
        ),
        make_lr_predictor(models_config, FEATURE_COLUMNS),
        make_xgb_predictor(models_config, FEATURE_COLUMNS),
    ]
    result = run_walk_forward(
        timeline,
        predictors,
        harness_config,
        features_by_uid=features,
        run_dir=run_dir,
        extra_versions={"features": "match_row_v1", "models": models_config.version},
    )
    report = build_report(result, harness_config)
    if run_dir is not None:
        (run_dir / "p4a_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return result, report


def build_report(result: RunResult, harness_config: HarnessConfig) -> dict[str, object]:
    """Summaries + preregistered contrasts for full and scored windows."""
    report: dict[str, object] = {"windows": {}}
    windows = {
        "full": result.per_match,
        "scored": result.per_match[result.per_match["scored"]],
    }
    present = set(result.per_match["predictor"])
    for window_name, frame in windows.items():
        contrasts = {}
        for challenger, champion in CONTRASTS:
            if challenger not in present or champion not in present:
                continue
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
    return report
