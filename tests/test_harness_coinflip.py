"""Harness verification: the coin-flip gate, predict-before-reveal, manifests."""

import json
from pathlib import Path

import pandas as pd
import pytest

from badminton_vision import paths
from badminton_vision.errors import DataContractError
from badminton_vision.eval.harness import load_harness_config, run_walk_forward
from badminton_vision.eval.metrics import summarize
from badminton_vision.eval.predictors import (
    CoinFlip,
    EloPredictor,
    Log5Predictor,
    MatchContext,
    MatchResult,
    make_baseline_predictors,
)
from badminton_vision.ingest.pipeline import build_global_timeline
from tests.conftest import cell, make_synthetic_timeline

CONFIG = load_harness_config(paths.CONFIGS_DIR / "harness_v1.yaml")


@pytest.mark.unit
def test_coinflip_brier_exact() -> None:
    # THE harness gate: any per-match Brier != 0.25 means the harness is broken.
    result = run_walk_forward(make_synthetic_timeline(), [CoinFlip()], CONFIG)
    assert (result.per_match["brier"] == 0.25).all()
    assert result.per_match["brier"].mean() == 0.25
    assert result.summary.loc[0, "brier"] == 0.25


class _Probe:
    """Records how many outcomes were revealed before each predict call."""

    name = "probe"

    def __init__(self) -> None:
        self.observed = 0
        self.revealed_at_predict: list[int] = []

    def predict(self, _ctx: MatchContext) -> float:
        self.revealed_at_predict.append(self.observed)
        return 0.5

    def observe(self, _result: MatchResult) -> None:
        self.observed += 1


@pytest.mark.unit
def test_predict_precedes_reveal() -> None:
    timeline = make_synthetic_timeline(12)
    probe = _Probe()
    run_walk_forward(timeline, [probe], CONFIG)
    # For match k the predictor must have seen exactly k outcomes, never k+1.
    assert probe.revealed_at_predict == list(range(12))


@pytest.mark.unit
def test_out_of_bounds_prediction_raises() -> None:
    class Bad:
        name = "bad"

        def predict(self, _ctx: MatchContext) -> float:
            return 1.5

        def observe(self, _result: MatchResult) -> None:
            return None

    with pytest.raises(DataContractError, match="predicted"):
        run_walk_forward(make_synthetic_timeline(3), [Bad()], CONFIG)


@pytest.mark.unit
def test_duplicate_predictor_names_raise() -> None:
    with pytest.raises(DataContractError, match="duplicate predictor names"):
        run_walk_forward(make_synthetic_timeline(3), [CoinFlip(), CoinFlip()], CONFIG)


@pytest.mark.unit
def test_run_writes_parquet_and_manifest(tmp_path: Path) -> None:
    timeline = make_synthetic_timeline(6)
    run_dir = tmp_path / "run"
    result = run_walk_forward(
        timeline, [CoinFlip(), EloPredictor(CONFIG.elo)], CONFIG, run_dir=run_dir
    )
    stored = pd.read_parquet(run_dir / "predictions.parquet")
    assert len(stored) == 6 * 2
    manifest = json.loads((run_dir / "manifest.json").read_text())
    required = {"git_sha", "config_sha256", "seed", "timeline_sha256", "packages", "predictors"}
    assert required <= set(manifest)
    assert manifest == result.manifest


@pytest.mark.unit
def test_elo_and_log5_beat_coinflip_on_biased_synthetic() -> None:
    # Sanity gate on a deterministic 90%-dominant player, not a statistical claim.
    timeline = make_synthetic_timeline(60, strong_period=10)
    result = run_walk_forward(
        timeline, [EloPredictor(CONFIG.elo), Log5Predictor(CONFIG.log5)], CONFIG
    )
    scored = result.per_match[result.per_match["scored"]]
    by_predictor = summarize(scored).set_index("predictor")
    assert cell(by_predictor, "elo", "brier") < 0.25
    assert cell(by_predictor, "log5", "brier") < 0.25


@pytest.mark.integration
def test_real_timeline_run() -> None:
    timeline = build_global_timeline()
    predictors = make_baseline_predictors(["coinflip", "elo", "log5"], CONFIG.elo, CONFIG.log5)
    result = run_walk_forward(timeline, predictors, CONFIG)
    coin = result.per_match[result.per_match["predictor"] == "coinflip"]
    assert len(coin) == 103
    assert (coin["brier"] == 0.25).all()
    scored = result.per_match[result.per_match["scored"]]
    by_predictor = summarize(scored).set_index("predictor")
    # Sanity: ratings carry signal on the elite timeline (H1 rejecting H0 is
    # claimed with the bootstrap in the experiment runners, not here).
    assert cell(by_predictor, "elo", "brier") < 0.25
    assert cell(by_predictor, "log5", "brier") < 0.25
