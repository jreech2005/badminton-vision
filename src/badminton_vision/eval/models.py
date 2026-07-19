"""Model predictors: refit-on-revealed-history classifiers over feature rows.

Leak posture: a model predictor accumulates training rows exclusively through
observe(); it never sees the feature table, so future rows cannot reach a fit.
"""

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from badminton_vision.errors import DataContractError
from badminton_vision.eval.predictors import MatchContext, MatchResult


class LrParams(BaseModel):
    model_config = ConfigDict(frozen=True)

    c: float


class XgbParams(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_depth: int
    n_estimators: int
    learning_rate: float
    min_child_weight: int
    subsample: float
    colsample_bytree: float
    reg_lambda: float
    seed: int


class ModelsConfig(BaseModel):
    """Hyperparameters for the claim-bearing LR and the rehearsal XGB."""

    model_config = ConfigDict(frozen=True)

    version: str
    min_train: int
    clip_eps: float
    lr: LrParams
    xgb: XgbParams
    monotone_increasing: tuple[str, ...]


def load_models_config(path: Path) -> ModelsConfig:
    """Load and validate the models YAML."""
    with path.open(encoding="utf-8") as fh:
        return ModelsConfig.model_validate(yaml.safe_load(fh))


class RefitModelPredictor:
    """Refit a fresh classifier on all revealed history before each prediction."""

    def __init__(
        self,
        name: str,
        feature_columns: Sequence[str],
        make_model: Callable[[], Any],
        min_train: int,
        clip_eps: float,
    ) -> None:
        self.name = name
        self._feature_columns = list(feature_columns)
        self._make_model = make_model
        self._min_train = min_train
        self._clip_eps = clip_eps
        self._features: list[pd.Series[float]] = []
        self._labels: list[int] = []

    def predict(self, ctx: MatchContext) -> float:
        if ctx.features is None:
            raise DataContractError(f"{self.name} needs feature rows; harness passed none")
        # Fallback chain floor: with too little history the honest answer is 0.5.
        if len(self._labels) < self._min_train or len(set(self._labels)) < 2:
            return 0.5
        train_x = pd.DataFrame(self._features)[self._feature_columns]
        model = self._make_model()
        model.fit(train_x, np.asarray(self._labels))
        row = ctx.features[self._feature_columns].to_frame().T
        probability = float(model.predict_proba(row)[0, 1])
        return float(np.clip(probability, self._clip_eps, 1.0 - self._clip_eps))

    def observe(self, result: MatchResult) -> None:
        if result.ctx.features is None:
            raise DataContractError(f"{self.name} observed a result without features")
        self._features.append(result.ctx.features)
        self._labels.append(result.y)


def make_lr_predictor(config: ModelsConfig, feature_columns: Sequence[str]) -> RefitModelPredictor:
    """L2 logistic regression: median-impute + missingness flags + standardize."""

    def build() -> Pipeline:
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler()),
                ("lr", LogisticRegression(C=config.lr.c, max_iter=1000)),
            ]
        )

    return RefitModelPredictor("lr", feature_columns, build, config.min_train, config.clip_eps)


def make_xgb_predictor(config: ModelsConfig, feature_columns: Sequence[str]) -> RefitModelPredictor:
    """Heavily regularized XGBoost with NaN passthrough and monotone constraints."""
    unknown = set(config.monotone_increasing) - set(feature_columns)
    if unknown:
        raise DataContractError(f"monotone constraint on unknown features: {sorted(unknown)}")
    constraints = tuple(
        1 if column in config.monotone_increasing else 0 for column in feature_columns
    )

    def build() -> XGBClassifier:
        return XGBClassifier(
            max_depth=config.xgb.max_depth,
            n_estimators=config.xgb.n_estimators,
            learning_rate=config.xgb.learning_rate,
            min_child_weight=config.xgb.min_child_weight,
            subsample=config.xgb.subsample,
            colsample_bytree=config.xgb.colsample_bytree,
            reg_lambda=config.xgb.reg_lambda,
            monotone_constraints=constraints,
            random_state=config.xgb.seed,
            n_jobs=1,
            tree_method="hist",
            eval_metric="logloss",
        )

    return RefitModelPredictor("xgb", feature_columns, build, config.min_train, config.clip_eps)
