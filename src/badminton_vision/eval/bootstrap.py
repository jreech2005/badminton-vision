"""Paired cluster bootstrap over matches — the design's only uncertainty currency."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from badminton_vision.errors import DataContractError


@dataclass(frozen=True)
class BootResult:
    """Bootstrap contrast of per-match Brier between two predictors (a minus b)."""

    mean_diff: float
    ci_low: float
    ci_high: float
    p_value: float
    n_matches: int
    n_boot: int


def paired_bootstrap_brier(
    per_match: pd.DataFrame, pred_a: str, pred_b: str, n_boot: int, seed: int
) -> BootResult:
    """Resample matches (the cluster unit) of paired Brier differences."""
    pivot = per_match.pivot_table(index="match_uid", columns="predictor", values="brier")
    if pred_a not in pivot.columns or pred_b not in pivot.columns:
        raise DataContractError(f"predictors {pred_a!r}/{pred_b!r} not both present in the log")
    diffs = (pivot[pred_a] - pivot[pred_b]).dropna().to_numpy()
    if diffs.size == 0:
        raise DataContractError("no overlapping matches between the two predictors")
    rng = np.random.default_rng(seed)
    samples = rng.choice(diffs, size=(n_boot, diffs.size), replace=True).mean(axis=1)
    ci_low, ci_high = np.quantile(samples, [0.025, 0.975])
    # Two-sided bootstrap p: how often the resampled mean lands on the far side of zero.
    tail = min(float((samples <= 0).mean()), float((samples >= 0).mean()))
    return BootResult(
        mean_diff=float(diffs.mean()),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        p_value=min(1.0, 2.0 * tail),
        n_matches=int(diffs.size),
        n_boot=n_boot,
    )
