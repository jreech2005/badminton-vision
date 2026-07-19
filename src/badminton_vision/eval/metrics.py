"""Scoring rules and calibration: pure functions of prediction/label arrays."""

from typing import Final

import numpy as np
import pandas as pd

LOG_LOSS_EPS: Final[float] = 1e-15  # clip bound; keeps log loss finite at p in {0, 1}


def brier(p: "pd.Series[float]", y: "pd.Series[int]") -> "pd.Series[float]":
    """Per-match squared error; 0.5 predictions score float-exactly 0.25."""
    return (p - y) ** 2


def log_loss(p: "pd.Series[float]", y: "pd.Series[int]") -> "pd.Series[float]":
    """Per-match negative log likelihood, clipped away from 0 and 1."""
    clipped = p.clip(LOG_LOSS_EPS, 1.0 - LOG_LOSS_EPS)
    return -(y * np.log(clipped) + (1 - y) * np.log(1.0 - clipped))


def accuracy(p: "pd.Series[float]", y: "pd.Series[int]") -> float:
    """Directional accuracy with half credit at exactly 0.5 (the coin flip scores 0.5)."""
    correct = ((p > 0.5) & (y == 1)) | ((p < 0.5) & (y == 0))
    return float(correct.mean() + 0.5 * (p == 0.5).mean())


def calibration_table(p: "pd.Series[float]", y: "pd.Series[int]", n_buckets: int) -> pd.DataFrame:
    """Bucketed predicted-vs-empirical rates; small N wants few buckets."""
    edges = np.linspace(0.0, 1.0, n_buckets + 1)
    bucket = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, n_buckets - 1)
    rows = []
    for b in range(n_buckets):
        mask = bucket == b
        rows.append(
            {
                "bucket_low": edges[b],
                "bucket_high": edges[b + 1],
                "n": int(mask.sum()),
                "mean_p": float(p[mask].mean()) if mask.any() else float("nan"),
                "empirical_rate": float(y[mask].mean()) if mask.any() else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def summarize(per_match: pd.DataFrame) -> pd.DataFrame:
    """Per-predictor mean Brier/log-loss/accuracy over a per-match prediction log."""
    summaries = []
    for name, group in per_match.groupby("predictor", sort=True):
        summaries.append(
            {
                "predictor": name,
                "n": len(group),
                "brier": float(group["brier"].mean()),
                "log_loss": float(group["log_loss"].mean()),
                "accuracy": accuracy(group["p"], group["y"]),
            }
        )
    return pd.DataFrame(summaries)
