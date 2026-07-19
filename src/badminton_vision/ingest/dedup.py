"""Cross-dataset twin detection: run before any shared timeline exists."""

import datetime as dt
from typing import Final

import pandas as pd

from badminton_vision.errors import DataContractError

# On a duplicate, the earlier-priority source's row is kept.
SOURCE_PRIORITY: Final[dict[str, int]] = {"shuttleset": 0, "shuttleset22": 1, "badminton_db": 2}
NEAR_MISS_WINDOW_DAYS: Final[int] = 3  # same pair within this window -> manual-review audit


def _pair_key(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(
        [tuple(sorted((w, lo))) for w, lo in zip(frame["winner"], frame["loser"], strict=True)],
        index=frame.index,
    )


def dedup_matches(
    matches: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split matches into (unique, dropped_duplicates, near_miss_audit).

    Duplicates are keyed on (player pair, exact day); only day-precision rows
    can match the key (week-precision dates are not exact — see DECISIONS.md).
    Conflicting winners inside one duplicate group are a cross-source
    disagreement and raise instead of silently resolving.
    """
    frame = matches.copy()
    unknown_sources = set(frame["source"]) - set(SOURCE_PRIORITY)
    if unknown_sources:
        raise DataContractError(f"sources without dedup priority: {sorted(unknown_sources)}")
    frame["_pair"] = _pair_key(frame)
    frame["_priority"] = frame["source"].map(SOURCE_PRIORITY)

    day_rows = frame[frame["date_precision"] == "day"]
    kept_idx: list[int] = list(frame.index[frame["date_precision"] != "day"])
    dropped_rows: list[pd.Series] = []
    for (_, _), group in day_rows.groupby(["_pair", "date"], sort=False):
        if len(set(group["winner"])) > 1:
            raise DataContractError(
                f"duplicate group {sorted(group['match_uid'])} disagrees on the winner"
            )
        ordered = group.sort_values(["_priority", "match_uid"])
        kept = ordered.iloc[0]
        kept_idx.append(int(ordered.index[0]))
        for _, row in ordered.iloc[1:].iterrows():
            dropped = row.copy()
            dropped["kept_match_uid"] = kept["match_uid"]
            dropped_rows.append(dropped)

    unique = frame.loc[sorted(kept_idx)].drop(columns=["_pair", "_priority"]).reset_index(drop=True)
    dropped_frame = (
        pd.DataFrame(dropped_rows).drop(columns=["_pair", "_priority"]).reset_index(drop=True)
        if dropped_rows
        else pd.DataFrame(columns=[*matches.columns, "kept_match_uid"])
    )
    return unique, dropped_frame, _near_misses(unique)


def _near_misses(unique: pd.DataFrame) -> pd.DataFrame:
    """Same pair on different days within the window: legit rematches or dirty dates."""
    frame = unique.copy()
    frame["_pair"] = _pair_key(frame)
    audits: list[dict[str, object]] = []
    for _, group in frame.groupby("_pair", sort=False):
        rows = group.sort_values("date")
        for (_, a), (_, b) in zip(rows.iterrows(), rows.iloc[1:].iterrows(), strict=False):
            a_date, b_date = a["date"], b["date"]
            assert isinstance(a_date, dt.date)
            assert isinstance(b_date, dt.date)
            gap = (b_date - a_date).days
            if 0 < gap <= NEAR_MISS_WINDOW_DAYS:
                audits.append(
                    {
                        "match_uid_a": a["match_uid"],
                        "match_uid_b": b["match_uid"],
                        "date_a": a_date,
                        "date_b": b_date,
                        "gap_days": gap,
                        "pair": a["_pair"],
                    }
                )
    return pd.DataFrame(
        audits, columns=["match_uid_a", "match_uid_b", "date_a", "date_b", "gap_days", "pair"]
    )
