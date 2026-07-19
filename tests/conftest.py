"""Shared synthetic fixture builders; unit tests never touch real data."""

from pathlib import Path
from typing import cast

import pandas as pd

from badminton_vision.ingest.shuttleset import PROVENANCE_COLUMNS, STROKE_COLUMNS


def make_loader_frame(n_rallies: int = 2, strokes_per_rally: int = 3) -> pd.DataFrame:
    """Build a loader-shaped stroke frame (30 contract columns + provenance)."""
    rows: list[dict[str, object]] = []
    for rally in range(1, n_rallies + 1):
        for ball_round in range(1, strokes_per_rally + 1):
            is_last = ball_round == strokes_per_rally
            rows.append(
                {
                    "rally": rally,
                    "ball_round": ball_round,
                    "time": "00:00:01",
                    "frame_num": rally * 100 + ball_round,
                    "roundscore_A": rally,
                    "roundscore_B": rally - 1,
                    "player": "A" if ball_round % 2 == 1 else "B",
                    "server": 2,
                    "type": "發短球" if ball_round == 1 else "放小球",
                    "aroundhead": 0,
                    "backhand": 0,
                    "hit_height": 1,
                    "hit_area": 1,
                    "hit_x": 100.0,
                    "hit_y": 200.0,
                    "landing_height": 1,
                    "landing_area": 2,
                    "landing_x": 150.0,
                    "landing_y": 250.0,
                    "lose_reason": "出界" if is_last else float("nan"),
                    "win_reason": "對手失誤" if is_last else float("nan"),
                    "getpoint_player": "A" if is_last else float("nan"),
                    "flaw": float("nan"),
                    "player_location_area": 1,
                    "player_location_x": 100.0,
                    "player_location_y": 300.0,
                    "opponent_location_area": 2,
                    "opponent_location_x": 120.0,
                    "opponent_location_y": 320.0,
                    "db": float("nan"),
                }
            )
    frame = pd.DataFrame(rows, columns=list(STROKE_COLUMNS))
    frame["source"] = "synthetic"
    frame["match_uid"] = "syn:1"
    frame["video"] = "Synthetic_Match"
    frame["set_no"] = 1
    return frame


def write_mini_dataset(root: Path, *, float_dates: bool = False, homography: bool = True) -> Path:
    """Write a two-match ShuttleSet-shaped tree under root/set and return that set dir."""
    set_dir = root / "set"
    set_dir.mkdir(parents=True)
    year, month, day = ("2022.0", "3.0", "5.0") if float_dates else ("2022", "3", "5")
    matches = pd.DataFrame(
        {
            "id": [1, 2],
            "video": ["Alpha_Beta_Open_Finals", "Gamma_Delta_Open_SemiFinals"],
            "tournament": ["Open", "Open"],
            "round": ["Finals", "SemiFinals"],
            "year": [year, year],
            "month": [month, month],
            "day": [day, "6.0" if float_dates else "6"],
            "set": [2, 1],
            "duration": [44, 51],
            "winner": ["Alpha", "Gamma"],
            "loser": ["Beta", "Delta"],
            "downcourt": [0, 1],
            "url": ["u1", "u2"],
        }
    )
    matches.to_csv(set_dir / "match.csv", index=False)
    stroke_cols = make_loader_frame()[list(STROKE_COLUMNS)]
    for video, n_sets in (("Alpha_Beta_Open_Finals", 2), ("Gamma_Delta_Open_SemiFinals", 1)):
        match_dir = set_dir / video
        match_dir.mkdir()
        for set_no in range(1, n_sets + 1):
            stroke_cols.to_csv(match_dir / f"set{set_no}.csv", index=False)
    if homography:
        pd.DataFrame(
            {
                "id": [1, 2],
                "video": ["Alpha_Beta_Open_Finals", "Gamma_Delta_Open_SemiFinals"],
                "homography_matrix": ["[[1,0,0],[0,1,0],[0,0,1]]"] * 2,
                "upleft_x": [0, 0],
            }
        ).to_csv(set_dir / "homography.csv", index=False)
    return set_dir


def cell(frame: pd.DataFrame, row: int | str, col: str) -> float:
    """Read one numeric cell; cast around pandas-stubs' wide .loc union type."""
    return cast(float, frame.loc[row, col])


def make_synthetic_timeline(n_matches: int = 24, strong_period: int = 10) -> pd.DataFrame:
    """Four MS players; player S wins every match except each strong_period-th."""
    import datetime as dt

    from badminton_vision.ingest.shuttleset import MATCH_TABLE_COLUMNS
    from badminton_vision.ingest.timeline import build_timeline

    players = ["S", "P1", "P2", "P3"]
    rows = []
    for i in range(n_matches):
        opponent = players[1 + i % 3]
        winner, loser = (
            ("S", opponent) if i % strong_period != strong_period - 1 else (opponent, "S")
        )
        rows.append(
            {
                "match_uid": f"syn:{i}",
                "source": "shuttleset",
                "video": f"syn_{i}",
                "date": dt.date(2021, 1, 1) + dt.timedelta(days=i),
                "date_precision": "day",
                "tournament": "T",
                "round_name": "Finals",
                "winner": winner,
                "loser": loser,
                "n_sets": 2,
                "duration_min": None,
            }
        )
    frame = pd.DataFrame(rows, columns=list(MATCH_TABLE_COLUMNS))
    return build_timeline(frame, seeds={"S": "MS"})


__all__ = ["PROVENANCE_COLUMNS", "cell", "make_loader_frame", "write_mini_dataset"]
