"""Event-JSON data contract v1.

The interface future CV work writes to and analytics reads from. Additive-only:
fields are never removed or re-typed; a breaking change becomes events_v2 with
a new version string. Sides are symmetrized (side0/side1 per assign_sides),
never winner-first; outcome lives ONLY in MatchEventFile.outcome so the
observation stream stays outcome-free.
"""

import datetime as dt
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict


class SourceKind(StrEnum):
    """Provenance of a datum: which kind of producer emitted it."""

    human_annotation = "human_annotation"
    cv_tracknet = "cv_tracknet"
    cv_pose = "cv_pose"
    cv_fused = "cv_fused"
    manual_entry = "manual_entry"


class FrameObservation(BaseModel):
    """Per-frame channel (heavy; optional) that CV increments will populate."""

    model_config = ConfigDict(frozen=True)

    frame: int
    t_s: float
    shuttle_xy_px: tuple[float, float] | None = None
    shuttle_visible: bool
    pose_side0: list[tuple[float, float]] | None = None
    pose_side1: list[tuple[float, float]] | None = None
    confidence: float | None = None
    source: SourceKind


class ShotEvent(BaseModel):
    """One stroke; shot_class None means the label was unknown at annotation time."""

    model_config = ConfigDict(frozen=True)

    rally_idx: int
    shot_idx: int
    t_hit_s: float | None = None
    frame_hit: int | None = None
    striker: Literal["side0", "side1"]
    shot_class: str | None = None
    shot_class_version: str
    around_head: bool | None = None
    backhand: bool | None = None
    striker_xy: tuple[float, float] | None = None
    receiver_xy: tuple[float, float] | None = None
    landing_xy: tuple[float, float] | None = None
    coord_space: Literal["px", "court_m"] | None = None
    confidence: float | None = None
    source: SourceKind


class RallyEvent(BaseModel):
    """One rally; deliberately carries no winner/reason — outcome is match-level only."""

    model_config = ConfigDict(frozen=True)

    rally_idx: int
    shots: list[ShotEvent]


class MatchOutcome(BaseModel):
    """The outcome channel, separable from the observation stream."""

    model_config = ConfigDict(frozen=True)

    winner_side: Literal["side0", "side1"]
    game_scores: list[tuple[int, int]] | None = None


class MatchEventFile(BaseModel):
    """One match's full event record under contract version 1.0."""

    model_config = ConfigDict(frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    match_uid: str
    date: dt.date
    discipline: str
    side0_player: str
    side1_player: str
    homography: list[list[float]] | None = None
    rallies: list[RallyEvent]
    frames: list[FrameObservation] | None = None
    outcome: MatchOutcome | None = None
