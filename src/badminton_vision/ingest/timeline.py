"""The global timeline: one calendar, strict total order, discipline inference."""

from collections.abc import Mapping
from typing import Final

import pandas as pd

from badminton_vision.errors import DataContractError

# Within one calendar day, earlier tournament rounds sort first. Covers both
# the ShuttleSet round vocabulary and BadmintonDB's compact codes.
ROUND_RANK: Final[dict[str, int]] = {
    "Group-Stage": 0,
    "R32": 1,
    "R16": 2,
    "Quarter-finals": 3,
    "QF": 3,
    "Semi-finals": 4,
    "SF": 4,
    "Finals": 5,
    "F": 5,
}

# Component seeds for discipline inference; move to a config if this ever grows.
# Michelle LI seeds the isolated WS component {KOSETSKAYA, LI} — the pair played
# only each other in these datasets (Denmark Open 2020), so connectivity alone
# cannot label them; both are women's singles players (BWF).
DISCIPLINE_SEEDS: Final[dict[str, str]] = {
    "Kento MOMOTA": "MS",
    "An Se Young": "WS",
    "Michelle LI": "WS",
}


def infer_discipline(
    matches: pd.DataFrame, seeds: Mapping[str, str] = DISCIPLINE_SEEDS
) -> dict[str, str]:
    """Label every player via connected components of the opponent graph.

    Singles players only ever meet within their own discipline, so the graph
    must split into one component per seed; anything else is a data defect
    (most likely an alias failure) and raises.
    """
    adjacency: dict[str, set[str]] = {}
    for winner, loser in zip(matches["winner"], matches["loser"], strict=True):
        adjacency.setdefault(winner, set()).add(loser)
        adjacency.setdefault(loser, set()).add(winner)
    unvisited = set(adjacency)
    labels: dict[str, str] = {}
    while unvisited:
        stack = [unvisited.pop()]
        component = {stack[0]}
        while stack:
            for neighbour in adjacency[stack.pop()]:
                if neighbour in unvisited:
                    unvisited.remove(neighbour)
                    component.add(neighbour)
                    stack.append(neighbour)
        seed_hits = sorted(set(seeds) & component)
        if len(seed_hits) != 1:
            raise DataContractError(
                f"opponent-graph component with {len(seed_hits)} seeds "
                f"(players: {sorted(component)[:6]}...); expected exactly one"
            )
        labels.update(dict.fromkeys(component, seeds[seed_hits[0]]))
    return labels


def build_timeline(
    unique: pd.DataFrame, seeds: Mapping[str, str] = DISCIPLINE_SEEDS
) -> pd.DataFrame:
    """Sort deduped matches into a strict total order and stamp t_idx + discipline."""
    frame = unique.copy()
    if frame["date"].isna().any():
        raise DataContractError("timeline rows with null dates")
    if frame["match_uid"].duplicated().any():
        raise DataContractError("timeline rows with duplicate match_uid")
    unknown_rounds = set(frame["round_name"]) - set(ROUND_RANK)
    if unknown_rounds:
        raise DataContractError(f"rounds without a rank: {sorted(unknown_rounds)}")
    frame["round_rank"] = frame["round_name"].map(ROUND_RANK)
    frame = frame.sort_values(["date", "round_rank", "match_uid"]).reset_index(drop=True)
    frame["t_idx"] = range(len(frame))
    labels = infer_discipline(frame, seeds)
    frame["discipline"] = [labels[w] for w in frame["winner"]]
    mismatched = [
        uid
        for uid, w, lo in zip(frame["match_uid"], frame["winner"], frame["loser"], strict=True)
        if labels[w] != labels[lo]
    ]
    if mismatched:
        raise DataContractError(f"winner/loser discipline mismatch in {mismatched}")
    return frame
