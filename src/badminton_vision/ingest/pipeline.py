"""The assembled ingest pipeline: raw + curated inputs into the global timeline."""

import pandas as pd

from badminton_vision import paths
from badminton_vision.ingest.aliases import load_player_aliases
from badminton_vision.ingest.badminton_db import load_bdb_match_table, load_date_overrides
from badminton_vision.ingest.dedup import dedup_matches
from badminton_vision.ingest.shuttleset import (
    SHUTTLESET22_SPEC,
    SHUTTLESET_SPEC,
    load_match_table,
    load_strokes,
)
from badminton_vision.ingest.timeline import build_timeline


def build_global_timeline() -> pd.DataFrame:
    """Load all three sources, dedup, and return the strict 103-match timeline."""
    aliases = load_player_aliases(paths.CURATED_DIR / "player_aliases.csv")
    overrides = load_date_overrides(paths.CURATED_DIR / "match_date_overrides.csv")
    all_matches = pd.concat(
        [
            load_match_table(SHUTTLESET_SPEC, aliases),
            load_match_table(SHUTTLESET22_SPEC, aliases),
            load_bdb_match_table(paths.BADMINTON_DB_JSON_DIR, aliases, overrides),
        ],
        ignore_index=True,
    )
    unique, _dropped, _near = dedup_matches(all_matches)
    return build_timeline(unique)


def build_elite_timeline() -> pd.DataFrame:
    """The 94-match ShuttleSet-family subset (stroke data exists for every row)."""
    timeline = build_global_timeline()
    elite = timeline[timeline["source"] != "badminton_db"].drop(
        columns=["t_idx", "round_rank", "discipline"]
    )
    return build_timeline(elite.reset_index(drop=True))


def load_elite_strokes(timeline: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Loader stroke frames keyed by match_uid for every ShuttleSet-family row."""
    specs = {SHUTTLESET_SPEC.name: SHUTTLESET_SPEC, SHUTTLESET22_SPEC.name: SHUTTLESET22_SPEC}
    strokes: dict[str, pd.DataFrame] = {}
    for match in timeline.itertuples(index=False):
        source = str(match.source)
        if source not in specs:
            continue
        strokes[str(match.match_uid)] = load_strokes(
            specs[source], str(match.match_uid), str(match.video), int(str(match.n_sets))
        )
    return strokes
