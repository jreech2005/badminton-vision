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
