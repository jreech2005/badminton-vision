-- Club records schema v1. SQL constraints are the backstop; the authoritative
-- gates live in club_db.py (validate_match / validate_game_score).
CREATE TABLE IF NOT EXISTS players (
    id INTEGER PRIMARY KEY,
    canonical_name TEXT NOT NULL UNIQUE COLLATE NOCASE,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS matches (
    id INTEGER PRIMARY KEY,
    match_type TEXT NOT NULL CHECK (match_type IN ('singles', 'doubles')),
    played_at TEXT NOT NULL,
    side_a_p1 INTEGER NOT NULL REFERENCES players (id),
    side_a_p2 INTEGER REFERENCES players (id),
    side_b_p1 INTEGER NOT NULL REFERENCES players (id),
    side_b_p2 INTEGER REFERENCES players (id),
    winner_side TEXT NOT NULL CHECK (winner_side IN ('A', 'B')),
    source TEXT NOT NULL DEFAULT 'manual',
    entered_at TEXT NOT NULL DEFAULT (datetime('now')),
    CHECK (side_a_p1 <> side_b_p1),
    CHECK ((match_type = 'singles') = (side_a_p2 IS NULL AND side_b_p2 IS NULL))
);

CREATE TABLE IF NOT EXISTS match_games (
    match_id INTEGER NOT NULL REFERENCES matches (id) ON DELETE CASCADE,
    game_number INTEGER NOT NULL CHECK (game_number BETWEEN 1 AND 3),
    side_a_score INTEGER NOT NULL CHECK (side_a_score BETWEEN 0 AND 30),
    side_b_score INTEGER NOT NULL CHECK (side_b_score BETWEEN 0 AND 30),
    PRIMARY KEY (match_id, game_number)
);
