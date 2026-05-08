-- NBA Shot Quality Database Schema
-- Mirrors the plate-appearance ingestion pattern from Statcast pipeline

CREATE TABLE IF NOT EXISTS shots (
    shot_id         TEXT PRIMARY KEY,
    player_id       INTEGER NOT NULL,
    player_name     TEXT NOT NULL,
    team_id         INTEGER,
    team_name       TEXT,
    season          TEXT NOT NULL,          -- e.g. '2024-25'
    game_id         TEXT NOT NULL,
    game_date       TEXT,
    period          INTEGER,
    minutes_remaining INTEGER,
    seconds_remaining INTEGER,
    shot_type       TEXT,                   -- '2PT' or '3PT'
    shot_zone_basic TEXT,                   -- 'Mid-Range', 'Above the Break 3', etc.
    shot_zone_area  TEXT,                   -- 'Left Side', 'Center', etc.
    shot_zone_range TEXT,                   -- '8-16 ft', '16-24 ft', etc.
    shot_distance   REAL,                   -- feet
    loc_x           REAL,                   -- court x coordinate (tenths of inch)
    loc_y           REAL,                   -- court y coordinate
    shot_attempted  INTEGER DEFAULT 1,
    shot_made       INTEGER NOT NULL,       -- 0 or 1
    action_type     TEXT,                   -- 'Jump Shot', 'Layup', etc.
    league          TEXT DEFAULT 'NBA',     -- 'NBA', 'NCAA', 'G-League'
    data_source     TEXT DEFAULT 'nba_api', -- 'nba_api', 'cbbd'
    p_make          REAL,                   -- model predicted make probability
    created_at      TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_shots_player_season ON shots(player_id, season);
CREATE INDEX IF NOT EXISTS idx_shots_game ON shots(game_id);
CREATE INDEX IF NOT EXISTS idx_shots_season ON shots(season);

-- Aggregated player-season summary (populated by models/xpts_model.py)
CREATE TABLE IF NOT EXISTS player_season_summary (
    player_id       INTEGER NOT NULL,
    player_name     TEXT NOT NULL,
    season          TEXT NOT NULL,
    total_shots     INTEGER,
    shots_made      INTEGER,
    fg_pct          REAL,
    total_pts       REAL,
    expected_pts    REAL,
    pts_above_exp   REAL,                   -- key metric: skill vs luck
    avg_shot_dist   REAL,
    pct_3pt         REAL,
    updated_at      TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (player_id, season)
);
