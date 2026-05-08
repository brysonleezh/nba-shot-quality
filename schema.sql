-- NBA Shot Quality Database Schema
-- Mirrors pattern from Statcast ingestion pipeline (plate appearance → season grain)
-- Here: shot → game → player → season grain

-- ─────────────────────────────────────────────
-- Players dimension
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS players (
    player_id       INTEGER PRIMARY KEY,
    player_name     TEXT NOT NULL,
    team_id         INTEGER,
    team_abbr       TEXT,
    season          TEXT NOT NULL  -- e.g. '2024-25'
);

-- ─────────────────────────────────────────────
-- Shot-level fact table (finest grain)
-- One row per field goal attempt
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS shots (
    shot_id             INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id             TEXT NOT NULL,
    game_date           TEXT NOT NULL,         -- YYYY-MM-DD
    season              TEXT NOT NULL,          -- e.g. '2024-25'
    player_id           INTEGER NOT NULL,
    player_name         TEXT NOT NULL,
    team_id             INTEGER,
    team_abbr           TEXT,

    -- Shot geometry (NBA coordinate system: center of basket = 0,0)
    loc_x               REAL,                  -- horizontal, inches from center
    loc_y               REAL,                  -- vertical, inches from basket

    -- Shot context
    shot_type           TEXT,                  -- '2PT Field Goal' | '3PT Field Goal'
    action_type         TEXT,                  -- 'Jump Shot', 'Driving Layup', etc.
    shot_zone_basic     TEXT,                  -- 'Restricted Area', 'Mid-Range', etc.
    shot_zone_area      TEXT,                  -- 'Left Side', 'Right Side Center', etc.
    shot_zone_range     TEXT,                  -- 'Less Than 8 ft', '8-16 ft', etc.
    shot_distance       REAL,                  -- feet from basket

    -- Outcome
    shot_made_flag      INTEGER NOT NULL,       -- 1 = made, 0 = missed
    shot_value          INTEGER NOT NULL,       -- 2 or 3

    -- Game clock
    period              INTEGER,
    minutes_remaining   INTEGER,
    seconds_remaining   INTEGER,

    -- Model outputs (populated after model.py runs)
    xfgpct              REAL,                  -- expected FG% from model
    xpts                REAL,                  -- expected points (xfgpct * shot_value)
    pts_above_exp       REAL,                  -- shot_made_flag*shot_value - xpts

    FOREIGN KEY (player_id) REFERENCES players(player_id)
);

CREATE INDEX IF NOT EXISTS idx_shots_player_season ON shots(player_id, season);
CREATE INDEX IF NOT EXISTS idx_shots_game ON shots(game_id);
CREATE INDEX IF NOT EXISTS idx_shots_zone ON shots(shot_zone_basic, season);

-- ─────────────────────────────────────────────
-- Player-season aggregated summary
-- Refreshed by model.py after each ingest run
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS player_season_summary (
    player_id           INTEGER NOT NULL,
    player_name         TEXT NOT NULL,
    team_abbr           TEXT,
    season              TEXT NOT NULL,

    -- Volume
    fga                 INTEGER,               -- field goal attempts
    fgm                 INTEGER,               -- field goal makes
    fg_pct              REAL,                  -- actual FG%
    fga_3pt             INTEGER,
    fgm_3pt             INTEGER,
    fg3_pct             REAL,
    pts                 INTEGER,

    -- Shot quality (model)
    xfgpct_avg          REAL,                  -- avg expected FG% across attempts (shot diet quality)
    xpts_total          REAL,                  -- total expected points
    pts_above_exp       REAL,                  -- total PAE (shot making above expectation)
    pae_per_100         REAL,                  -- PAE per 100 attempts (rate-normalised)

    -- Zone distribution
    pct_rim             REAL,                  -- % of FGA from restricted area
    pct_midrange        REAL,
    pct_corner3         REAL,
    pct_above_break3    REAL,

    PRIMARY KEY (player_id, season),
    FOREIGN KEY (player_id) REFERENCES players(player_id)
);

-- ─────────────────────────────────────────────
-- Zone-level summary (player × season × zone)
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS player_zone_summary (
    player_id           INTEGER NOT NULL,
    season              TEXT NOT NULL,
    shot_zone_basic     TEXT NOT NULL,

    fga                 INTEGER,
    fgm                 INTEGER,
    fg_pct              REAL,
    xfgpct_avg          REAL,
    pts_above_exp       REAL,

    PRIMARY KEY (player_id, season, shot_zone_basic),
    FOREIGN KEY (player_id) REFERENCES players(player_id)
);
