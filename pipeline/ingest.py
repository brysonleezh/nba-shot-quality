"""
NBA Shot Ingestion Pipeline
----------------------------
Pulls shot-chart data from nba_api for a given season and set of players,
persists to SQLite — mirrors the Statcast ingestion pattern.

Usage:
    python pipeline/ingest.py --season 2024-25                      # default: min 200 FGA
    python pipeline/ingest.py --season 2024-25 --min-fga 100        # lower threshold
    python pipeline/ingest.py --season 2024-25 --player-ids 2544 1629029
"""

import argparse
import sqlite3
import time
import logging
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import shotchartdetail, leaguedashplayerstats
from nba_api.stats.static import players as nba_players

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "nba_shots.db"
SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


# ── DB helpers ────────────────────────────────────────────────────────────────

def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    schema = SCHEMA_PATH.read_text()
    conn.executescript(schema)
    conn.commit()
    log.info("DB initialised at %s", DB_PATH)


# ── Player lookup ─────────────────────────────────────────────────────────────

def get_players_by_min_fga(season: str, min_fga: int = 200) -> list[dict]:
    """Return all players who attempted at least min_fga shots in the season."""
    log.info("Fetching players with FGA >= %d for %s …", min_fga, season)
    resp = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
    )
    df = resp.get_data_frames()[0]
    df = df[df["FGA"] >= min_fga].sort_values("FGA", ascending=False)
    log.info("  → %d players qualify (FGA >= %d)", len(df), min_fga)
    return [{"player_id": int(r.PLAYER_ID), "player_name": r.PLAYER_NAME}
            for r in df.itertuples()]


# ── Shot ingestion ────────────────────────────────────────────────────────────

def fetch_shots(player_id: int, player_name: str, season: str) -> pd.DataFrame:
    """Pull all shot-chart rows for a single player-season."""
    resp = shotchartdetail.ShotChartDetail(
        player_id=player_id,
        team_id=0,
        season_nullable=season,
        season_type_all_star="Regular Season",
        context_measure_simple="FGA",
    )
    df = resp.get_data_frames()[0]
    if df.empty:
        return df

    df = df.rename(columns=str.lower)
    df["shot_id"]      = df["game_id"] + "_" + df["game_event_id"].astype(str)
    df["player_id"]    = player_id
    df["player_name"]  = player_name
    df["season"]       = season
    df["shot_made"]    = df["shot_made_flag"].astype(int)

    keep = [
        "shot_id", "player_id", "player_name", "team_id", "team_name",
        "season", "game_id", "game_date", "period",
        "minutes_remaining", "seconds_remaining",
        "shot_type", "shot_zone_basic", "shot_zone_area", "shot_zone_range",
        "shot_distance", "loc_x", "loc_y",
        "shot_attempted", "shot_made", "action_type",
    ]
    # guard against missing columns
    keep = [c for c in keep if c in df.columns]
    return df[keep]


def upsert_shots(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    """Insert shots, silently skip any duplicate shot_id rows."""
    if df.empty:
        return 0
    cols = ", ".join(df.columns)
    placeholders = ", ".join(["?"] * len(df.columns))
    sql = f"INSERT OR IGNORE INTO shots ({cols}) VALUES ({placeholders})"
    conn.executemany(sql, df.itertuples(index=False, name=None))
    written = conn.execute("SELECT changes()").fetchone()[0]
    conn.commit()
    return written


def already_ingested(conn: sqlite3.Connection, player_id: int, season: str) -> bool:
    cur = conn.execute(
        "SELECT 1 FROM shots WHERE player_id=? AND season=? LIMIT 1",
        (player_id, season)
    )
    return cur.fetchone() is not None


# ── Main ──────────────────────────────────────────────────────────────────────

def run(season: str, player_ids: list[int] | None = None, min_fga: int = 200,
        delay: float = 0.6) -> None:
    conn = get_connection()
    init_db(conn)

    if player_ids:
        all_players_info = nba_players.get_players()
        lookup = {p["id"]: p["full_name"] for p in all_players_info}
        targets = [{"player_id": pid, "player_name": lookup.get(pid, str(pid))}
                   for pid in player_ids]
    else:
        targets = get_players_by_min_fga(season, min_fga=min_fga)

    total_rows = 0
    for i, player in enumerate(targets, 1):
        pid, name = player["player_id"], player["player_name"]

        if already_ingested(conn, pid, season):
            log.info("[%d/%d] %s — already in DB, skipping", i, len(targets), name)
            continue

        log.info("[%d/%d] Ingesting %s (id=%d) …", i, len(targets), name, pid)
        try:
            df = fetch_shots(pid, name, season)
            n = upsert_shots(conn, df)
            total_rows += n
            log.info("  → %d shots written", n)
        except Exception as exc:
            log.warning("  → failed for %s: %s", name, exc)

        time.sleep(delay)   # respect nba_api rate limits

    log.info("Done. Total shots persisted this run: %d", total_rows)
    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NBA shot ingestion pipeline")
    parser.add_argument("--season",     default="2024-25", help="e.g. 2024-25")
    parser.add_argument("--min-fga",   type=int, default=200,
                        help="Minimum FGA threshold to include a player (default: 200)")
    parser.add_argument("--player-ids", type=int, nargs="+",
                        help="Specific player IDs to ingest (ignores --min-fga)")
    args = parser.parse_args()

    run(season=args.season, player_ids=args.player_ids, min_fga=args.min_fga)
