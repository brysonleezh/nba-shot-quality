"""
Ingest 2025-26 NCAA prospect box-score stats via cbbd StatsApi.

Pulls get_player_season_stats(season=2026) for every team that has a
2026 draft prospect, then filters to prospect names and writes to the
player_season_box table in the DB.

Usage:
    python pipeline/ingest_box_scores.py
"""

from __future__ import annotations

import logging
import os
import sqlite3
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ingest_prospects import _cbbd_client, get_conn, init_db
from prospects_2026 import NCAA_PROSPECTS

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "nba_shots.db"

PROSPECT_TEAMS = sorted({p["team"] for p in NCAA_PROSPECTS})
PROSPECT_NAMES = {p["name"] for p in NCAA_PROSPECTS}

import re as _re
def _strip_suffix(name: str) -> str:
    """Remove generational suffixes so 'Labaron Philon Jr.' matches 'Labaron Philon'."""
    return _re.sub(r"\s+(Jr\.?|Sr\.?|II+|III+|IV+|V+)$", "", name.strip(), flags=_re.IGNORECASE)

# Canonical name lookup: stripped API name → prospect list name
_STRIPPED_TO_PROSPECT = {_strip_suffix(n): n for n in PROSPECT_NAMES}

CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS player_season_box (
    player_id       INTEGER,
    player_name     TEXT,
    team            TEXT,
    season          TEXT,
    league          TEXT DEFAULT 'NCAA',
    games           INTEGER,
    minutes         INTEGER,
    points          INTEGER,
    fg_made         INTEGER,
    fg_attempted    INTEGER,
    fg_pct          REAL,
    two_made        INTEGER,
    two_attempted   INTEGER,
    two_pct         REAL,
    three_made      INTEGER,
    three_attempted INTEGER,
    three_pct       REAL,
    ft_made         INTEGER,
    ft_attempted    INTEGER,
    ft_pct          REAL,
    reb_total       INTEGER,
    reb_off         INTEGER,
    reb_def         INTEGER,
    assists         INTEGER,
    blocks          INTEGER,
    steals          INTEGER,
    turnovers       INTEGER,
    fouls           INTEGER,
    efg_pct         REAL,
    ts_pct          REAL,
    usg_pct         REAL,
    ft_rate         REAL,
    ortg            REAL,
    drtg            REAL,
    net_rtg         REAL,
    ast_to_ratio    REAL,
    PRIMARY KEY (player_name, season, league)
)
"""


def init_box_table(conn: sqlite3.Connection) -> None:
    conn.execute(CREATE_TABLE)
    conn.commit()


def fetch_team_stats(team: str, season: int = 2026) -> list:
    import cbbd
    client, _ = _cbbd_client()
    with client as api_client:
        stats_api = cbbd.StatsApi(api_client)
        return stats_api.get_player_season_stats(season=season, team=team) or []


def stat_attr(obj, attr, default=None):
    val = getattr(obj, attr, None)
    if val is None:
        return default
    # Some fields are objects with sub-attributes (e.g. field_goals.pct)
    if hasattr(val, "__dict__"):
        return val
    return val


def parse_stat_row(s, season_str: str) -> dict | None:
    name = getattr(s, "name", None) or getattr(s, "player", None)
    if not name:
        return None

    fg   = getattr(s, "field_goals",           None)
    two  = getattr(s, "two_point_field_goals",  None)
    three= getattr(s, "three_point_field_goals",None)
    ft   = getattr(s, "free_throws",            None)
    reb  = getattr(s, "rebounds",               None)
    ws   = getattr(s, "win_shares",             None)

    def g(obj, a, default=None):
        return getattr(obj, a, default) if obj is not None else default

    fg_made  = g(fg,    "made");    fg_att  = g(fg,    "attempted")
    fg_pct   = g(fg,    "pct")
    two_made = g(two,   "made");    two_att = g(two,   "attempted")
    two_pct  = g(two,   "pct")
    t3_made  = g(three, "made");    t3_att  = g(three, "attempted")
    t3_pct   = g(three, "pct")
    ft_made  = g(ft,    "made");    ft_att  = g(ft,    "attempted")
    ft_pct   = g(ft,    "pct")
    reb_tot  = g(reb,   "total");   reb_off = g(reb, "offensive")
    reb_def  = g(reb,   "defensive")

    ortg = getattr(s, "offensive_rating", None)
    drtg = getattr(s, "defensive_rating", None)
    net  = getattr(s, "net_rating",       None)
    if ortg and drtg and net is None:
        net = round(ortg - drtg, 1)

    return {
        "player_id":       getattr(s, "athlete_id", None),
        "player_name":     name,
        "team":            getattr(s, "team", ""),
        "season":          season_str,
        "league":          "NCAA",
        "games":           getattr(s, "games",     None),
        "minutes":         getattr(s, "minutes",   None),
        "points":          getattr(s, "points",    None),
        "fg_made":         fg_made,  "fg_attempted":    fg_att,
        "fg_pct":          fg_pct,
        "two_made":        two_made, "two_attempted":   two_att,
        "two_pct":         two_pct,
        "three_made":      t3_made,  "three_attempted": t3_att,
        "three_pct":       t3_pct,
        "ft_made":         ft_made,  "ft_attempted":    ft_att,
        "ft_pct":          ft_pct,
        "reb_total":       reb_tot,  "reb_off": reb_off, "reb_def": reb_def,
        "assists":         getattr(s, "assists",   None),
        "blocks":          getattr(s, "blocks",    None),
        "steals":          getattr(s, "steals",    None),
        "turnovers":       getattr(s, "turnovers", None),
        "fouls":           getattr(s, "fouls",     None),
        "efg_pct":         getattr(s, "effective_field_goal_pct", None),
        "ts_pct":          getattr(s, "true_shooting_pct",        None),
        "usg_pct":         getattr(s, "usage",                    None),
        "ft_rate":         getattr(s, "free_throw_rate",          None),
        "ortg":            ortg,
        "drtg":            drtg,
        "net_rtg":         net,
        "ast_to_ratio":    getattr(s, "assists_turnover_ratio",   None),
    }


def run(season: int = 2026, season_str: str = "2025-26") -> None:
    conn = get_conn()
    init_db(conn)
    init_box_table(conn)

    client, _ = _cbbd_client()
    rows: list[dict] = []

    for i, team in enumerate(PROSPECT_TEAMS, 1):
        log.info("[%d/%d] %s …", i, len(PROSPECT_TEAMS), team)
        try:
            stats = fetch_team_stats(team, season)
            for s in stats:
                api_name = getattr(s, "name", None) or getattr(s, "player", None)
                if not api_name:
                    continue
                # Match exact name or suffix-stripped variant
                canonical = (api_name if api_name in PROSPECT_NAMES
                             else _STRIPPED_TO_PROSPECT.get(_strip_suffix(api_name)))
                if not canonical:
                    continue
                row = parse_stat_row(s, season_str)
                if row:
                    row["player_name"] = canonical   # store under prospect-list name
                    rows.append(row)
                    log.info("  + %s (api: %s)", canonical, api_name)
        except Exception as exc:
            log.warning("  FAILED for %s: %s", team, exc)
        time.sleep(0.4)

    if not rows:
        log.error("No rows fetched — check API key and season.")
        conn.close()
        return

    df = pd.DataFrame(rows)
    df.to_sql("player_season_box", conn,
              if_exists="replace", index=False, method="multi")
    conn.commit()
    log.info("player_season_box: %d rows written", len(df))
    conn.close()


if __name__ == "__main__":
    run()
