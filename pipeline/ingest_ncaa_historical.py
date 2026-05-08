"""
Historical NCAA shot ingestion for model training.

Pulls 3 prior seasons (2022-23, 2023-24, 2024-25) from major conferences
via cbbd, storing ALL players' shots as the training corpus for the xPTS model.

Usage:
    python pipeline/ingest_ncaa_historical.py
    python pipeline/ingest_ncaa_historical.py --seasons 2024 2025
    python pipeline/ingest_ncaa_historical.py --conference "Big Ten"
"""

from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import time
from pathlib import Path

import pandas as pd

try:
    from pipeline.ingest_prospects import (
        _cbbd_client, classify_zone, action_type_from_text,
        to_halfcourt, parse_clock, get_conn, init_db, write_shots,
    )
except ModuleNotFoundError:
    from ingest_prospects import (
        _cbbd_client, classify_zone, action_type_from_text,
        to_halfcourt, parse_clock, get_conn, init_db, write_shots,
    )

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DB_PATH     = Path(__file__).resolve().parent.parent / "data" / "nba_shots.db"
SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"

# cbbd integer season → DB season string
SEASON_MAP = {
    2023: "2022-23",
    2024: "2023-24",
    2025: "2024-25",
}

DEFAULT_SEASONS = [2023, 2024, 2025]

# Major conference teams — names must match cbbd's team naming
MAJOR_CONFERENCE_TEAMS: dict[str, list[str]] = {
    "ACC": [
        "Duke", "North Carolina", "Virginia", "Syracuse", "Georgia Tech",
        "Florida State", "Miami", "Clemson", "NC State", "Notre Dame",
        "Pittsburgh", "Louisville", "Boston College", "Virginia Tech",
        "Wake Forest", "Stanford", "California",
    ],
    "Big Ten": [
        "Michigan", "Michigan State", "Ohio State", "Indiana", "Purdue",
        "Illinois", "Iowa", "Minnesota", "Wisconsin", "Penn State",
        "Northwestern", "Nebraska", "Maryland", "Rutgers",
        "Oregon", "UCLA", "Washington",
    ],
    "Big 12": [
        "Kansas", "Baylor", "Texas", "Texas Tech", "Oklahoma",
        "Oklahoma State", "Iowa State", "Kansas State", "TCU",
        "West Virginia", "Houston", "BYU", "Cincinnati", "UCF",
        "Arizona", "Arizona State", "Utah", "Colorado",
    ],
    "SEC": [
        "Kentucky", "Tennessee", "Auburn", "Alabama", "Arkansas",
        "LSU", "Florida", "Georgia", "Mississippi State", "Ole Miss",
        "Vanderbilt", "South Carolina", "Missouri", "Texas A&M",
    ],
    "Big East": [
        "UConn", "Villanova", "Georgetown", "Creighton", "Marquette",
        "Xavier", "Providence", "St. John's", "Seton Hall", "Butler",
        "DePaul",
    ],
}


def player_id_from_name(name: str, team: str) -> int:
    return -(abs(hash(name + "|" + team)) % 10_000_000)


def already_ingested(conn: sqlite3.Connection, team: str, season_str: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM shots WHERE team_name=? AND season=? AND league='NCAA' LIMIT 1",
        (team, season_str),
    ).fetchone()
    return row is not None


def fetch_and_store_team(conn: sqlite3.Connection, team: str,
                         cbbd_season: int, season_str: str) -> int:
    client, PlaysApi = _cbbd_client()
    with client as api_client:
        from cbbd import PlaysApi as _PlaysApi
        plays_api = _PlaysApi(api_client)
        plays = plays_api.get_plays_by_team(
            season=cbbd_season,
            team=team,
            shooting_plays_only=True,
        )

    if not plays:
        return 0

    rows: list[dict] = []
    for play in plays:
        si = getattr(play, "shot_info", None)
        if si is None:
            continue

        shooter_obj  = getattr(si, "shooter", None)
        shooter_name = getattr(shooter_obj, "name", "") or ""
        if not shooter_name:
            continue

        range_str = getattr(si, "range", None)
        if range_str == "free_throw":
            continue

        loc   = getattr(si, "location", None)
        x_raw = loc.x if loc and loc.x is not None else None
        y_raw = loc.y if loc and loc.y is not None else None

        if x_raw is not None and y_raw is not None:
            hx, hy = to_halfcourt(float(x_raw), float(y_raw))
        else:
            hx, hy = None, None

        zone = classify_zone(range_str, hx, hy)
        if zone is None:
            continue

        made   = bool(getattr(si, "made", False))
        action = action_type_from_text(getattr(play, "play_text", None))

        secs_total = getattr(play, "seconds_remaining", None)
        if secs_total is not None:
            mins, secs = divmod(int(secs_total), 60)
        else:
            mins, secs = parse_clock(getattr(play, "clock", None))

        cbbd_pid = getattr(shooter_obj, "id", None)
        pid      = cbbd_pid if cbbd_pid is not None else player_id_from_name(shooter_name, team)
        play_id  = getattr(play, "id", None) or f"idx{len(rows)}"
        shot_id  = f"hist_{play.game_id}_{play_id}_{pid}"
        game_date = str(getattr(play, "game_start_date", "") or "")[:10]

        rows.append({
            "shot_id":           shot_id,
            "player_id":         int(pid),
            "player_name":       shooter_name,
            "team_id":           0,
            "team_name":         team,
            "season":            season_str,
            "game_id":           str(play.game_id),
            "game_date":         game_date,
            "period":            int(getattr(play, "period", 1) or 1),
            "minutes_remaining": mins,
            "seconds_remaining": secs,
            "shot_type":         zone["shot_type"],
            "shot_zone_basic":   zone["shot_zone_basic"],
            "shot_zone_area":    zone["shot_zone_area"],
            "shot_zone_range":   zone["shot_zone_range"],
            "shot_distance":     zone["shot_distance"],
            "loc_x":             round(hx, 1) if hx is not None else None,
            "loc_y":             round(hy, 1) if hy is not None else None,
            "shot_attempted":    1,
            "shot_made":         int(made),
            "action_type":       action,
            "league":            "NCAA",
            "data_source":       "cbbd_historical",
        })

    df = pd.DataFrame(rows)
    return write_shots(conn, df)


def run(seasons: list[int] | None = None,
        conferences: list[str] | None = None) -> None:
    seasons     = seasons or DEFAULT_SEASONS
    conferences = conferences or list(MAJOR_CONFERENCE_TEAMS.keys())

    teams: list[str] = []
    for conf in conferences:
        teams.extend(MAJOR_CONFERENCE_TEAMS.get(conf, []))
    teams = list(dict.fromkeys(teams))  # dedupe while preserving order

    conn = get_conn()
    init_db(conn)

    total = 0
    combos = [(team, s) for s in seasons for team in teams]
    log.info("═══ Historical NCAA ingest: %d teams × %d seasons = %d API calls ═══",
             len(teams), len(seasons), len(combos))

    for idx, (team, cbbd_season) in enumerate(combos, 1):
        season_str = SEASON_MAP[cbbd_season]

        if already_ingested(conn, team, season_str):
            log.info("[%d/%d] %s %s — already in DB, skipping",
                     idx, len(combos), team, season_str)
            continue

        log.info("[%d/%d] %s  %s …", idx, len(combos), team, season_str)
        try:
            n = fetch_and_store_team(conn, team, cbbd_season, season_str)
            total += n
            log.info("  → %d shots written", n)
        except Exception as exc:
            log.warning("  → FAILED: %s", exc)

        time.sleep(0.5)

    log.info("═══ Done. Total shots persisted: %d ═══", total)
    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest historical NCAA shots for model training")
    parser.add_argument(
        "--seasons", nargs="+", type=int, default=DEFAULT_SEASONS,
        help="cbbd season integers to pull (e.g. 2023 2024 2025)")
    parser.add_argument(
        "--conference", nargs="+", default=None,
        metavar="CONF",
        help='Limit to specific conferences (e.g. "ACC" "Big Ten")')
    args = parser.parse_args()
    run(seasons=args.seasons, conferences=args.conference)
