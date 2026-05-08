"""
Ingest 2026 NBA draft prospect shot data.

NCAA shots  → cbbd (CollegeBasketballData.com Python client)
G-League    → nba_api with league_id='20' (same as existing pipeline)

Setup:
    pip install cbbd
    export CBB_API_KEY="your_key"   # free key at app.collegebasketballdata.com

Usage:
    python pipeline/ingest_prospects.py                          # all NCAA prospects
    python pipeline/ingest_prospects.py --source gleague
    python pipeline/ingest_prospects.py --source all
    python pipeline/ingest_prospects.py --names "AJ Dybantsa" "Cameron Boozer"

Shot coordinates (loc_x, loc_y) are stored in cbbd raw units (divide by 10 for feet).
The xPTS model uses zone labels and shot_distance as its primary features, so
coordinate precision matters less than zone accuracy. Zones are derived from
shot_info.range (provided by cbbd) plus x-position for left/right classification.
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
    from pipeline.prospects_2026 import GLEAGUE_PROSPECTS, NCAA_PROSPECTS
except ModuleNotFoundError:
    from prospects_2026 import GLEAGUE_PROSPECTS, NCAA_PROSPECTS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DB_PATH     = Path(__file__).resolve().parent.parent / "data" / "nba_shots.db"
SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"

NCAA_SEASON = 2026      # cbbd integer format (2025-26 season)
DB_SEASON   = "2025-26" # season string stored in shots table


# ── Synthetic player IDs for college prospects ─────────────────────────────────
# NBA Stats IDs are large positive integers; we use small negatives to avoid clash.

def prospect_id(name: str) -> int:
    return -(abs(hash(name)) % 10_000_000)


# ── Coordinate conversion ─────────────────────────────────────────────────────
# cbbd uses FULL-COURT coordinates (x: 0→940, y: 0→500, units: tenths-of-foot)
# Left basket at x≈52.5, right basket at x≈888.5, centre y=250
# NBA uses HALF-COURT coordinates with basket at origin (same tenths-of-foot units)
# We convert cbbd → NBA so loc_x/loc_y/shot_distance are model-compatible.

_BASKET_X_L = 52.5
_BASKET_X_R = 888.5
_BASKET_Y   = 250.0


def to_halfcourt(x_raw: float, y_raw: float) -> tuple[float, float]:
    """Convert cbbd full-court coords to half-court coords (basket = origin)."""
    if x_raw <= 470:
        return x_raw - _BASKET_X_L, y_raw - _BASKET_Y
    else:
        return _BASKET_X_R - x_raw, y_raw - _BASKET_Y


# ── Action type from play_text ────────────────────────────────────────────────

def action_type_from_text(play_text: str | None) -> str:
    """Extract NBA-style action_type from cbbd play_text."""
    t = (play_text or "").lower()
    if "dunk"       in t: return "Dunk Shot"
    if "layup"      in t: return "Layup Shot"
    if "hook"       in t: return "Hook Shot"
    if "tip"        in t: return "Tip Shot"
    if "jump"       in t: return "Jump Shot"
    if "three point" in t: return "Jump Shot"
    if "jumper"     in t: return "Jump Shot"
    return "Jump Shot"


# ── Zone classification ────────────────────────────────────────────────────────
# cbbd range enum: 'rim' | 'jumper' | 'three_pointer' | 'free_throw'
# hx, hy are half-court coordinates (basket = origin, tenths-of-foot)

def classify_zone(range_str: str | None, hx: float | None, hy: float | None) -> dict | None:
    """
    Map cbbd range + half-court coordinates to NBA-style zone fields.
    Returns None for free_throw (caller skips those rows).
    """
    key = (range_str or "").lower().strip()

    if key == "free_throw":
        return None

    has_coords = hx is not None and hy is not None

    # ── Distance ──
    if has_coords:
        distance = (hx ** 2 + hy ** 2) ** 0.5 / 10.0
    else:
        distance = {"rim": 4.0, "jumper": 17.0, "three_pointer": 22.0}.get(key, 17.0)

    # ── Shot type + zone_basic + zone_range ──
    if key == "rim":
        shot_type  = "2PT Field Goal"
        zone_basic = "Restricted Area"
        zone_range = "Less Than 8 ft."

    elif key == "jumper":
        shot_type = "2PT Field Goal"
        if distance < 8:
            zone_basic = "In The Paint (Non-RA)"
            zone_range = "Less Than 8 ft."
        elif distance < 16:
            zone_basic = "In The Paint (Non-RA)"
            zone_range = "8-16 ft."
        elif distance < 24:
            zone_basic = "Mid-Range"
            zone_range = "16-24 ft."
        else:
            zone_basic = "Mid-Range"
            zone_range = "24+ ft."

    elif key == "three_pointer":
        shot_type  = "3PT Field Goal"
        zone_range = "24+ ft."
        if has_coords and hx < 90 and hy < -220:
            zone_basic = "Left Corner 3"
        elif has_coords and hx < 90 and hy > 220:
            zone_basic = "Right Corner 3"
        else:
            zone_basic = "Above the Break 3"

    else:
        # Unknown range — default to mid-range jumper
        shot_type  = "2PT Field Goal"
        zone_basic = "Mid-Range"
        zone_range = "16-24 ft."

    # ── Zone area (left / center / right) from hy ──
    if zone_basic == "Left Corner 3":
        zone_area = "Left Side(L)"
    elif zone_basic == "Right Corner 3":
        zone_area = "Right Side(R)"
    elif has_coords:
        if hy < -80:
            zone_area = "Left Side(L)"
        elif hy > 80:
            zone_area = "Right Side(R)"
        else:
            zone_area = "Center(C)"
    else:
        zone_area = "Center(C)"

    return {
        "shot_type":       shot_type,
        "shot_zone_basic": zone_basic,
        "shot_zone_area":  zone_area,
        "shot_zone_range": zone_range,
        "shot_distance":   round(distance, 1),
    }


# ── Clock parsing ─────────────────────────────────────────────────────────────

def parse_clock(clock: object) -> tuple[int, int]:
    """Parse 'MM:SS' clock string → (minutes_remaining, seconds_remaining)."""
    try:
        m, s = str(clock).split(":")
        return int(m), int(s)
    except Exception:
        return 0, 0


# ── DB helpers ────────────────────────────────────────────────────────────────

def get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)


def init_db(conn: sqlite3.Connection) -> None:
    """Create tables from schema and migrate existing DBs with ALTER TABLE."""
    conn.executescript(SCHEMA_PATH.read_text())
    conn.commit()

    # Add new columns to pre-existing databases that lack them
    for col, default in [("league", "'NBA'"), ("data_source", "'nba_api'")]:
        try:
            conn.execute(f"ALTER TABLE shots ADD COLUMN {col} TEXT DEFAULT {default}")
            conn.commit()
            log.info("DB migrated: added column '%s' to shots", col)
        except sqlite3.OperationalError:
            pass  # column already exists — expected on re-runs


def already_ingested(conn: sqlite3.Connection, player_id: int, season: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM shots WHERE player_id=? AND season=? AND league='NCAA' LIMIT 1",
        (player_id, season),
    ).fetchone()
    return row is not None


def write_shots(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    cols = ", ".join(df.columns)
    placeholders = ", ".join(["?"] * len(df.columns))
    sql = f"INSERT OR IGNORE INTO shots ({cols}) VALUES ({placeholders})"
    before = conn.execute("SELECT total_changes()").fetchone()[0]
    conn.executemany(sql, df.itertuples(index=False, name=None))
    written = conn.execute("SELECT total_changes()").fetchone()[0] - before
    conn.commit()
    return written


# ── NCAA ingestion via cbbd ───────────────────────────────────────────────────

def _cbbd_client():
    """Build and return a cbbd PlaysApi client. Raises on missing deps/key."""
    try:
        import cbbd
        from cbbd import ApiClient, Configuration, PlaysApi  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "cbbd not installed. Run:\n"
            "  pip install cbbd\n"
            "Then set your free API key:\n"
            "  export CBB_API_KEY='your_key'  # app.collegebasketballdata.com"
        ) from exc

    api_key = os.getenv("CBB_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "CBB_API_KEY environment variable not set.\n"
            "Get a free key at app.collegebasketballdata.com"
        )

    from cbbd import ApiClient, Configuration, PlaysApi

    config = Configuration()
    config.access_token = api_key  # cbbd uses Bearer auth via access_token, not api_key dict
    config.verify_ssl = False  # Mac SSL cert workaround — run Install Certificates.command to fix permanently
    return ApiClient(config), PlaysApi


def fetch_team_plays(team: str, season: int = NCAA_SEASON) -> list:
    """
    Fetch all shooting plays for a team in one API call.
    Shared across all prospects from the same team — saves API quota.
    """
    client, PlaysApi = _cbbd_client()
    with client as api_client:
        from cbbd import PlaysApi as _PlaysApi
        plays_api = _PlaysApi(api_client)
        plays = plays_api.get_plays_by_team(
            season=season,
            team=team,
            shooting_plays_only=True,
        )
    return plays or []


def extract_player_shots(plays: list, prospect: dict) -> pd.DataFrame:
    """
    Filter a team's play list down to one prospect's shots.
    Called once per prospect after fetch_team_plays().
    """
    pid    = prospect_id(prospect["name"])
    target = prospect["name"].lower()
    rows: list[dict] = []

    for play in plays:
        si = getattr(play, "shot_info", None)
        if si is None:
            continue

        shooter_obj  = getattr(si, "shooter", None)
        shooter_name = getattr(shooter_obj, "name", "") or ""
        # Match by name substring (cbbd has no stable player ID per prospect dict)
        if target not in shooter_name.lower():
            continue
        # Use cbbd's native player ID if available (more reliable for dedup)
        cbbd_pid = getattr(shooter_obj, "id", None)

        # ① skip free throws
        range_str = getattr(si, "range", None)
        if range_str == "free_throw":
            continue

        # ② extract raw coordinates (None if missing)
        loc   = getattr(si, "location", None)
        x_raw = loc.x if loc and loc.x is not None else None
        y_raw = loc.y if loc and loc.y is not None else None

        # ③ convert full-court → half-court (basket = origin)
        if x_raw is not None and y_raw is not None:
            hx, hy = to_halfcourt(float(x_raw), float(y_raw))
        else:
            hx, hy = None, None

        # ④⑤⑥ classify zone + distance
        zone = classify_zone(range_str, hx, hy)
        if zone is None:
            continue  # safety: skip any unhandled free throws

        made = bool(getattr(si, "made", False))

        # ⑦ action_type from play_text
        action = action_type_from_text(getattr(play, "play_text", None))

        # ⑧ clock
        secs_total = getattr(play, "seconds_remaining", None)
        if secs_total is not None:
            mins, secs = divmod(int(secs_total), 60)
        else:
            mins, secs = parse_clock(getattr(play, "clock", None))

        play_id = getattr(play, "id", None) or f"idx{len(rows)}"
        id_part = cbbd_pid if cbbd_pid is not None else pid
        shot_id = f"ncaa_{play.game_id}_{play_id}_{id_part}"

        # game_date: take first 10 chars → "2025-11-05"
        game_date = str(getattr(play, "game_start_date", "") or "")[:10]

        rows.append({
            "shot_id":           shot_id,
            "player_id":         pid,
            "player_name":       prospect["name"],
            "team_id":           0,
            "team_name":         prospect["team"],
            "season":            DB_SEASON,
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
            "data_source":       "cbbd",
        })

    return pd.DataFrame(rows)


# ── G-League ingestion via nba_api ────────────────────────────────────────────

def fetch_gleague_shots(prospect: dict, delay: float = 0.6) -> pd.DataFrame:
    """Pull G-League shots for a prospect via nba_api (league_id='20')."""
    from nba_api.stats.endpoints import shotchartdetail

    resp = shotchartdetail.ShotChartDetail(
        player_id=prospect["nba_player_id"],
        team_id=0,
        season_nullable=DB_SEASON,
        season_type_all_star="Regular Season",
        context_measure_simple="FGA",
        league_id="20",
    )
    time.sleep(delay)

    df = resp.get_data_frames()[0]
    if df.empty:
        return df

    df = df.rename(columns=str.lower)
    df["shot_id"]     = "gl_" + df["game_id"] + "_" + df["game_event_id"].astype(str)
    df["player_id"]   = prospect["nba_player_id"]
    df["player_name"] = prospect["name"]
    df["season"]      = DB_SEASON
    df["shot_made"]   = df["shot_made_flag"].astype(int)
    df["league"]      = "G-League"
    df["data_source"] = "nba_api"

    keep = [
        "shot_id", "player_id", "player_name", "team_id", "team_name",
        "season", "game_id", "game_date", "period",
        "minutes_remaining", "seconds_remaining",
        "shot_type", "shot_zone_basic", "shot_zone_area", "shot_zone_range",
        "shot_distance", "loc_x", "loc_y",
        "shot_attempted", "shot_made", "action_type",
        "league", "data_source",
    ]
    return df[[c for c in keep if c in df.columns]]


# ── Main ──────────────────────────────────────────────────────────────────────

def run(source: str = "ncaa", names: list[str] | None = None) -> None:
    conn = get_conn()
    init_db(conn)
    total = 0

    # ── NCAA ──
    if source in ("ncaa", "all"):
        targets = NCAA_PROSPECTS
        if names:
            targets = [p for p in targets if p["name"] in names]

        # Group by team — one API call per team regardless of how many prospects
        from collections import defaultdict
        by_team: dict[str, list[dict]] = defaultdict(list)
        for p in targets:
            by_team[p["team"]].append(p)

        unique_teams = list(by_team.keys())
        log.info("═══ NCAA: %d prospects across %d teams (%d API calls) ═══",
                 len(targets), len(unique_teams), len(unique_teams))

        for t_idx, team in enumerate(unique_teams, 1):
            prospects_in_team = by_team[team]

            # Skip entire team if all prospects already in DB
            pending = [p for p in prospects_in_team
                       if not already_ingested(conn, prospect_id(p["name"]), DB_SEASON)]
            if not pending:
                log.info("[%d/%d] %s — all prospects already in DB, skipping",
                         t_idx, len(unique_teams), team)
                continue

            log.info("[%d/%d] Fetching %s (%d prospects) …",
                     t_idx, len(unique_teams), team, len(pending))
            try:
                plays = fetch_team_plays(team)
                if not plays:
                    log.warning("  → no plays returned for %s", team)
                    continue

                for prospect in pending:
                    df = extract_player_shots(plays, prospect)
                    n  = write_shots(conn, df)
                    total += n
                    log.info("  → %-28s  %d shots written", prospect["name"], n)

            except Exception as exc:
                log.warning("  → FAILED for %s: %s", team, exc)

            time.sleep(0.5)

    # ── G-League ──
    if source in ("gleague", "all") and GLEAGUE_PROSPECTS:
        targets = GLEAGUE_PROSPECTS
        if names:
            targets = [p for p in targets if p["name"] in names]

        log.info("═══ G-League prospects: %d targets ═══", len(targets))
        for i, prospect in enumerate(targets, 1):
            log.info("[%d/%d] %s …", i, len(targets), prospect["name"])
            try:
                df = fetch_gleague_shots(prospect)
                n  = write_shots(conn, df)
                total += n
                log.info("  → %d shots written", n)
            except Exception as exc:
                log.warning("  → FAILED: %s", exc)

    log.info("═══ Done. Total shots persisted: %d ═══", total)
    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest 2026 draft prospect shot data")
    parser.add_argument(
        "--source", choices=["ncaa", "gleague", "all"], default="ncaa",
        help="Data source to pull from (default: ncaa)",
    )
    parser.add_argument(
        "--names", nargs="+", default=None,
        metavar="NAME",
        help='Limit to specific prospects, e.g. --names "AJ Dybantsa" "Cameron Boozer"',
    )
    args = parser.parse_args()
    run(source=args.source, names=args.names)
