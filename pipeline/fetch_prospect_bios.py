"""
Fetch 2026 draft prospect biographical data from ESPN.

Steps:
  1. Search ESPN by player name + school → get ESPN athlete ID
  2. Fetch ESPN athlete detail endpoint → height, weight, DOB, hometown, headshot

Output:
  data/prospect_bios.json  —  dict keyed by player name

Usage:
    python pipeline/fetch_prospect_bios.py
    python pipeline/fetch_prospect_bios.py --force   # re-fetch even if already saved
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prospects_2026 import NCAA_PROSPECTS, INTERNATIONAL_PROSPECTS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "prospect_bios.json"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}

ESPN_SEARCH_URL  = "https://site.api.espn.com/apis/common/v3/search"
ESPN_ATHLETE_URL = ("https://sports.core.api.espn.com/v2/sports/basketball"
                    "/leagues/mens-college-basketball/athletes/{espn_id}")


# ── ESPN helpers ──────────────────────────────────────────────────────────────

def search_espn(name: str, team: str) -> dict | None:
    """Return first ESPN search result that plausibly matches the player."""
    try:
        r = requests.get(
            ESPN_SEARCH_URL,
            params={"query": name, "limit": 8, "type": "player", "sport": "basketball"},
            headers=HEADERS,
            timeout=10,
        )
        r.raise_for_status()
        items = r.json().get("items", [])
    except Exception as exc:
        log.warning("ESPN search error for %s: %s", name, exc)
        return None

    name_parts = name.lower().split()
    team_lower = team.lower()

    for item in items:
        espn_name = item.get("displayName", "").lower()
        espn_team = (item.get("team", {}) or {}).get("displayName", "").lower()

        # Require first AND last name tokens to match
        first_ok = name_parts[0] in espn_name
        last_ok  = name_parts[-1] in espn_name
        team_ok  = any(t in espn_team for t in team_lower.split())

        if first_ok and last_ok and (team_ok or len(items) == 1):
            return item

    # Fallback: first + last name match without team check
    for item in items:
        espn_name = item.get("displayName", "").lower()
        if name_parts[0] in espn_name and name_parts[-1] in espn_name:
            return item

    return None


def fetch_athlete_detail(espn_id: str) -> dict:
    """Fetch full athlete bio from ESPN athlete endpoint."""
    try:
        r = requests.get(
            ESPN_ATHLETE_URL.format(espn_id=espn_id),
            headers=HEADERS,
            timeout=10,
        )
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        log.warning("ESPN athlete detail error for id=%s: %s", espn_id, exc)
        return {}


def inches_to_ft_in(inches: int | None) -> str:
    if not inches:
        return ""
    return f"{inches // 12}'{inches % 12}\""


def parse_bio(name: str, team: str, rank: int, position: str,
              league: str = "NCAA") -> dict:
    """Build a bio record for one prospect."""
    bio = {
        "name":         name,
        "team":         team,
        "position":     position,
        "rank":         rank,
        "league":       league,
        "espn_id":      None,
        "headshot_url": None,
        "height":       "",
        "weight":       "",
        "dob":          "",
        "birthplace":   "",
        "jersey":       "",
    }

    if league != "NCAA":
        return bio  # no ESPN college data for international players

    result = search_espn(name, team)
    if not result:
        log.warning("No ESPN match found: %s (%s)", name, team)
        return bio

    espn_id = str(result.get("id", ""))
    bio["espn_id"] = espn_id
    bio["headshot_url"] = (
        f"https://a.espncdn.com/i/headshots/ncb/players/{espn_id}.png"
    )

    detail = fetch_athlete_detail(espn_id)
    if detail:
        height_in = detail.get("height")
        bio["height"]     = inches_to_ft_in(int(height_in)) if height_in else ""
        weight = detail.get("weight")
        bio["weight"]     = f"{int(weight)} lbs" if weight else ""
        dob = detail.get("dateOfBirth") or ""
        bio["dob"]        = dob[:10] if dob else ""
        bp = detail.get("birthPlace") or {}
        parts = [bp.get("city",""), bp.get("state",""), bp.get("country","")]
        bio["birthplace"] = ", ".join(p for p in parts if p)
        bio["jersey"]     = str(detail.get("jersey", ""))
        pos = detail.get("position") or {}
        if isinstance(pos, dict) and pos.get("abbreviation"):
            bio["position"] = pos["abbreviation"]
        # Headshot from search result (better quality URL)
        bio["headshot_url"] = (
            f"https://a.espncdn.com/i/headshots/mens-college-basketball/players/full/{espn_id}.png"
        )

    return bio


# ── Main ──────────────────────────────────────────────────────────────────────

def run(force: bool = False) -> None:
    # Load existing results so we can resume
    existing: dict = {}
    if OUTPUT_PATH.exists() and not force:
        existing = json.loads(OUTPUT_PATH.read_text())
        log.info("Loaded %d existing bios from %s", len(existing), OUTPUT_PATH)

    all_prospects = NCAA_PROSPECTS + INTERNATIONAL_PROSPECTS
    results = dict(existing)

    for i, p in enumerate(all_prospects, 1):
        name     = p["name"]
        team     = p["team"]
        rank     = p["rank"]
        position = p["position"]
        league   = "International" if p in INTERNATIONAL_PROSPECTS else "NCAA"

        if name in results and not force:
            log.info("[%d/%d] %s — already fetched, skipping", i, len(all_prospects), name)
            continue

        log.info("[%d/%d] Fetching %s (%s) …", i, len(all_prospects), name, team)
        bio = parse_bio(name, team, rank, position, league)
        results[name] = bio

        log.info("  → espn_id=%s  height=%s  weight=%s  dob=%s",
                 bio["espn_id"], bio["height"], bio["weight"], bio["dob"])

        # Save after every player so we can resume on failure
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False))

        time.sleep(0.8)  # be polite to ESPN API

    log.info("Done. %d / %d prospects have ESPN data.",
             sum(1 for v in results.values() if v.get("espn_id")),
             len(results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="Re-fetch all players even if already saved")
    args = parser.parse_args()
    run(force=args.force)
