"""
2026 NBA Draft prospect list.

Sources (merged):
- FanSided Top 60 Big Board (April 2026)
- No Ceilings NBA Big Board V.6 (April 2026)

Note: List is fluid until the May 27 NCAA early entry withdrawal deadline.
International prospects are listed separately — no public shot-level API available.
"""

from __future__ import annotations

# ── NCAA prospects ─────────────────────────────────────────────────────────────
# team must match CollegeBasketballData.com team name (used by cbbd API)
# Multiple prospects from the same team share one API call in ingest_prospects.py

NCAA_PROSPECTS: list[dict] = [
    # ── Consensus top 10 ──
    {"name": "Cameron Boozer",        "team": "Duke",            "position": "F",   "rank": 1},
    {"name": "Darryn Peterson",       "team": "Kansas",          "position": "G",   "rank": 2},
    {"name": "AJ Dybantsa",           "team": "BYU",             "position": "F",   "rank": 3},
    {"name": "Caleb Wilson",          "team": "North Carolina",  "position": "F",   "rank": 4},
    {"name": "Mikel Brown Jr.",       "team": "Louisville",      "position": "G",   "rank": 5},
    {"name": "Kingston Flemings",     "team": "Houston",         "position": "G",   "rank": 6},
    {"name": "Hannes Steinbach",      "team": "Washington",      "position": "F",   "rank": 7},
    {"name": "Koa Peat",              "team": "Arizona",         "position": "F",   "rank": 8},
    {"name": "Labaron Philon",        "team": "Alabama",         "position": "G",   "rank": 9},
    {"name": "Bennett Stirtz",        "team": "Iowa",            "position": "G",   "rank": 10},
    # ── 11-20 ──
    {"name": "Nate Ament",            "team": "Tennessee",       "position": "F",   "rank": 11},
    {"name": "Jayden Quaintance",     "team": "Kentucky",        "position": "C",   "rank": 12},
    {"name": "Aday Mara",             "team": "Michigan",        "position": "C",   "rank": 13},
    {"name": "Cameron Carr",          "team": "Baylor",          "position": "G",   "rank": 14},
    {"name": "Darius Acuff Jr.",      "team": "Arkansas",        "position": "G",   "rank": 15},
    {"name": "Yaxel Lendeborg",       "team": "Michigan",        "position": "F",   "rank": 16},
    {"name": "Thomas Haugh",          "team": "Florida",         "position": "F",   "rank": 17},
    {"name": "Keaton Wagler",         "team": "Illinois",        "position": "G",   "rank": 18},
    {"name": "Nikolas Khamenia",      "team": "Duke",            "position": "F",   "rank": 19},
    {"name": "Cayden Boozer",         "team": "Duke",            "position": "G",   "rank": 20},
    # ── 21-30 ──
    {"name": "Elyjah Freeman",        "team": "Auburn",          "position": "G",   "rank": 21},
    {"name": "Patrick Ngongba II",    "team": "Duke",            "position": "C",   "rank": 22},
    {"name": "Dailyn Swain",          "team": "Texas",           "position": "F",   "rank": 23},
    {"name": "Chris Cenac Jr.",       "team": "Houston",         "position": "C",   "rank": 24},
    {"name": "Tounde Yessoufou",      "team": "Baylor",          "position": "F",   "rank": 25},
    {"name": "Paul McNeil",           "team": "NC State",        "position": "G",   "rank": 26},
    {"name": "Meleek Thomas",         "team": "Arkansas",        "position": "G",   "rank": 27},
    {"name": "Braylon Mullins",       "team": "UConn",           "position": "G",   "rank": 28},
    {"name": "Neoklis Avdalas",       "team": "Virginia Tech",   "position": "F",   "rank": 29},
    {"name": "Zuby Ejiofor",          "team": "St. John's",      "position": "C",   "rank": 30},
    # ── 31-40 ──
    {"name": "Isaiah Evans",          "team": "Duke",            "position": "F",   "rank": 31},
    {"name": "Joshua Jefferson",      "team": "Iowa State",      "position": "F",   "rank": 32},
    {"name": "JT Toppin",             "team": "Texas Tech",      "position": "F",   "rank": 33},
    {"name": "Malachi Moreno",        "team": "Kentucky",        "position": "C",   "rank": 34},
    {"name": "Flory Bidunga",         "team": "Kansas",          "position": "C",   "rank": 35},
    {"name": "Christian Anderson",    "team": "Texas Tech",      "position": "G",   "rank": 36},
    {"name": "Dame Sarr",             "team": "Duke",            "position": "F",   "rank": 37},
    {"name": "Morez Johnson Jr.",     "team": "Michigan",        "position": "F",   "rank": 38},
    {"name": "Ebuka Okorie",          "team": "Stanford",        "position": "G",   "rank": 39},
    {"name": "Henri Veesaar",         "team": "North Carolina",  "position": "C",   "rank": 40},
    # ── 41-50 ──
    {"name": "Amari Allen",           "team": "Alabama",         "position": "F",   "rank": 41},
    {"name": "Anthony Robinson II",   "team": "Missouri",        "position": "G",   "rank": 42},
    {"name": "Darrion Williams",      "team": "NC State",        "position": "F",   "rank": 43},
    {"name": "Miles Byrd",            "team": "San Diego State", "position": "F",   "rank": 44},
    {"name": "Daniel Jacobsen",       "team": "Purdue",          "position": "C",   "rank": 45},
    {"name": "Joseph Tugler",         "team": "Houston",         "position": "F",   "rank": 46},
    {"name": "Brayden Burries",       "team": "Arizona",         "position": "G",   "rank": 47},
    {"name": "Tahaad Pettiford",      "team": "Auburn",          "position": "G",   "rank": 48},
    {"name": "Juke Harris",           "team": "Wake Forest",     "position": "F",   "rank": 49},
    {"name": "Richie Saunders",       "team": "BYU",             "position": "F",   "rank": 50},
    # ── 51-65 ──
    {"name": "Alex Karaban",          "team": "UConn",           "position": "F",   "rank": 51},
    {"name": "Tarris Reed Jr.",       "team": "UConn",           "position": "C",   "rank": 52},
    {"name": "Tyler Tanner",          "team": "Vanderbilt",      "position": "G",   "rank": 53},
    {"name": "Sebastian Williams-Adams", "team": "Auburn",       "position": "F",   "rank": 54},
    {"name": "Nate Bittle",           "team": "Oregon",          "position": "C",   "rank": 55},
    {"name": "Johann Grunloh",        "team": "Virginia",        "position": "C",   "rank": 56},
    {"name": "Billy Richmond",        "team": "Arkansas",        "position": "G",   "rank": 57},
    {"name": "Zvonimir Ivisic",       "team": "Illinois",        "position": "C",   "rank": 58},
    {"name": "Emanuel Sharp",         "team": "Houston",         "position": "G",   "rank": 59},
    {"name": "Mario Saint-Supery",    "team": "Gonzaga",         "position": "G",   "rank": 60},
    {"name": "Pryce Sandfort",        "team": "Nebraska",        "position": "F",   "rank": 61},
    {"name": "Milan Momcilovic",      "team": "Iowa State",      "position": "F",   "rank": 62},
    {"name": "Trevon Brazile",        "team": "Arkansas",        "position": "F",   "rank": 63},
    {"name": "Jaden Bradley",         "team": "Arizona",         "position": "G",   "rank": 64},
    {"name": "Dillon Mitchell",       "team": "St. John's",      "position": "F",   "rank": 65},
]

# ── G-League / NBA two-way prospects ──────────────────────────────────────────
GLEAGUE_PROSPECTS: list[dict] = []

# ── International prospects — no public shot-level API ────────────────────────
INTERNATIONAL_PROSPECTS: list[dict] = [
    {"name": "Karim Lopez",      "team": "NZ Breakers",  "position": "F", "rank": 18},
    {"name": "Dash Daniels",     "team": "Australia",    "position": "G", "rank": 37},
    {"name": "Sergio De Larrea", "team": "Spain",        "position": "G", "rank": 40},
]

# ── Unique teams (for optimised API calls — one call per team) ────────────────
NCAA_TEAMS: list[dict] = []
_seen: set[str] = set()
for p in NCAA_PROSPECTS:
    if p["team"] not in _seen:
        _seen.add(p["team"])
        NCAA_TEAMS.append({"team": p["team"]})

ALL_INGESTIBLE: list[dict] = NCAA_PROSPECTS + GLEAGUE_PROSPECTS
