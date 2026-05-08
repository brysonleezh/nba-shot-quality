# NBA Draft 2026 — Shot Quality Scouting Portal

A scouting tool built for the **2026 NBA Draft** that evaluates NCAA and international prospects through the lens of shot quality. The core question:

> *Is a prospect a shot-maker, or just a shot-taker?*

Built with `cbbd`, LightGBM, Empirical Bayes shrinkage, Plotly, and Streamlit.

---

## Core Metric: Points Above Expectation (PAE)

Each shot is assigned an expected point value by a LightGBM model trained on shot location, zone, shot type, and action type. PAE measures how many points a player generates **above** what an average shooter would score from the same shot attempts.

```
PAE = Actual Points − Expected Points (xPTS)
PAE/100 = PAE per 100 shot attempts  ← primary display metric
```

Because prospects have varying sample sizes, raw PAE is shrunk toward the class mean using **Empirical Bayes shrinkage**:

```
Shrunk PAE = λ × raw_PAE + (1 − λ) × class_mean

where  λ = n / (n + k)   (data trust factor)
       n = player's shot attempts
       k = prior strength (tuned to 150 attempts)
```

A player with few attempts gets pulled toward average; a player with 500+ attempts is trusted almost fully. The λ value is displayed on each player profile as a **"data trust"** indicator.

---

## Project Structure

```
nba-shot-quality/
├── pipeline/
│   ├── prospects_2026.py          # Master prospect list (62 NCAA + 3 intl)
│   ├── ingest_prospects.py        # cbbd PlaysApi → shots table in SQLite
│   ├── ingest_box_scores.py       # cbbd StatsApi → player_season_box table
│   ├── ingest_ncaa_historical.py  # Historical shot data for comp players
│   └── fetch_prospect_bios.py     # Name/school/position → prospect_bios.json
├── models/
│   └── xpts_model.py              # LightGBM xPTS model + Empirical Bayes shrinkage
├── dashboard/
│   └── app.py                     # Streamlit scouting portal (3 pages)
├── data/
│   ├── nba_shots.db               # SQLite: shots, player_season_summary, player_season_box
│   ├── prospect_bios.json         # Player metadata (school, position, height, etc.)
│   └── international_stats.json   # Manually curated stats for intl prospects
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Set your CollegeBasketballData.com API key:

```bash
export CBBD_API_KEY="your_key_here"
```

### 2. Ingest shot data (cbbd PlaysApi)

```bash
python pipeline/ingest_prospects.py
```

Pulls shot-by-shot play data for all 62 NCAA prospects from CollegeBasketballData.com and writes to the `shots` table.

### 3. Ingest box scores (cbbd StatsApi)

```bash
python pipeline/ingest_box_scores.py
```

Pulls full season box scores (FT%, TS%, USG%, PPG, RPG, APG, etc.) and writes to the `player_season_box` table. This is the source for stats not available in the shot-chart API (e.g., free throws).

### 4. Train the xPTS model and generate summaries

```bash
python models/xpts_model.py
```

Trains a LightGBM classifier on all ingested shots, computes expected points per shot, applies Empirical Bayes shrinkage, and populates the `player_season_summary` table.

### 5. Launch the dashboard

```bash
streamlit run dashboard/app.py
```

---

## Dashboard Pages

### Player Profile

A deep-dive card for a single prospect:

- **Bio strip** — school, position, draft rank, height/weight
- **Stat bar** — PPG, RPG, APG, FT%, TS%, USG% (from box scores)
- **Shot quality cards** — Attempts, FG%, xFG%, PAE/100 (shrunk), data trust (λ)
- **5-Dimension Radar** — Plotly interactive radar across five shooting dimensions (see below)
- **Zone Court** — Plotly interactive half-court diagram colored by zone-level performance vs class average

International prospects (Karim Lopez, Dash Daniels, Sergio De Larrea) route to a separate box-score-only profile since no public shot-chart API exists for their leagues.

### Compare Players (Head-to-Head)

Side-by-side comparison of any two prospects:

- **Bio cards** for each player
- **H2H Bar Chart** — Plotly back-to-back horizontal bars across 10+ metrics; both values are normalized to a [10, 90] display scale so the winner always has the longer bar regardless of absolute unit differences
- **Radar overlay** — Both players on a single 5-dimension Plotly radar with the class average shown as a dashed reference line

### Shooting Report

Class-wide leaderboard and zone breakdown:

- Filterable table: rank, PAE/100, FG%, xFG%, data trust
- Zone court for any selected player
- Quick stats panel (FT%, PPG, etc.)

---

## Five Shooting Dimensions

Each dimension is expressed as a **percentile rank** within the 2026 prospect class.

| Dimension | What it measures | Input |
|---|---|---|
| **Shot Making** | True shot-making ability above expectation | Shrunk PAE/100 percentile |
| **Outside Range** | Perimeter scoring threat | 3PAr × 3P% composite |
| **At-Rim** | Finishing ability around the basket | Restricted Area FG% |
| **Shot Diet** | Quality of shot selection (higher = harder shots) | Avg xFG% — inverted so tougher diets score higher |
| **Consistency** | Shot-to-shot reliability | PAE std dev — inverted so lower variance scores higher |

---

## Data Schema

### `shots` table

| Column | Description |
|---|---|
| `player_name` | Prospect name |
| `team` | NCAA team |
| `season` | e.g. `2025-26` |
| `loc_x / loc_y` | Court coordinates |
| `shot_zone_basic` | Zone label (e.g. `Mid-Range`, `Above the Break 3`) |
| `shot_made_flag` | 0 / 1 |
| `shot_type` | `2PT` / `3PT` |
| `action_type` | Jump Shot, Layup, Dunk, etc. |
| `p_make` | Model-predicted P(make) |
| `pae` | Points Above Expectation for this shot |

### `player_season_summary` table

| Column | Description |
|---|---|
| `player_name` | Prospect name |
| `fga / fgm / fg_pct` | Field goal attempts, makes, percentage |
| `xfg_pct` | Average expected FG% across all attempts |
| `raw_pae_per100` | Raw PAE per 100 attempts |
| `shrunk_pae_per100` | Empirical Bayes shrunk PAE/100 |
| `shrinkage_factor` | λ (data trust; 0 = fully shrunk, 1 = fully trusted) |
| `pae_std` | Standard deviation of per-shot PAE |
| `dim_*` | Percentile rank for each of the 5 shooting dimensions |

### `player_season_box` table

Full season box scores from cbbd StatsApi — 35 columns including games, minutes, points, FG/2P/3P/FT splits, rebounds, assists, blocks, steals, turnovers, eFG%, TS%, USG%, ortg, drtg, net_rtg, ast/to ratio.

---

## Prospect Coverage

**62 NCAA prospects** from the 2026 draft class (sources: FanSided Top 60 Big Board + No Ceilings Big Board V.6, April 2026), covering consensus top picks through mid-round targets.

**3 international prospects** with manually curated stats:
- **Karim Lopez** — NBL, New Zealand Breakers
- **Dash Daniels** — NBL, Melbourne United
- **Sergio De Larrea** — ACB + EuroLeague, Valencia Basket (combined + per-league splits)

---

## Methodology Notes

**xPTS model features:**
- Shot location (x/y coordinates, distance from basket)
- Shot zone (basic zone, area, range category)
- Shot type (2PT / 3PT)
- Action type (Jump Shot, Layup, Dunk, Hook Shot, etc.)

**Model:** LightGBM binary classifier predicting P(shot_made). Expected points = P(make) × point value (2 or 3).

**Shrinkage prior:** k = 150 attempts was chosen so that a prospect with one full college season (~300 FGA) receives λ ≈ 0.67 — meaningful but not fully trusted. A two-year player (~600 FGA) reaches λ ≈ 0.80.

---

## Future Work

- **Shot Mechanics tab** — MediaPipe Pose keypoint extraction from game film to evaluate elbow alignment, release angle, kinematic chain timing, and shot consistency across frames
- Defender distance / contest features (requires tracking data)
- G-League and two-way player coverage
- Multi-season trend view
