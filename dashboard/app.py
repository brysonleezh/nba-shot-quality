"""
2026 NBA Draft — Shot Quality Scouting Portal
----------------------------------------------
Pages (sidebar):
  🏆 Draft Board    — ranked leaderboard with sortable columns + headshots
  👤 Player Profile — bio card + shot analysis + player comps
  📄 Shooting Report — per-player shooting profile & metrics
"""

import io
import json
import sqlite3
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Arc
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
import streamlit as st

DB_PATH       = Path(__file__).resolve().parent.parent / "data" / "nba_shots.db"
BIOS_PATH     = Path(__file__).resolve().parent.parent / "data" / "prospect_bios.json"
INTL_PATH     = Path(__file__).resolve().parent.parent / "data" / "international_stats.json"
COMBINE_PATH  = Path(__file__).resolve().parent.parent / "data" / "combine_2026.json"
TARGET_SEASON = "2025-26"

# Combine uses legal names; map to names used in the shots DB / prospect list
_COMBINE_NAME_MAP = {
    "Anicet Dybantsa":       "AJ Dybantsa",
    "Christopher Brown Jr":  "Mikel Brown Jr.",
}

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
try:
    from prospects_2026 import NCAA_PROSPECTS, INTERNATIONAL_PROSPECTS
    ALL_PROSPECTS = NCAA_PROSPECTS + INTERNATIONAL_PROSPECTS
    PROSPECT_META = {p["name"]: p for p in ALL_PROSPECTS}
    INTL_NAMES    = {p["name"] for p in INTERNATIONAL_PROSPECTS}
except ImportError:
    ALL_PROSPECTS = []
    PROSPECT_META = {}
    INTL_NAMES    = set()

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="2026 NBA Draft Scouting",
    page_icon="🏀",
    layout="wide",
)

st.markdown("""
<style>
  .bio-card {
    background: linear-gradient(135deg, #f0f4ff 0%, #ffffff 100%);
    border: 1px solid #dde3f5;
    border-radius: 16px;
    padding: 24px 28px;
    display: flex;
    align-items: center;
    gap: 24px;
    margin-bottom: 8px;
  }
  .bio-info h1 {color:#1a1a2e; font-size:28px; font-weight:700; margin:0 0 4px;}
  .bio-info .subtitle {color:#f0a500; font-size:14px; font-weight:600; margin-bottom:12px;}
  .bio-pills {display:flex; flex-wrap:wrap; gap:8px; margin-top:8px;}
  .bio-pill {
    background:#eef1f8; border-radius:20px;
    padding:4px 12px; font-size:12px; color:#555;
  }
  .bio-pill span {color:#1a1a2e; font-weight:600;}

  .metric-card {
    background:#f5f7fa; border:1px solid #e4e8f0;
    border-radius:12px; padding:14px 16px; text-align:center;
  }
  .metric-label {color:#888; font-size:12px; margin-bottom:2px;}
  .metric-value {color:#1a1a2e; font-size:22px; font-weight:700;}
  .metric-sub   {color:#f0a500; font-size:11px; margin-top:2px;}

  .section-hdr {
    color:#1a1a2e; font-size:16px; font-weight:600;
    border-left:3px solid #f0b429;
    padding-left:10px; margin:24px 0 12px;
  }

  .rank-badge {
    background:#f0b429; color:#000;
    border-radius:6px; padding:2px 8px;
    font-size:11px; font-weight:700;
  }

  .comp-card {
    background:#f5f7fa; border:1px solid #e4e8f0;
    border-radius:12px; padding:12px 16px; margin-bottom:8px;
    display:flex; align-items:center; gap:14px;
  }
  .comp-rank {color:#f0a500; font-size:18px; font-weight:700; min-width:28px;}
  .comp-name {color:#1a1a2e; font-size:14px; font-weight:600;}
  .comp-team {color:#666; font-size:12px;}
  .comp-sim  {color:#2a6fc9; font-size:12px; font-weight:600;}

  /* Head-to-head comparison table */
  .h2h-table {width:100%; border-collapse:collapse; font-size:14px;}
  .h2h-table tr {border-bottom:1px solid #f0f0f5;}
  .h2h-table td {padding:9px 6px; vertical-align:middle;}
  .h2h-val {font-size:18px; font-weight:700;}
  .h2h-lbl {text-align:center; color:#999; font-size:11px; font-weight:500; width:110px;}
  .h2h-bar-wrap {position:relative; height:5px; background:#eef0f5;
                 border-radius:3px; width:100%; margin-top:5px;}
  .h2h-bar-a {position:absolute; right:50%; top:0; height:100%;
              border-radius:3px 0 0 3px; background:#f0b429;}
  .h2h-bar-b {position:absolute; left:50%; top:0; height:100%;
              border-radius:0 3px 3px 0; background:#4a90d9;}
  .h2h-win-a {color:#f0b429;}
  .h2h-win-b {color:#4a90d9;}
  .h2h-neutral {color:#555;}
</style>
""", unsafe_allow_html=True)


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=86400)
def load_box_scores() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(
            "SELECT * FROM player_season_box WHERE season='2025-26'", conn)
    except Exception:
        df = pd.DataFrame()
    conn.close()
    # DB stores pct columns as 0–100 integers; normalise to 0–1 decimals
    for col in ("fg_pct", "two_pct", "three_pct", "ft_pct",
                "efg_pct", "ts_pct", "usg_pct", "ft_rate"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce") / 100
    return df


@st.cache_data(ttl=86400)
def load_shots(league: str = "NCAA") -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(f"SELECT * FROM shots WHERE league='{league}'", conn)
    conn.close()
    return df


@st.cache_data(ttl=86400)
def load_summary(league: str | None = None, season: str | None = None) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    conditions, params = [], []
    if league:
        conditions.append("league = ?")
        params.append(league)
    if season:
        conditions.append("season = ?")
        params.append(season)
    q = "SELECT * FROM player_season_summary"
    if conditions:
        q += " WHERE " + " AND ".join(conditions)
    df = pd.read_sql_query(q, conn, params=params)
    conn.close()
    return df


@st.cache_data(ttl=3600)
def load_bios() -> dict:
    if BIOS_PATH.exists():
        return json.loads(BIOS_PATH.read_text())
    return {}


@st.cache_data(ttl=3600)
def load_intl_stats() -> dict:
    if INTL_PATH.exists():
        return json.loads(INTL_PATH.read_text())
    return {}


@st.cache_data(ttl=3600)
def _to_float(v):
    """Convert a combine field value to float, leave strings that aren't numeric as-is."""
    if v is None:
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return v


def load_combine() -> dict:
    """Returns dict: player_name (shots-DB spelling) → combine row dict."""
    if not COMBINE_PATH.exists():
        return {}
    records = json.loads(COMBINE_PATH.read_text())
    result = {}
    for r in records:
        name = r.get("PLAYER_NAME", "")
        mapped = _COMBINE_NAME_MAP.get(name, name)
        result[mapped] = {k: _to_float(v) for k, v in r.items()}
    return result


def _combine_rank(combine: dict, field: str, value, lower_is_better: bool = False) -> str:
    """Return 'Xth percentile' rank of value within combine participants."""
    if value is None:
        return ""
    vals = [r[field] for r in combine.values() if r.get(field) is not None]
    if not vals:
        return ""
    rank = sum(v < value for v in vals) / len(vals)
    if lower_is_better:
        rank = 1 - rank
    return f"{rank:.0%} pct"


def age_from_dob(dob: str) -> str:
    if not dob:
        return ""
    try:
        d = date.fromisoformat(dob)
        today = date.today()
        age = today.year - d.year - ((today.month, today.day) < (d.month, d.day))
        return str(age)
    except Exception:
        return ""


# ── Court drawing ─────────────────────────────────────────────────────────────

def draw_court(ax, color="#cccccc", lw=1.5):
    for el in [
        Circle((0, 0), 7.5, linewidth=lw, color=color, fill=False),
        Rectangle((-30, -7.5), 60, -1, linewidth=lw, color=color),
        Rectangle((-80, -47.5), 160, 190, linewidth=lw, color=color, fill=False),
        Rectangle((-60, -47.5), 120, 190, linewidth=lw, color=color, fill=False),
        Arc((0, 142.5), 120, 120, theta1=0,   theta2=180, linewidth=lw, color=color, fill=False),
        Arc((0, 142.5), 120, 120, theta1=180, theta2=0,   linewidth=lw, color=color,
            fill=False, linestyle="dashed"),
        Arc((0, 0), 80, 80, theta1=0, theta2=180, linewidth=lw, color=color, fill=False),
        Rectangle((-220, -47.5), 0, 140, linewidth=lw, color=color),
        Rectangle(( 220, -47.5), 0, 140, linewidth=lw, color=color),
        Arc((0, 0), 475, 475, theta1=22, theta2=158, linewidth=lw, color=color),
        Arc((0, 422.5), 120, 120, theta1=180, theta2=0, linewidth=lw, color=color),
    ]:
        ax.add_patch(el)
    ax.set_xlim(-250, 250)
    ax.set_ylim(-47.5, 470)
    ax.set_aspect("equal")
    ax.axis("off")
    return ax


def metric_card(col, label, value, sub=""):
    with col:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">{label}</div>
          <div class="metric-value">{value}</div>
          <div class="metric-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)


def section_header(title):
    st.markdown(f'<div class="section-hdr">{title}</div>', unsafe_allow_html=True)


# ── Player bio card ───────────────────────────────────────────────────────────

def render_bio_card(name: str, bios: dict, summary_row) -> None:
    bio  = bios.get(name, {})
    meta = PROSPECT_META.get(name, {})

    headshot = bio.get("headshot_url") or ""
    position = bio.get("position") or meta.get("position", "")
    team     = meta.get("team", bio.get("team", ""))
    rank     = meta.get("rank", "?")
    height   = bio.get("height", "")
    weight   = bio.get("weight", "")
    bp       = bio.get("birthplace", "")
    dob      = bio.get("dob", "")
    age      = age_from_dob(dob)
    jersey   = bio.get("jersey", "")

    pills = []
    if position: pills.append(("Pos", position))
    if height:   pills.append(("Height", height))
    if weight:   pills.append(("Weight", weight))
    if age:      pills.append(("Age", age))
    if bp:       pills.append(("From", bp))
    if jersey:   pills.append(("#", jersey))

    pills_html = "".join(
        f'<div class="bio-pill">{k}: <span>{v}</span></div>'
        for k, v in pills
    )

    img_html = ""
    if headshot:
        img_html = (f'<img src="{headshot}" style="width:110px;height:110px;'
                    f'border-radius:50%;object-fit:cover;'
                    f'border:3px solid #f0b429;background:#1a1a2e;" '
                    f'onerror="this.style.display=\'none\'">')

    st.markdown(f"""
    <div class="bio-card">
      {img_html}
      <div class="bio-info">
        <div style="margin-bottom:4px;">
          <span class="rank-badge">#{rank} Overall</span>
        </div>
        <h1>{name}</h1>
        <div class="subtitle">{team}</div>
        <div class="bio-pills">{pills_html}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ── Player Comps ──────────────────────────────────────────────────────────────

COMP_FEATURES  = ["fg_pct", "pct_3pt", "avg_shot_dist", "shrunk_pae_per100"]
HIST_SEASONS   = {"2022-23", "2023-24", "2024-25"}
TARGET_SEASON  = "2025-26"


def compute_comps(player_name: str, all_ncaa_summary: pd.DataFrame,
                  n: int = 5) -> pd.DataFrame:
    """
    Return top-n most similar historical NCAA players (2022-25) to a given
    2025-26 prospect, using Euclidean distance on standardised shooting-profile
    features. Similarity score = exp(-dist / median_dist) mapped to 0-100.
    """
    target_row = all_ncaa_summary[
        (all_ncaa_summary["player_name"] == player_name) &
        (all_ncaa_summary["season"] == TARGET_SEASON)
    ]
    if target_row.empty:
        return pd.DataFrame()

    feats = [f for f in COMP_FEATURES if f in all_ncaa_summary.columns]
    if not feats:
        return pd.DataFrame()

    hist = all_ncaa_summary[
        all_ncaa_summary["season"].isin(HIST_SEASONS)
    ].dropna(subset=feats).copy()

    if hist.empty:
        return pd.DataFrame()

    scaler = StandardScaler()
    hist_scaled   = scaler.fit_transform(hist[feats].values)
    target_scaled = scaler.transform(target_row[feats].fillna(0).values)

    dists = np.linalg.norm(hist_scaled - target_scaled[0], axis=1)
    median_dist = float(np.median(dists)) or 1.0
    # similarity: 100% = identical, decays as distance grows relative to median
    hist = hist.copy()
    hist["distance"]   = dists
    hist["similarity"] = np.exp(-dists / median_dist)   # (0, 1]
    top = hist.nsmallest(n, "distance")[["player_name", "season", "similarity"] + feats]
    return top.reset_index(drop=True)


# ── Page 1: Compare Players ───────────────────────────────────────────────────

def _player_metrics(name: str, shots: pd.DataFrame, summary: pd.DataFrame,
                    intl_stats: dict, box_scores: pd.DataFrame | None = None) -> dict:
    """Return a flat dict of display metrics for any prospect."""
    if name in intl_stats:
        ist = intl_stats[name]
        fga  = ist.get("fga", 0) or 0
        tpa  = ist.get("three_pa", 0) or 0
        fg   = ist.get("fg_pct", 0) or 0
        t3   = ist.get("three_pct", 0) or 0
        efg  = ist.get("efg_pct") or ((fg * fga + 0.5 * t3 * tpa) / fga if fga else None)
        return {
            "FG%":       fg,
            "3P%":       t3,
            "3PAr":      tpa / fga if fga else None,
            "FT%":       ist.get("ft_pct"),
            "eFG%":      efg,
            "TS%":       ist.get("ts_pct"),
            "PPG":       ist.get("ppg"),
            "RPG":       ist.get("rpg"),
            "APG":       ist.get("apg"),
            "USG%":      ist.get("usg_pct"),
            "PAE/100":   None,
            "Avg Dist":  None,
            "FGA":       fga * (ist.get("gp") or 1),
            "_is_intl":  True,
            "_league":   ist.get("league", ""),
        }

    row_df = summary[summary["player_name"] == name]
    if row_df.empty:
        return {"_is_intl": False}
    row  = row_df.iloc[0]
    pdata = shots[(shots["player_name"] == name) &
                  (shots["season"] == TARGET_SEASON)]
    threes = pdata[pdata["shot_type"] == "3PT Field Goal"]
    twos   = pdata[pdata["shot_type"] != "3PT Field Goal"]
    fg   = float(row.get("fg_pct", 0))
    par  = float(row.get("pct_3pt", 0))
    t3   = float(threes["shot_made"].mean()) if len(threes) >= 5 else None
    efg  = fg + 0.5 * par * (t3 or 0)

    ft_pct = ts_pct = ppg = rpg = apg = usg = None
    if box_scores is not None and not box_scores.empty:
        brow = box_scores[box_scores["player_name"] == name]
        if not brow.empty:
            b = brow.iloc[0]
            games  = b.get("games") or 0
            ft_pct = b.get("ft_pct")
            ts_pct = b.get("ts_pct")
            usg    = b.get("usg_pct")
            ppg    = (b.get("points") / games) if games else None
            rpg    = (b.get("reb_total") / games) if games else None
            apg    = (b.get("assists") / games) if games else None

    return {
        "FG%":      fg,
        "3P%":      t3,
        "3PAr":     par,
        "FT%":      ft_pct,
        "eFG%":     efg,
        "TS%":      ts_pct,
        "PPG":      ppg,
        "RPG":      rpg,
        "APG":      apg,
        "USG%":     usg,
        "PAE/100":  float(row["shrunk_pae_per100"]) if pd.notna(row.get("shrunk_pae_per100")) else None,
        "Avg Dist": float(row.get("avg_shot_dist", 0)),
        "FGA":      int(row.get("total_shots", 0)),
        "_is_intl": False,
        "_pdata":   pdata,
        "_row":     row,
    }


def _radar_compare(scores_a: dict, scores_b: dict,
                   name_a: str, name_b: str) -> go.Figure:
    short  = ["Making", "Range", "At-Rim", "Shot Diet", "Consistency"]
    full   = [
        "Shot Making (Shrunk PAE/100 %ile)",
        "Outside Range (3PAr × 3P% composite)",
        "At-Rim Finishing (Restricted Area FG%)",
        "Shot Diet (Difficulty, inv. %ile)",
        "Consistency (PAE variance, inv. %ile)",
    ]
    keys   = ["making", "range", "at_rim", "shot_diet", "consistency"]
    vals_a = [float(scores_a.get(k) or 50) for k in keys]
    vals_b = [float(scores_b.get(k) or 50) for k in keys]
    theta  = short + [short[0]]

    def hover(vals, nm):
        return [f"<b>{nm.split()[0]}</b><br>{d}<br>Percentile: <b>{v:.0f}th</b>"
                for d, v in zip(full, vals)] + [""]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=[50]*6, theta=theta, mode="lines",
        line=dict(color="#bbbbbb", width=1, dash="dash"),
        name="Class Avg", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatterpolar(
        r=vals_a + [vals_a[0]], theta=theta,
        fill="toself", fillcolor="rgba(240,180,41,0.18)",
        line=dict(color="#f0b429", width=2.5),
        marker=dict(size=8, color="#f0b429"),
        name=name_a.split()[0],
        hovertext=hover(vals_a, name_a), hoverinfo="text",
    ))
    fig.add_trace(go.Scatterpolar(
        r=vals_b + [vals_b[0]], theta=theta,
        fill="toself", fillcolor="rgba(74,144,217,0.15)",
        line=dict(color="#4a90d9", width=2.5),
        marker=dict(size=8, color="#4a90d9"),
        name=name_b.split()[0],
        hovertext=hover(vals_b, name_b), hoverinfo="text",
    ))
    # Value labels — offset outward so they sit just outside the markers
    _offset = 10
    fig.add_trace(go.Scatterpolar(
        r=[v + _offset for v in vals_a] + [vals_a[0] + _offset], theta=theta,
        mode="text",
        text=[f"<b>{v:.0f}</b>" for v in vals_a] + [""],
        textfont=dict(size=10, color="#c8921a"),
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatterpolar(
        r=[v + _offset for v in vals_b] + [vals_b[0] + _offset], theta=theta,
        mode="text",
        text=[f"<b>{v:.0f}</b>" for v in vals_b] + [""],
        textfont=dict(size=10, color="#2d6bb0"),
        showlegend=False, hoverinfo="skip",
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(range=[0, 115], tickvals=[25, 50, 75, 100],
                            tickfont=dict(size=9, color="#999"), gridcolor="#dee2e6"),
            angularaxis=dict(tickfont=dict(size=10, color="#1a1a2e",
                                           family="Arial Black"), gridcolor="#dee2e6"),
            bgcolor="#f8f9fa",
        ),
        showlegend=True,
        legend=dict(x=0.85, y=1.1, font=dict(size=11)),
        height=420, margin=dict(t=30, b=30, l=60, r=60),
        paper_bgcolor="white",
        hoverlabel=dict(bgcolor="white", bordercolor="#ddd", font=dict(size=12)),
    )
    return fig


def _h2h_bars(metrics_a: dict, metrics_b: dict,
              name_a: str, name_b: str) -> "plt.Figure":
    """Horizontal back-to-back bar chart for key shooting metrics."""
    rows = [
        ("FG%",      "FG%",      ".1%"),
        ("3P%",      "3P%",      ".1%"),
        ("3PAr",     "3PAr",     ".1%"),
        ("eFG%",     "eFG%",     ".1%"),
        ("PAE/100",  "PAE/100",  "+.1f"),
        ("Avg Dist", "Avg Dist", ".1f"),
    ]
    labels, vals_a, vals_b = [], [], []
    for label, key, _ in rows:
        va = metrics_a.get(key)
        vb = metrics_b.get(key)
        if va is None and vb is None:
            continue
        labels.append(label)
        vals_a.append(va if va is not None else 0)
        vals_b.append(vb if vb is not None else 0)

    n   = len(labels)
    y   = np.arange(n)
    fig, ax = plt.subplots(figsize=(5.5, max(3, n * 0.65)), facecolor="white")
    ax.set_facecolor("#f5f7fa")

    max_val = max(max(abs(v) for v in vals_a), max(abs(v) for v in vals_b), 0.01)

    bar_h = 0.35
    bars_a = ax.barh(y + bar_h/2, vals_a, bar_h, color="#f0b429", label=name_a.split()[0])
    bars_b = ax.barh(y - bar_h/2, vals_b, bar_h, color="#4a90d9",
                     label=name_b.split()[0], alpha=0.85)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, color="#333", fontsize=9)
    ax.axvline(0, color="#aaa", lw=0.8)
    ax.tick_params(colors="#333")
    ax.spines[:].set_color("#dee2e6")
    ax.legend(facecolor="white", fontsize=8, loc="lower right")
    ax.set_title("Metric Comparison", color="#1a1a2e", fontsize=10)
    fig.patch.set_facecolor("white")
    plt.tight_layout()
    return fig


def _cmp_metric_row(label: str, va, vb, fmt: str, higher_better: bool = True) -> None:
    """Render one comparison row: [value A] [label] [value B]."""
    def fmtv(v):
        if v is None: return "—"
        try:    return f"{v:{fmt}}"
        except: return "—"

    sa = fmtv(va); sb = fmtv(vb)

    if va is not None and vb is not None:
        a_wins = (va > vb) if higher_better else (va < vb)
    else:
        a_wins = None

    col_a_style = "color:#f0b429;font-weight:700;" if a_wins is True  else \
                  "color:#4a90d9;font-weight:700;" if a_wins is False else "color:#333;"
    col_b_style = "color:#4a90d9;font-weight:700;" if a_wins is True  else \
                  "color:#f0b429;font-weight:700;" if a_wins is False else "color:#333;"

    st.markdown(f"""
    <div style="display:grid;grid-template-columns:1fr auto 1fr;
                align-items:center;gap:8px;margin:6px 0;">
      <div style="text-align:right;font-size:18px;{col_a_style}">{sa}</div>
      <div style="text-align:center;color:#888;font-size:11px;
                  white-space:nowrap;min-width:90px;">{label}</div>
      <div style="text-align:left;font-size:18px;{col_b_style}">{sb}</div>
    </div>""", unsafe_allow_html=True)


def _h2h_table(ma: dict, mb: dict, name_a: str, name_b: str) -> None:
    """Interactive Plotly back-to-back bar chart for head-to-head comparison."""

    ROWS = [
        # (label,                       key,       fmt,    higher_better, description)
        ("Field Goal Attempts",          "FGA",    ".0f",  True,  "Total FG attempts this season"),
        ("Field Goal %",                 "FG%",    ".1%",  True,  "Overall field goal percentage"),
        ("3-Point %",                    "3P%",    ".1%",  True,  "3-point shooting percentage"),
        ("3PT Attempt Rate",             "3PAr",   ".1%",  True,  "% of all shots taken from 3-point range"),
        ("Effective FG%",                "eFG%",   ".1%",  True,  "(FGM + 0.5 × 3PM) / FGA — accounts for 3PT value"),
        ("Free Throw %",                 "FT%",    ".1%",  True,  "Free throw percentage"),
        ("Pts Above Expected / 100",     "PAE/100","+.1f", True,  "Shot quality vs xPTS model per 100 attempts"),
        ("Avg Shot Distance (ft)",       "Avg Dist",".1f", False, "Mean shot distance in feet"),
    ]

    def fmtv(v, fmt):
        if v is None: return "—"
        try:    return f"{v:{fmt}}"
        except: return "—"

    # Realistic domain for each metric — bars are proportional to actual values
    _DOMAINS = {
        "FGA":     (0,    500),
        "FG%":     (0.20, 0.65),
        "3P%":     (0.0,  0.55),
        "3PAr":    (0.0,  1.0),
        "eFG%":    (0.25, 0.75),
        "FT%":     (0.30, 1.0),
        "PAE/100": (-15,  20),
        "Avg Dist":(5,    28),
    }

    def norm_bars(va, vb, key, hb):
        """Scale values to [0,100] within a realistic domain for the metric."""
        lo, hi = _DOMAINS.get(key, (0, 1))
        def scale(v):
            if v is None:
                return 0.0
            return max(2.0, min(100.0, (v - lo) / (hi - lo) * 100))
        na, nb = scale(va), scale(vb)
        if not hb:   # lower is better → invert so shorter bar = worse
            na = 102.0 - na
            nb = 102.0 - nb
        return na, nb

    labels, bar_a, bar_b = [], [], []
    color_a, color_b     = [], []
    text_a,  text_b      = [], []
    hover_a, hover_b     = [], []

    na_first = name_a.split()[0]
    nb_first = name_b.split()[0]

    for label, key, fmt, hb, desc in ROWS:
        va = ma.get(key)
        vb = mb.get(key)
        if key == "FT%" and va is None: va = ma.get("TS%")
        if key == "FT%" and vb is None: vb = mb.get("TS%")
        if va is None and vb is None:
            continue

        a_wins = (va > vb) if (hb and va is not None and vb is not None) else \
                 (va < vb) if (not hb and va is not None and vb is not None) else None

        na, nb = norm_bars(va, vb, key, hb)
        labels.append(label)
        bar_a.append(-na)   # negative → goes left
        bar_b.append(nb)    # positive → goes right
        color_a.append("#f0b429" if a_wins is True  else
                        "#f5d98a" if a_wins is False else "#f0b429")
        color_b.append("#4a90d9" if a_wins is False else
                        "#a0c4e8" if a_wins is True  else "#4a90d9")
        text_a.append(fmtv(va, fmt))
        text_b.append(fmtv(vb, fmt))
        win_tag_a = "  ✓" if a_wins is True  else ("  ✗" if a_wins is False else "")
        win_tag_b = "  ✓" if a_wins is False else ("  ✗" if a_wins is True  else "")
        hover_a.append(f"<b>{na_first}</b>: {fmtv(va,fmt)}{win_tag_a}<br>"
                       f"<i>{label}</i><br>{desc}")
        hover_b.append(f"<b>{nb_first}</b>: {fmtv(vb,fmt)}{win_tag_b}<br>"
                       f"<i>{label}</i><br>{desc}")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=labels, x=bar_a, orientation="h",
        name=na_first, marker_color=color_a,
        text=text_a, textposition="outside",
        textfont=dict(size=10, color="#c8921a"),
        hovertext=hover_a, hoverinfo="text",
        cliponaxis=False,
    ))
    fig.add_trace(go.Bar(
        y=labels, x=bar_b, orientation="h",
        name=nb_first, marker_color=color_b,
        text=text_b, textposition="outside",
        textfont=dict(size=10, color="#2d6bb0"),
        hovertext=hover_b, hoverinfo="text",
        cliponaxis=False,
    ))
    fig.update_layout(
        barmode="relative",
        xaxis=dict(
            range=[-125, 125], showgrid=True, gridcolor="#eeeeee",
            zeroline=True, zerolinecolor="#888888", zerolinewidth=2,
            tickvals=[], showticklabels=False,
        ),
        yaxis=dict(
            tickfont=dict(size=11, color="#444"),
            autorange="reversed",
        ),
        height=max(260, len(labels) * 42),
        margin=dict(t=8, b=8, l=190, r=70),
        paper_bgcolor="white", plot_bgcolor="#fafafa",
        legend=dict(orientation="h", y=1.04, x=0.5, xanchor="center",
                    font=dict(size=12)),
        hoverlabel=dict(bgcolor="white", bordercolor="#cccccc",
                        font=dict(size=12)),
        uniformtext=dict(minsize=8, mode="hide"),
    )
    fig.add_annotation(x=-62, y=-0.6, xref="x", yref="paper",
                       text=f"← {na_first}", showarrow=False,
                       font=dict(size=10, color="#f0b429"))
    fig.add_annotation(x=62, y=-0.6, xref="x", yref="paper",
                       text=f"{nb_first} →", showarrow=False,
                       font=dict(size=10, color="#4a90d9"))
    st.plotly_chart(fig, use_container_width=True)


# ── PAE/100 class distribution (Plotly) ──────────────────────────────────────

def _pae_dist_fig(all_pae: pd.Series,
                  highlights: list,
                  title: str = "Shot Making  —  PAE/100  Class Distribution") -> go.Figure:
    """KDE + histogram + jittered player dots + annotated player highlights."""
    from scipy.stats import gaussian_kde as _gkde

    vals = all_pae.dropna().values
    if len(vals) < 3:
        return go.Figure()

    lo, hi = vals.min() - 1.5, vals.max() + 1.5
    x_kde = np.linspace(lo, hi, 300)
    try:
        _kde = _gkde(vals, bw_method=0.5)
        y_kde = _kde(x_kde)
    except Exception:
        y_kde = np.zeros_like(x_kde)

    y_max = y_kde.max() if y_kde.max() > 0 else 1.0
    DOT_Y = -y_max * 0.14

    fig = go.Figure()

    # histogram
    fig.add_trace(go.Histogram(
        x=vals, histnorm="probability density", nbinsx=22,
        marker=dict(color="#dae4f5", line=dict(color="#b8cce8", width=0.5)),
        name="2026 Class", hoverinfo="skip", opacity=0.85,
    ))

    # KDE curve
    fig.add_trace(go.Scatter(
        x=x_kde, y=y_kde, mode="lines",
        line=dict(color="#4a90d9", width=2),
        showlegend=False, hoverinfo="skip",
    ))

    # zero reference
    fig.add_vline(x=0, line=dict(color="#aaaaaa", width=1, dash="dot"))

    # individual player dots (jittered along baseline)
    rng = np.random.default_rng(42)
    jitter = rng.uniform(-y_max * 0.035, y_max * 0.035, len(vals))
    fig.add_trace(go.Scatter(
        x=vals, y=np.full(len(vals), DOT_Y) + jitter,
        mode="markers",
        marker=dict(size=5, color="#9aaac2", opacity=0.50),
        showlegend=False,
        hovertemplate="%{x:.1f}<extra></extra>",
    ))

    # highlighted players
    for nm, val, clr in highlights:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        pct = float((vals < val).mean() * 100)
        try:
            kde_at_val = float(_gkde(vals, bw_method=0.5)(np.array([val]))[0])
        except Exception:
            kde_at_val = y_max * 0.5

        fig.add_shape(type="line",
                      x0=val, x1=val,
                      y0=DOT_Y - y_max * 0.04,
                      y1=kde_at_val + y_max * 0.02,
                      line=dict(color=clr, width=2, dash="dot"))

        fig.add_trace(go.Scatter(
            x=[val], y=[DOT_Y],
            mode="markers",
            marker=dict(size=14, color=clr,
                        line=dict(color="white", width=2)),
            name=f"{nm.split()[0]}  {val:+.1f}  ({pct:.0f}th %ile)",
            hovertemplate=(f"<b>{nm.split()[0]}</b><br>"
                           f"PAE/100: <b>{val:+.1f}</b><br>"
                           f"Percentile: <b>{pct:.0f}th</b><extra></extra>"),
        ))

        fig.add_annotation(
            x=val, y=kde_at_val + y_max * 0.15,
            text=(f"<b>{nm.split()[0]}</b><br>"
                  f"{val:+.1f} · {pct:.0f}th %ile"),
            showarrow=True, arrowhead=2,
            arrowcolor=clr, arrowwidth=1.5, ax=0, ay=-32,
            font=dict(size=10, color=clr),
            bgcolor="white", bordercolor=clr, borderwidth=1.5, borderpad=5,
            align="center",
        )

    fig.update_layout(
        title=dict(text=title, font=dict(size=12, color="#1a1a2e"), x=0.5),
        xaxis=dict(title="Shrunk PAE/100", gridcolor="#eeeeee",
                   tickfont=dict(size=9, color="#555"),
                   zerolinecolor="#aaaaaa", zerolinewidth=1),
        yaxis=dict(range=[DOT_Y - y_max * 0.18, y_max * 1.45],
                   showticklabels=False, showgrid=False, zeroline=False),
        height=300,
        margin=dict(t=55, b=40, l=10, r=10),
        paper_bgcolor="white", plot_bgcolor="#fafafa",
        legend=dict(orientation="h", y=-0.22, x=0.5, xanchor="center",
                    font=dict(size=10)),
        hoverlabel=dict(bgcolor="white", bordercolor="#ccc", font=dict(size=11)),
        barmode="overlay",
    )
    return fig


# ── Zone-coloured half-court diagram ─────────────────────────────────────────

_ZONE_CLASS_DEFAULTS = {
    "Restricted Area":       0.63,
    "In The Paint (Non-RA)": 0.41,
    "Mid-Range":             0.38,
    "Left Corner 3":         0.36,
    "Right Corner 3":        0.36,
    "Above the Break 3":     0.35,
}

def _zone_court_fig(pdata: pd.DataFrame, name: str,
                    shots_all: pd.DataFrame | None = None,
                    title_color: str = "#f0b429",
                    show_cbar: bool = True,
                    figsize: tuple = (4.8, 5.2)) -> "plt.Figure":
    """Half-court coloured by FG% vs class average per zone.
    Green = above class avg, red = below.  Grey = insufficient data (<3 att).
    """
    zone_stats = (pdata.groupby("shot_zone_basic")
                  .agg(att=("shot_attempted", "sum"), made=("shot_made", "sum"))
                  .assign(fg_pct=lambda d: d["made"] / d["att"].clip(lower=1)))

    class_avg = dict(_ZONE_CLASS_DEFAULTS)
    if shots_all is not None and not shots_all.empty:
        sa = shots_all[shots_all["season"] == TARGET_SEASON]
        if not sa.empty:
            ca = (sa.groupby("shot_zone_basic")
                  .agg(att=("shot_attempted","sum"), made=("shot_made","sum"))
                  .assign(fg_pct=lambda d: d["made"] / d["att"].clip(lower=1)))
            class_avg.update(ca["fg_pct"].to_dict())

    MIN_ATT = 3

    def rdylgn(norm_0_1: float) -> str:
        """Scalar [0,1] → CSS rgba string using RdYlGn palette."""
        n = np.clip(norm_0_1, 0, 1)
        if n < 0.5:
            r, g = 220, int(220 * n * 2)
        else:
            r, g = int(220 * (1 - n) * 2), 190
        return f"rgba({r},{g},30,0.80)"

    def zone_rgba(zone: str) -> str:
        if zone not in zone_stats.index or zone_stats.loc[zone, "att"] < MIN_ATT:
            return "rgba(190,195,205,0.45)"
        fg  = zone_stats.loc[zone, "fg_pct"]
        ca  = class_avg.get(zone, 0.45)
        return rdylgn(np.clip((fg - ca + 0.20) / 0.40, 0, 1))

    fig = go.Figure()

    # ── Zone fill polygons (largest → smallest so smaller zones paint on top) ──
    # 1. Above the Break 3  — full court background
    fig.add_trace(go.Scatter(
        x=[-250, 250, 250, -250, -250], y=[-47.5, -47.5, 360, 360, -47.5],
        fill="toself", fillcolor=zone_rgba("Above the Break 3"),
        line=dict(width=0), mode="lines", hoverinfo="skip", showlegend=False,
    ))

    # 2. Mid-Range  — inside 3PT arc
    t3 = np.linspace(np.radians(22), np.radians(158), 80)
    ax3, ay3 = 237.5 * np.cos(t3), 237.5 * np.sin(t3)
    fig.add_trace(go.Scatter(
        x=[220, *ax3, -220, -220, 220], y=[-47.5, *ay3, -47.5, -47.5, -47.5],
        fill="toself", fillcolor=zone_rgba("Mid-Range"),
        line=dict(width=0), mode="lines", hoverinfo="skip", showlegend=False,
    ))

    # 3. Left Corner 3
    fig.add_trace(go.Scatter(
        x=[-250, -220, -220, -250, -250], y=[-47.5, -47.5, 92.5, 92.5, -47.5],
        fill="toself", fillcolor=zone_rgba("Left Corner 3"),
        line=dict(width=0), mode="lines", hoverinfo="skip", showlegend=False,
    ))

    # 4. Right Corner 3
    fig.add_trace(go.Scatter(
        x=[220, 250, 250, 220, 220], y=[-47.5, -47.5, 92.5, 92.5, -47.5],
        fill="toself", fillcolor=zone_rgba("Right Corner 3"),
        line=dict(width=0), mode="lines", hoverinfo="skip", showlegend=False,
    ))

    # 5. In The Paint (Non-RA)
    fig.add_trace(go.Scatter(
        x=[-80, 80, 80, -80, -80], y=[-47.5, -47.5, 142.5, 142.5, -47.5],
        fill="toself", fillcolor=zone_rgba("In The Paint (Non-RA)"),
        line=dict(width=0), mode="lines", hoverinfo="skip", showlegend=False,
    ))

    # 6. Restricted Area  — semicircle
    tra = np.linspace(0, np.pi, 40)
    rax, ray = list(40 * np.cos(tra)), list(40 * np.sin(tra))
    fig.add_trace(go.Scatter(
        x=rax + [-40, 40], y=ray + [-47.5, -47.5],
        fill="toself", fillcolor=zone_rgba("Restricted Area"),
        line=dict(width=0), mode="lines", hoverinfo="skip", showlegend=False,
    ))

    # ── Court lines ───────────────────────────────────────────────────────────
    CL = "#3a4a5a"
    LW = 1.5

    def cline(x, y, dash=None):
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines",
            line=dict(color=CL, width=LW, **({} if not dash else {"dash": dash})),
            hoverinfo="skip", showlegend=False,
        ))

    cline([-250, 250],   [-47.5, -47.5])               # baseline
    cline([-250, -250],  [-47.5, 360])                  # left sideline
    cline([250,  250],   [-47.5, 360])                  # right sideline
    cline([-80, 80, 80, -80, -80],                      # outer paint
          [-47.5, -47.5, 142.5, 142.5, -47.5])
    cline([-60, -60], [-47.5, 142.5])                   # lane lines
    cline([ 60,  60], [-47.5, 142.5])
    cline([-30, 30],     [-7.5, -7.5])                  # backboard
    tb = np.linspace(0, 2*np.pi, 40)                    # basket circle
    cline(list(7.5*np.cos(tb)), list(7.5*np.sin(tb)))
    cline(list(40*np.cos(tra)), list(40*np.sin(tra)))   # RA arc
    cline([-220, -220], [-47.5, 92.5])                  # corner 3 lines
    cline([ 220,  220], [-47.5, 92.5])
    cline(list(ax3), list(ay3))                         # 3PT arc
    tft = np.linspace(0, np.pi, 40)
    cline(list(60*np.cos(tft)),        list(142.5 + 60*np.sin(tft)))        # FT front
    cline(list(60*np.cos(tft+np.pi)), list(142.5 + 60*np.sin(tft+np.pi)),  # FT back
          dash="dot")

    # ── Hover zones (invisible large markers over each zone centre) ───────────
    ZONE_POS = {
        "Restricted Area":       (  0,  15),
        "In The Paint (Non-RA)": (  0,  95),
        "Mid-Range":             (  0, 185),
        "Left Corner 3":         (-235, 25),
        "Right Corner 3":        ( 235, 25),
        "Above the Break 3":     (  0, 305),
    }
    hx, hy, htxt, hlabel = [], [], [], []
    for zone, (x, y) in ZONE_POS.items():
        if zone in zone_stats.index and zone_stats.loc[zone, "att"] >= MIN_ATT:
            r   = zone_stats.loc[zone]
            fg  = r["fg_pct"]; att = int(r["att"]); made = int(r["made"])
            ca  = class_avg.get(zone, 0.45); delta = fg - ca
            htxt.append(
                f"<b>{zone}</b><br>"
                f"FG%: <b>{fg:.1%}</b>  ({made}/{att})<br>"
                f"Class Avg: {ca:.1%}<br>"
                f"vs Avg: <b>{'+'if delta>=0 else ''}{delta:.1%}</b>"
            )
            hlabel.append(f"{fg:.0%}<br>{att}")
        else:
            htxt.append(f"<b>{zone}</b><br>Insufficient data (< {MIN_ATT} att)")
            hlabel.append("")
        hx.append(x); hy.append(y)

    fig.add_trace(go.Scatter(
        x=hx, y=hy, mode="markers+text",
        marker=dict(size=34, color="rgba(0,0,0,0.01)",
                    line=dict(width=0)),
        text=hlabel,
        textfont=dict(color="white", size=9, family="Arial Black"),
        textposition="middle center",
        hovertext=htxt, hoverinfo="text",
        showlegend=False,
        hoverlabel=dict(bgcolor="white", bordercolor="#cccccc",
                        font=dict(size=12)),
    ))

    # ── Colorbar (show_cbar) ──────────────────────────────────────────────────
    if show_cbar:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(colorscale="RdYlGn", showscale=True,
                        cmin=-0.20, cmax=0.20,
                        colorbar=dict(
                            title=dict(text="vs Class Avg", font=dict(size=10)),
                            tickvals=[-0.20, -0.10, 0, 0.10, 0.20],
                            ticktext=["-20%", "-10%", "Avg", "+10%", "+20%"],
                            thickness=12, len=0.55, x=1.01,
                            tickfont=dict(size=9),
                        )),
            hoverinfo="skip", showlegend=False,
        ))

    fig.update_layout(
        title=dict(text=name, font=dict(size=12, color=title_color), x=0.5),
        xaxis=dict(range=[-255, 255], showgrid=False, zeroline=False,
                   showticklabels=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(range=[-52, 365],  showgrid=False, zeroline=False,
                   showticklabels=False),
        height=420, margin=dict(t=36, b=4, l=4, r=4),
        paper_bgcolor="white", plot_bgcolor="#dce3ec",
        showlegend=False,
    )
    return fig


def page_compare_players(shots: pd.DataFrame, summary: pd.DataFrame,
                         bios: dict, intl_stats: dict,
                         all_scores: pd.DataFrame,
                         box_scores: pd.DataFrame | None = None) -> None:
    st.markdown("## ⚖️ Compare Players")

    sort_col     = "shrunk_pae_per100" if "shrunk_pae_per100" in summary.columns else "pts_above_exp"
    ncaa_ordered = summary.sort_values(sort_col, ascending=False)["player_name"].tolist()
    intl_ordered = [n for n in intl_stats if n not in ncaa_ordered]
    all_ordered  = ncaa_ordered + intl_ordered

    def pos_of(name): return PROSPECT_META.get(name, {}).get("position", "")

    # ── Step 1: Position filter (top of page) ────────────────────────────────
    st.markdown("**Position Group**")
    pos_filter = st.radio("pos", ["All", "G", "F", "C"],
                          horizontal=True, label_visibility="collapsed",
                          key="cmp_pos")

    filtered = [n for n in all_ordered
                if pos_filter == "All" or pos_of(n) == pos_filter]
    if len(filtered) < 2:
        st.warning(f"Not enough {pos_filter} players in the prospect pool.")
        return

    # ── Step 2: Player selectors (in main area, two columns) ─────────────────
    st.markdown("---")
    sel_a, sel_gap, sel_b = st.columns([5, 1, 5])

    with sel_a:
        st.markdown('<p style="color:#f0b429;font-weight:700;margin:0 0 4px;">Player A</p>',
                    unsafe_allow_html=True)
        player_a = st.selectbox("A", filtered, key="cmp_a",
                                label_visibility="collapsed")

    with sel_gap:
        st.markdown(
            '<div style="text-align:center;padding-top:28px;font-size:18px;'
            'font-weight:800;color:#ccc;">VS</div>',
            unsafe_allow_html=True)

    # Player B auto-filtered to same position as Player A
    a_pos  = pos_of(player_a)
    b_pool = [n for n in filtered if n != player_a and
              (pos_filter != "All" or pos_of(n) == a_pos or not a_pos)]
    if len(b_pool) == 0:
        b_pool = [n for n in filtered if n != player_a]

    with sel_b:
        st.markdown('<p style="color:#4a90d9;font-weight:700;margin:0 0 4px;">Player B</p>',
                    unsafe_allow_html=True)
        player_b = st.selectbox("B", b_pool, key="cmp_b",
                                label_visibility="collapsed")

    st.markdown("---")

    ma = _player_metrics(player_a, shots, summary, intl_stats, box_scores)
    mb = _player_metrics(player_b, shots, summary, intl_stats, box_scores)

    # ── Bio cards ─────────────────────────────────────────────────────────────
    _empty = type("", (), {})()
    ca, cm, cb = st.columns([10, 1, 10])
    with ca:
        row_a = ma.get("_row")
        render_bio_card(player_a, bios, row_a if row_a is not None else _empty)
    with cm:
        st.markdown(
            '<div style="text-align:center;padding-top:36px;font-size:22px;'
            'font-weight:800;color:#f0b429;">VS</div>',
            unsafe_allow_html=True)
    with cb:
        row_b = mb.get("_row")
        render_bio_card(player_b, bios, row_b if row_b is not None else _empty)

    # ── Head-to-head + radar ──────────────────────────────────────────────────
    section_header("Head-to-Head")
    col_tbl, col_radar = st.columns([1, 1])

    with col_tbl:
        _h2h_table(ma, mb, player_a, player_b)

    with col_radar:
        sa_row   = all_scores[all_scores["player_name"] == player_a]
        sb_row   = all_scores[all_scores["player_name"] == player_b]
        scores_a = sa_row.iloc[0].to_dict() if not sa_row.empty else {}
        scores_b = sb_row.iloc[0].to_dict() if not sb_row.empty else {}
        if scores_a or scores_b:
            fig = _radar_compare(scores_a, scores_b, player_a, player_b)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Radar unavailable — run xPTS model first.")

    # ── Zone breakdown (NCAA only) ────────────────────────────────────────────
    pdata_a = ma.get("_pdata")
    pdata_b = mb.get("_pdata")

    if (pdata_a is not None and not pdata_a.empty and
            pdata_b is not None and not pdata_b.empty):

        section_header("Zone Breakdown")

        cz_a, cz_b = st.columns(2)
        with cz_a:
            fig = _zone_court_fig(pdata_a, player_a, shots,
                                  title_color="#f0b429", show_cbar=False)
            st.plotly_chart(fig, use_container_width=True)
        with cz_b:
            fig = _zone_court_fig(pdata_b, player_b, shots,
                                  title_color="#4a90d9", show_cbar=False)
            st.plotly_chart(fig, use_container_width=True)
        st.caption("Hover over each zone for FG%, attempts, and vs class average")

        # PAE class distribution
        if "shrunk_pae_per100" in summary.columns:
            section_header("PAE/100  —  Class Context")
            all_pae = summary["shrunk_pae_per100"].dropna()
            pa_val  = ma.get("PAE/100"); pb_val = mb.get("PAE/100")
            hl = []
            if pa_val is not None: hl.append((player_a, pa_val, "#f0b429"))
            if pb_val is not None: hl.append((player_b, pb_val, "#4a90d9"))
            fig = _pae_dist_fig(all_pae, hl)
            st.plotly_chart(fig, use_container_width=True)


# ── International player profile ─────────────────────────────────────────────

def _page_intl_profile(player: str, istats: dict, bios: dict) -> None:
    """Profile view for international prospects (box-score + advanced, no xPTS)."""

    class _Row:
        pass
    render_bio_card(player, bios, _Row())

    league = istats.get("league", "International")
    team   = istats.get("team", "")
    source = istats.get("source", "")

    st.info(
        f"**{player}** plays in the **{league}** ({team}). "
        "Shot-chart data is unavailable for this league — the xPTS / PAE model "
        "(trained on NCAA data) does not apply. Full box-score and advanced stats shown below.",
        icon="🌍",
    )

    def _f(v, fmt, suffix=""):
        return f"{v:{fmt}}{suffix}" if v is not None else "—"

    # ── Section 1: Core Box Score ─────────────────────────────────────────────
    section_header("1 · Core Stats")
    st.caption(f"Source: {source}  ·  {istats.get('season', TARGET_SEASON)}")

    r1 = st.columns(7)
    metric_card(r1[0], "GP",  str(istats.get("gp", "—")),            "Games")
    metric_card(r1[1], "MPG", _f(istats.get("mpg"),  ".1f"),          "Min / game")
    metric_card(r1[2], "PPG", _f(istats.get("ppg"),  ".1f"),          "Pts / game")
    metric_card(r1[3], "RPG", _f(istats.get("rpg"),  ".1f"),          "Reb / game")
    metric_card(r1[4], "APG", _f(istats.get("apg"),  ".1f"),          "Ast / game")
    metric_card(r1[5], "SPG", _f(istats.get("spg"),  ".1f"),          "Stl / game")
    metric_card(r1[6], "BPG", _f(istats.get("bpg"),  ".1f"),          "Blk / game")
    st.markdown("<br>", unsafe_allow_html=True)

    r2 = st.columns(7)
    metric_card(r2[0], "FGA",  _f(istats.get("fga"),      ".1f"),     "Att / game")
    metric_card(r2[1], "FG%",  _f(istats.get("fg_pct"),   ".1%"),     "Field goal %")
    metric_card(r2[2], "3PA",  _f(istats.get("three_pa"), ".1f"),     "3-pt att / game")
    metric_card(r2[3], "3P%",  _f(istats.get("three_pct"),".1%"),     "3-point %")
    metric_card(r2[4], "FTA",  _f(istats.get("fta"),      ".1f"),     "FT att / game")
    ft_note = istats.get("ft_note", "")
    metric_card(r2[5], "FT%",  _f(istats.get("ft_pct"),   ".1%"),
                "Small sample — see note" if ft_note else "Free throw %")
    metric_card(r2[6], "TOV",  _f(istats.get("tov"),      ".1f"),     "TO / game")
    st.markdown("<br>", unsafe_allow_html=True)

    if ft_note:
        st.caption(f"⚠️ FT%: {ft_note}")

    # ── Section 2: Advanced Metrics ───────────────────────────────────────────
    section_header("2 · Advanced Metrics")

    r3 = st.columns(6)
    metric_card(r3[0], "eFG%",   _f(istats.get("efg_pct"), ".1%"),   "Effective FG%")
    metric_card(r3[1], "TS%",    _f(istats.get("ts_pct"),  ".1%"),   "True shooting %")
    metric_card(r3[2], "USG%",   _f(istats.get("usg_pct"), ".1%"),   "Usage rate")
    metric_card(r3[3], "AST/TO", _f(istats.get("ast_to"),  ".2f"),   "Assist/turnover")
    per = istats.get("per")
    metric_card(r3[4], "PER",    _f(per, ".1f") if per else "—",      "Player eff. rating")
    ortg = istats.get("ortg"); drtg = istats.get("drtg")
    net  = round(ortg - drtg, 1) if ortg and drtg else None
    metric_card(r3[5], "Net Rtg", _f(net, "+.1f") if net else "—",   f"ORtg {_f(ortg,'.0f')} / DRtg {_f(drtg,'.0f')}")
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Section 3: Shooting Visualisation ─────────────────────────────────────
    section_header("3 · Shooting Profile")

    has_splits = "splits" in istats
    n_charts   = 3 if has_splits else 2
    chart_cols = st.columns(n_charts)

    fg  = istats.get("fg_pct",    0) or 0
    t3  = istats.get("three_pct", 0) or 0
    ft  = istats.get("ft_pct",    0) or 0
    fga = istats.get("fga",       1) or 1
    tpa = istats.get("three_pa",  0) or 0
    efg = istats.get("efg_pct") or (fg * fga + 0.5 * t3 * tpa) / fga
    ts  = istats.get("ts_pct") or efg

    # Chart A: shooting % bar
    with chart_cols[0]:
        fig, ax = plt.subplots(figsize=(4.5, 3.5), facecolor="white")
        ax.set_facecolor("#f5f7fa")
        labels = ["FG%", "3P%", "FT%", "eFG%", "TS%"]
        vals   = [fg, t3, ft, efg, ts]
        clrs   = ["#f0b429", "#4a90d9", "#e05c5c", "#50c878", "#9b59b6"]
        bars   = ax.bar(labels, vals, color=clrs, width=0.55)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.008,
                    f"{val:.1%}", ha="center", va="bottom",
                    color="#333", fontsize=8, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.tick_params(colors="#333"); ax.spines[:].set_color("#dee2e6")
        ax.set_title("Shooting Percentages", color="#1a1a2e", fontsize=10)
        fig.patch.set_facecolor("white"); plt.tight_layout()
        st.pyplot(fig); plt.close(fig)

    # Chart B: shot diet donut
    with chart_cols[1]:
        fig, ax = plt.subplots(figsize=(4.5, 3.5), facecolor="white")
        two_pa = max(fga - tpa, 0)
        wedge_vals = [two_pa, tpa]
        wedge_lbls = [f"2PA  {two_pa/fga:.0%}", f"3PA  {tpa/fga:.0%}"]
        wedges, texts = ax.pie(
            wedge_vals, labels=wedge_lbls,
            colors=["#f0b429", "#4a90d9"], startangle=90,
            wedgeprops=dict(linewidth=2.5, edgecolor="white"),
        )
        for t in texts:
            t.set_color("#333"); t.set_fontsize(9)
        ax.add_artist(plt.Circle((0, 0), 0.62, fc="white"))
        ax.text(0, 0.06, f"eFG%", ha="center", va="center",
                fontsize=9, color="#888")
        ax.text(0, -0.1, f"{efg:.1%}", ha="center", va="center",
                fontsize=13, fontweight="bold", color="#1a1a2e")
        ax.set_title("Shot Diet", color="#1a1a2e", fontsize=10)
        fig.patch.set_facecolor("white"); plt.tight_layout()
        st.pyplot(fig); plt.close(fig)

    # Chart C: league splits (only for multi-league players)
    if has_splits:
        with chart_cols[2]:
            splits = istats["splits"]
            s_names = list(splits.keys())
            s_fg  = [splits[s].get("fg_pct",    0) for s in s_names]
            s_3p  = [splits[s].get("three_pct", 0) for s in s_names]
            s_ft  = [splits[s].get("ft_pct",    0) for s in s_names]

            fig, ax = plt.subplots(figsize=(4.5, 3.5), facecolor="white")
            ax.set_facecolor("#f5f7fa")
            x     = np.arange(len(s_names))
            width = 0.25
            ax.bar(x - width, s_fg, width, color="#f0b429", label="FG%")
            ax.bar(x,         s_3p, width, color="#4a90d9", label="3P%")
            ax.bar(x + width, s_ft, width, color="#e05c5c", label="FT%")
            ax.set_xticks(x); ax.set_xticklabels(s_names, color="#333", fontsize=9)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
            ax.set_ylim(0, 1.0)
            ax.tick_params(colors="#333"); ax.spines[:].set_color("#dee2e6")
            ax.legend(facecolor="white", fontsize=8)
            ax.set_title("Shooting by League", color="#1a1a2e", fontsize=10)
            fig.patch.set_facecolor("white"); plt.tight_layout()
            st.pyplot(fig); plt.close(fig)

    # Per-league splits table for multi-league players
    if has_splits:
        section_header("4 · League Splits")
        splits = istats["splits"]
        rows = []
        for lg_name, lg in splits.items():
            rows.append({
                "League": lg_name,
                "GP":   lg.get("gp", "—"),
                "MPG":  f"{lg.get('mpg', 0):.1f}" if lg.get("mpg") else "—",
                "PPG":  f"{lg.get('ppg', 0):.1f}" if lg.get("ppg") else "—",
                "FGA":  f"{lg.get('fga', 0):.1f}" if lg.get("fga") else "—",
                "FG%":  f"{lg['fg_pct']:.1%}"    if lg.get("fg_pct")    else "—",
                "3PA":  f"{lg.get('three_pa', 0):.1f}" if lg.get("three_pa") else "—",
                "3P%":  f"{lg['three_pct']:.1%}" if lg.get("three_pct") else "—",
                "FT%":  f"{lg['ft_pct']:.1%}"    if lg.get("ft_pct")    else "—",
                "RPG":  f"{lg.get('rpg', 0):.1f}" if lg.get("rpg") else "—",
                "APG":  f"{lg.get('apg', 0):.1f}" if lg.get("apg") else "—",
                "TOV":  f"{lg.get('tov', 0):.1f}" if lg.get("tov") else "—",
            })
        st.dataframe(pd.DataFrame(rows).set_index("League"), width="stretch")

    st.caption(
        f"eFG% = (FGM + 0.5×3PM) / FGA  ·  "
        f"xPTS / PAE not applicable — {league} defensive baseline differs from NCAA."
    )


# ── Page 2: Player Profile ────────────────────────────────────────────────────

def page_player_profile(shots: pd.DataFrame, summary: pd.DataFrame,
                        all_ncaa_summary: pd.DataFrame, bios: dict,
                        intl_stats: dict,
                        box_scores: pd.DataFrame | None = None) -> None:
    if summary.empty:
        st.warning("No data available.")
        return

    sort_col = "shrunk_pae_per100" if "shrunk_pae_per100" in summary.columns else "pts_above_exp"
    ordered  = summary.sort_values(sort_col, ascending=False)["player_name"].tolist()

    # Add international players not in NCAA summary
    intl_only = [n for n in intl_stats if n not in ordered]
    ordered   = ordered + intl_only

    with st.sidebar:
        st.markdown("---")
        st.markdown("**Select Prospect**")
        player = st.selectbox("Prospect", ordered, label_visibility="collapsed", key="profile_sel")

    # ── International player branch ───────────────────────────────────────────
    if player in intl_stats:
        _page_intl_profile(player, intl_stats[player], bios)
        return

    pdata  = shots[shots["player_name"] == player].copy()
    row_df = summary[summary["player_name"] == player]
    if pdata.empty or row_df.empty:
        st.warning("No shot data for this player.")
        return
    row  = row_df.iloc[0]
    ncaa = shots

    render_bio_card(player, bios, row)

    # ── Section 1: Core Metrics ───────────────────────────────────────────────
    section_header("1 · Core Metrics")

    threes    = pdata[pdata["shot_type"] == "3PT Field Goal"]
    twos      = pdata[pdata["shot_type"] != "3PT Field Goal"]
    three_pct = threes["shot_made"].mean() if len(threes) > 0 else 0.0
    two_pct   = twos["shot_made"].mean()   if len(twos)   > 0 else 0.0

    ft_pct_val = "N/A"
    ft_pct_sub = "Not in shot-chart API"
    if box_scores is not None and not box_scores.empty:
        brow = box_scores[box_scores["player_name"] == player]
        if not brow.empty:
            _b = brow.iloc[0]
            _ft = _b.get("ft_pct")
            _fta = _b.get("ft_attempted")
            if _ft is not None:
                ft_pct_val = f"{_ft:.1%}"
                ft_pct_sub = f"{int(_fta)} FTA" if _fta else "Box score"

    cols = st.columns(6)
    metric_card(cols[0], "FGA",  f"{int(row.total_shots):,}", "Total attempts")
    metric_card(cols[1], "FG%",  f"{row.fg_pct:.1%}",        "Overall")
    metric_card(cols[2], "2P%",  f"{two_pct:.1%}",           f"{len(twos)} att")
    metric_card(cols[3], "3P%",  f"{three_pct:.1%}",         f"{len(threes)} att")
    metric_card(cols[4], "3PAr", f"{row.pct_3pt:.1%}",       "3PT attempt rate")
    metric_card(cols[5], "FT%",  ft_pct_val,                 ft_pct_sub)
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Section 2: Shot Profile ───────────────────────────────────────────────
    section_header("2 · Shot Profile")

    col_l, col_m, col_r = st.columns([1.3, 1, 1.3])

    with col_l:
        fig = _zone_court_fig(pdata, player, ncaa,
                              title_color="#f0b429", show_cbar=True)
        st.plotly_chart(fig, use_container_width=True)

    with col_m:
        type_counts = pdata["action_type"].value_counts()
        fig, ax = plt.subplots(figsize=(4, 4), facecolor="white")
        ax.set_facecolor("white")
        clrs = ["#f0b429", "#4a90d9", "#e05c5c", "#50c878", "#9b59b6", "#e67e22"]
        wedges, texts, autos = ax.pie(
            type_counts.values, labels=type_counts.index,
            autopct="%1.0f%%", colors=clrs[:len(type_counts)],
            startangle=90, pctdistance=0.75,
            wedgeprops=dict(linewidth=2, edgecolor="white"))
        for t in texts + autos:
            t.set_color("#333333"); t.set_fontsize(8)
        ax.set_title("Shot Type Mix", color="#1a1a2e", fontsize=11, pad=8)
        fig.patch.set_facecolor("white")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col_r:
        fig, ax = plt.subplots(figsize=(5.5, 4), facecolor="white")
        ax.set_facecolor("#f5f7fa")
        bins = np.linspace(0, 35, 20)
        ax.hist(pdata["shot_distance"], bins=bins, density=True,
                alpha=0.85, color="#f0b429", label=player)
        ax.hist(ncaa["shot_distance"],  bins=bins, density=True,
                alpha=0.45, color="#4a90d9", label="NCAA Avg")
        ax.set_xlabel("Distance (ft)", color="#333333", fontsize=9)
        ax.tick_params(colors="#333333")
        ax.spines[:].set_color("#dee2e6")
        ax.legend(facecolor="white", labelcolor="#333333", fontsize=8)
        ax.set_title("Distance Distribution", color="#1a1a2e", fontsize=11, pad=8)
        fig.patch.set_facecolor("white")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ── Section 3: Shot Difficulty ────────────────────────────────────────────
    section_header("3 · Shot Difficulty")
    st.caption("Difficulty proxy: average xFG% of a player's attempts "
               "(lower = harder shot diet). Contested % unavailable from the API.")

    has_pmake = "p_make" in pdata.columns and pdata["p_make"].notna().any()

    if has_pmake:
        player_diff = pdata["p_make"].mean()
        ncaa_diff   = ncaa["p_make"].mean()
        diff_delta  = player_diff - ncaa_diff
        sign        = "easier" if diff_delta > 0 else "harder"

        c1, c2, c3, c4 = st.columns(4)
        metric_card(c1, "Avg xFG%",      f"{player_diff:.1%}", "Player shot difficulty")
        metric_card(c2, "NCAA Avg xFG%", f"{ncaa_diff:.1%}",   "All prospect average")
        metric_card(c3, "vs NCAA Avg",   f"{diff_delta:+.1%}", f"{sign} shot diet")
        metric_card(c4, "Avg Dist",      f"{row.avg_shot_dist:.1f} ft", "Mean shot distance")
        st.markdown("<br>", unsafe_allow_html=True)

        col_l2, col_r2 = st.columns(2)
        with col_l2:
            pz = pdata.groupby("shot_zone_basic")["p_make"].mean().rename("player_xfg")
            nz = ncaa.groupby("shot_zone_basic")["p_make"].mean().rename("ncaa_xfg")
            zd = pd.concat([pz, nz], axis=1).dropna()
            zd["delta"] = zd["player_xfg"] - zd["ncaa_xfg"]

            fig, ax = plt.subplots(figsize=(6, 4), facecolor="white")
            ax.set_facecolor("#f5f7fa")
            bar_colors = ["#dc3545" if v < 0 else "#2d9a4f" for v in zd["delta"]]
            ax.barh(zd.index, zd["delta"], color=bar_colors)
            ax.axvline(0, color="#aaa", linewidth=1)
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.0%}"))
            ax.tick_params(colors="#333333")
            ax.set_yticklabels(zd.index, color="#333333", fontsize=8)
            ax.spines[:].set_color("#dee2e6")
            ax.set_title("xFG% vs NCAA Avg by Zone\n(red = harder than avg)",
                         color="#1a1a2e", fontsize=10, pad=8)
            fig.patch.set_facecolor("white")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with col_r2:
            fig, ax = plt.subplots(figsize=(6, 4), facecolor="white")
            ax.set_facecolor("#f5f7fa")
            bins = np.linspace(0, 1, 25)
            ax.hist(pdata["p_make"], bins=bins, density=True,
                    alpha=0.85, color="#f0b429", label=player)
            ax.hist(ncaa["p_make"],  bins=bins, density=True,
                    alpha=0.45, color="#4a90d9", label="NCAA Avg")
            ax.axvline(player_diff, color="#f0b429", linestyle="--", linewidth=1.5)
            ax.axvline(ncaa_diff,   color="#4a90d9", linestyle="--", linewidth=1.5)
            ax.set_xlabel("xFG% per shot", color="#333333", fontsize=9)
            ax.tick_params(colors="#333333")
            ax.spines[:].set_color("#dee2e6")
            ax.legend(facecolor="white", labelcolor="#333333", fontsize=8)
            ax.set_title("Shot Difficulty Distribution", color="#1a1a2e", fontsize=11, pad=8)
            fig.patch.set_facecolor("white")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
    else:
        st.info("Run `python models/xpts_model.py` to populate shot difficulty data.")

    # ── Section 4: Shot Quality / xPTS ───────────────────────────────────────
    section_header("4 · Shot Quality & xPTS")

    c1, c2, c3 = st.columns(3)
    metric_card(c1, "Raw PAE", f"{row.pts_above_exp:+.1f}", "Total pts above expectation")
    if "shrunk_pae_per100" in row.index and pd.notna(row.shrunk_pae_per100):
        metric_card(c2, "Shrunk PAE/100", f"{row.shrunk_pae_per100:+.1f}",
                    "Partial pooling stabilised")
        lam = row.shrinkage_factor if "shrinkage_factor" in row.index else float("nan")
        metric_card(c3, "λ (data trust)", f"{lam:.2f}" if pd.notna(lam) else "—",
                    "1.0 = full trust in raw data")
    else:
        metric_card(c2, "xPTS", f"{row.expected_pts:.1f}", "Model expected points")
        metric_card(c3, "FGA",  f"{int(row.total_shots):,}", "Sample size")
    st.markdown("<br>", unsafe_allow_html=True)

    if has_pmake:
        col_l3, col_m3, col_r3 = st.columns(3)

        with col_l3:
            zq = pdata.groupby("shot_zone_basic").agg(
                actual_fg=("shot_made", "mean"),
                xfg=("p_make", "mean"),
                fga=("shot_attempted", "sum"),
            ).reset_index()
            zq = zq[zq["fga"] >= 5].sort_values("fga", ascending=False)

            fig, ax = plt.subplots(figsize=(6, 4), facecolor="white")
            ax.set_facecolor("#f5f7fa")
            x = np.arange(len(zq))
            ax.bar(x - 0.175, zq["actual_fg"], 0.35, color="#f0b429", label="Actual FG%")
            ax.bar(x + 0.175, zq["xfg"],       0.35, color="#4a90d9", label="xFG%", alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(zq["shot_zone_basic"],
                               rotation=30, ha="right", color="#333333", fontsize=8)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
            ax.tick_params(colors="#333333")
            ax.spines[:].set_color("#dee2e6")
            ax.legend(facecolor="white", labelcolor="#333333", fontsize=8)
            ax.set_title("Actual FG% vs xFG% by Zone", color="#1a1a2e", fontsize=11, pad=8)
            fig.patch.set_facecolor("white")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with col_m3:
            zq2 = pdata.groupby("shot_zone_basic").agg(
                actual_fg=("shot_made", "mean"),
                xfg=("p_make", "mean"),
                fga=("shot_attempted", "sum"),
            ).reset_index()
            zq2 = zq2[zq2["fga"] >= 5].copy()
            zq2["delta"] = zq2["actual_fg"] - zq2["xfg"]
            zq2 = zq2.sort_values("delta", ascending=True)

            fig, ax = plt.subplots(figsize=(6, 4), facecolor="white")
            ax.set_facecolor("#f5f7fa")
            bar_colors = ["#dc3545" if v < 0 else "#2d9a4f" for v in zq2["delta"]]
            bars = ax.barh(zq2["shot_zone_basic"], zq2["delta"], color=bar_colors)
            ax.axvline(0, color="#aaa", linewidth=1.2)
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.0%}"))
            ax.tick_params(colors="#333333")
            ax.set_yticklabels(zq2["shot_zone_basic"], color="#333333", fontsize=8)
            ax.spines[:].set_color("#dee2e6")
            ax.set_title("Actual − xFG% by Zone\n(green = outperforming model)",
                         color="#1a1a2e", fontsize=10, pad=8)
            for bar, val in zip(bars, zq2["delta"]):
                ax.text(bar.get_width() + (0.002 if val >= 0 else -0.002),
                        bar.get_y() + bar.get_height() / 2,
                        f"{val:+.1%}", va="center",
                        ha="left" if val >= 0 else "right",
                        color="#333333", fontsize=7)
            fig.patch.set_facecolor("white")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with col_r3:
            fig, ax = plt.subplots(figsize=(6, 6), facecolor="white")
            ax.set_facecolor("white")
            draw_court(ax, color="#aaaaaa")
            hb = ax.hexbin(pdata["loc_x"], pdata["loc_y"],
                           C=pdata["p_make"],
                           reduce_C_function=np.mean,
                           gridsize=20, cmap="RdYlGn",
                           vmin=0.3, vmax=0.65, mincnt=2,
                           extent=(-250, 250, -47.5, 470))
            plt.colorbar(hb, ax=ax, label="xFG%")
            ax.set_title("Shot Difficulty Map (xFG%)", color="#1a1a2e", fontsize=11, pad=8)
            fig.patch.set_facecolor("white")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    # ── Section 5: Historical NCAA Player Comps ───────────────────────────────
    section_header("5 · Historical NCAA Player Comps")
    st.caption(
        "Most similar players from the past 3 NCAA seasons (2022-23 to 2024-25) "
        "by shooting profile — FG%, 3PT%, avg shot distance, Shrunk PAE/100. "
        "Cosine similarity on standardised features."
    )

    comps = compute_comps(player, all_ncaa_summary, n=5)
    if comps.empty:
        st.info("Not enough data to compute comps.")
    else:
        for i, comp_row in comps.iterrows():
            sim_pct = comp_row["similarity"] * 100
            fg      = f"{comp_row['fg_pct']:.1%}" if pd.notna(comp_row.get("fg_pct")) else "—"
            pae     = (f"{comp_row['shrunk_pae_per100']:+.1f}"
                       if pd.notna(comp_row.get("shrunk_pae_per100")) else "—")
            dist    = (f"{comp_row['avg_shot_dist']:.1f} ft"
                       if pd.notna(comp_row.get("avg_shot_dist")) else "—")
            three   = f"{comp_row['pct_3pt']:.1%}" if pd.notna(comp_row.get("pct_3pt")) else "—"

            st.markdown(f"""
            <div class="comp-card">
              <div class="comp-rank">#{i+1}</div>
              <div>
                <div class="comp-name">{comp_row['player_name']}</div>
                <div class="comp-team">{comp_row['season']} · FG% {fg} · 3PT% {three} · Dist {dist} · PAE/100 {pae}</div>
              </div>
              <div class="comp-sim" style="margin-left:auto">{sim_pct:.0f}% match</div>
            </div>
            """, unsafe_allow_html=True)


# ── Page 3: Shooting Report ───────────────────────────────────────────────────

ZONE_ORDER = [
    "Restricted Area", "In The Paint (Non-RA)", "Mid-Range",
    "Left Corner 3", "Right Corner 3", "Above the Break 3",
]


@st.cache_data(ttl=86400)
def compute_all_report_scores(shots: pd.DataFrame,
                              summary: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorised 5-dim scoring (0–100 percentile within 2025-26 class).

    Dimensions
    ----------
    making      Shrunk PAE/100 percentile
    range       3PAr × 3P% × 1.5  (outside threat composite)
    at_rim      Restricted Area FG%
    shot_diet   avg xFG% of shots taken, INVERTED (lower = harder = better)
    consistency pae_shot std dev, INVERTED (lower = more consistent = better)
    """
    shots_cur = shots[shots["season"] == TARGET_SEASON].copy()
    min_fga   = 20

    grp   = shots_cur.groupby("player_name")
    total = grp["shot_attempted"].sum().rename("fga")

    # Making: from pre-computed summary
    making_raw = (summary.set_index("player_name")["shrunk_pae_per100"]
                  .rename("making_raw"))

    # Range
    threes     = shots_cur[shots_cur["shot_type"] == "3PT Field Goal"].groupby("player_name")
    three_att  = threes["shot_attempted"].sum().rename("three_att")
    three_made = threes["shot_made"].sum().rename("three_made")
    three_pct  = (three_made / three_att.clip(lower=1)).rename("three_pct")
    three_par  = (three_att / total.clip(lower=1)).rename("three_par")
    range_raw  = (three_par * three_pct * 1.5).rename("range_raw")

    # At-rim
    rim      = shots_cur[shots_cur["shot_zone_basic"] == "Restricted Area"].groupby("player_name")
    rim_att  = rim["shot_attempted"].sum().rename("rim_att")
    rim_made = rim["shot_made"].sum().rename("rim_made")
    rim_fg   = (rim_made / rim_att.clip(lower=1)).rename("rim_fg")

    # Shot diet
    has_pm = "p_make" in shots_cur.columns and shots_cur["p_make"].notna().any()
    diet_raw = grp["p_make"].mean().rename("diet_raw") if has_pm else pd.Series(dtype=float, name="diet_raw")

    # Consistency (PAE std dev)
    if "pae_shot" in shots_cur.columns and shots_cur["pae_shot"].notna().any():
        cons_raw = grp["pae_shot"].std().rename("cons_raw")
    elif has_pm:
        pv = shots_cur["shot_type"].map({"3PT Field Goal": 3}).fillna(2)
        shots_cur["_pae"] = shots_cur["shot_made"].astype(int) * pv - shots_cur["p_make"] * pv
        cons_raw = shots_cur.groupby("player_name")["_pae"].std().rename("cons_raw")
    else:
        cons_raw = pd.Series(dtype=float, name="cons_raw")

    df = pd.concat([total, making_raw, range_raw, rim_fg, rim_att,
                    three_att, diet_raw, cons_raw], axis=1).reindex(total.index)

    # Mask insufficient samples
    df.loc[df["fga"] < min_fga, ["making_raw", "range_raw", "diet_raw", "cons_raw"]] = np.nan
    if "rim_att" in df.columns:
        df.loc[df["rim_att"].fillna(0) < 5, "rim_fg"] = np.nan

    def pct_rank(s, ascending=True):
        return s.rank(pct=True, ascending=ascending, na_option="keep") * 100

    df["making"]      = pct_rank(df["making_raw"])
    df["range"]       = pct_rank(df["range_raw"])
    df["at_rim"]      = pct_rank(df["rim_fg"])
    df["shot_diet"]   = pct_rank(df["diet_raw"],  ascending=False)
    df["consistency"] = pct_rank(df["cons_raw"],  ascending=False)

    return df.reset_index()


def _grade(pct) -> tuple[str, str]:
    if pct is None or (isinstance(pct, float) and np.isnan(pct)):
        return "—", "#aaa"
    if pct >= 85: return "A+", "#2d9a4f"
    if pct >= 75: return "A",  "#2d9a4f"
    if pct >= 65: return "B+", "#6ab04c"
    if pct >= 55: return "B",  "#6ab04c"
    if pct >= 40: return "C+", "#f0b429"
    if pct >= 30: return "C",  "#f0b429"
    if pct >= 20: return "D",  "#e05c5c"
    return "F", "#dc3545"


def _radar_fig(scores: dict, player_name: str, figsize=(5, 5)) -> go.Figure:
    short = ["Making", "Range", "At-Rim", "Shot Diet", "Consistency"]
    full  = [
        "Shot Making (Shrunk PAE/100 %ile)",
        "Outside Range (3PAr × 3P% composite)",
        "At-Rim Finishing (Restricted Area FG%)",
        "Shot Diet (Difficulty, inv. %ile)",
        "Consistency (PAE variance, inv. %ile)",
    ]
    keys = ["making", "range", "at_rim", "shot_diet", "consistency"]
    vals = [float(scores.get(k) or 50) for k in keys]
    theta = short + [short[0]]

    hover = [f"<b>{d}</b><br>Percentile: <b>{v:.0f}th</b>"
             for d, v in zip(full, vals)] + [""]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=[50]*6, theta=theta, mode="lines",
        line=dict(color="#4a90d9", width=1.2, dash="dash"),
        name="Class Avg", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatterpolar(
        r=vals + [vals[0]], theta=theta,
        fill="toself", fillcolor="rgba(240,180,41,0.20)",
        line=dict(color="#f0b429", width=2.5),
        marker=dict(size=8, color="#f0b429"),
        name=player_name.split()[0],
        hovertext=hover, hoverinfo="text",
    ))
    _offset = 10
    fig.add_trace(go.Scatterpolar(
        r=[v + _offset for v in vals] + [vals[0] + _offset], theta=theta,
        mode="text",
        text=[f"<b>{v:.0f}</b>" for v in vals] + [""],
        textfont=dict(size=10, color="#c8921a"),
        showlegend=False, hoverinfo="skip",
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(range=[0, 115], tickvals=[25, 50, 75, 100],
                            tickfont=dict(size=9, color="#999"), gridcolor="#dee2e6"),
            angularaxis=dict(tickfont=dict(size=10, color="#1a1a2e",
                                           family="Arial Black"), gridcolor="#dee2e6"),
            bgcolor="#f8f9fa",
        ),
        showlegend=True,
        title=dict(text=player_name, font=dict(size=13, color="#1a1a2e"), x=0.5),
        height=420, margin=dict(t=50, b=30, l=60, r=60),
        paper_bgcolor="white",
        hoverlabel=dict(bgcolor="white", bordercolor="#ddd", font=dict(size=12)),
    )
    return fig


def page_shooting_report(shots: pd.DataFrame, summary: pd.DataFrame,
                         bios: dict,
                         box_scores: pd.DataFrame | None = None) -> None:
    st.markdown("## 📄 Player Shooting Report")

    if summary.empty:
        st.warning("No player data available.")
        return

    sort_col = "shrunk_pae_per100" if "shrunk_pae_per100" in summary.columns else "pts_above_exp"
    ordered  = summary.sort_values(sort_col, ascending=False)["player_name"].tolist()

    with st.sidebar:
        st.markdown("---")
        st.markdown("**Select Prospect**")
        player = st.selectbox("Prospect", ordered,
                              label_visibility="collapsed", key="report_sel")

    pdata  = shots[(shots["player_name"] == player) &
                   (shots["season"] == TARGET_SEASON)].copy()
    row_df = summary[summary["player_name"] == player]
    if row_df.empty:
        st.warning("No data for this player.")
        return
    row = row_df.iloc[0]

    render_bio_card(player, bios, row)

    # ── Compute scores ────────────────────────────────────────────────────────
    all_scores = compute_all_report_scores(shots, summary)
    score_row  = all_scores[all_scores["player_name"] == player]
    if score_row.empty:
        st.info("Insufficient data to generate shooting profile.")
        return
    s = score_row.iloc[0]

    # ── Grades strip ─────────────────────────────────────────────────────────
    section_header("Shooting Grades")
    dim_labels = ["Shot Making",  "Outside Range",  "At-Rim Finish", "Shot Diet",          "Consistency"]
    dim_keys   = ["making",       "range",          "at_rim",        "shot_diet",          "consistency"]
    dim_descs  = ["Shrunk PAE/100", "3PAr × 3P%",   "RA FG%",        "Hardness of diet",   "PAE variance"]

    gcols = st.columns(5)
    for col, lbl, key, desc in zip(gcols, dim_labels, dim_keys, dim_descs):
        pct   = s.get(key)
        grade, color = _grade(pct)
        pct_s = f"{pct:.0f}th" if (pct is not None and not np.isnan(float(pct))) else "—"
        with col:
            st.markdown(f"""
            <div class="metric-card" style="border-top:3px solid {color};">
              <div class="metric-label">{lbl}</div>
              <div class="metric-value" style="color:{color};font-size:30px;">{grade}</div>
              <div class="metric-sub">{pct_s} percentile</div>
              <div style="color:#aaa;font-size:10px;margin-top:4px;">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Radar + quick stats ───────────────────────────────────────────────────
    section_header("Shooting Profile")
    col_r, col_m = st.columns([1, 1])

    with col_r:
        fig = _radar_fig({k: s.get(k) for k in dim_keys}, player)
        st.plotly_chart(fig, use_container_width=True)

    with col_m:
        has_pm = "p_make" in pdata.columns and pdata["p_make"].notna().any()
        threes = pdata[pdata["shot_type"] == "3PT Field Goal"]
        rim    = pdata[pdata["shot_zone_basic"] == "Restricted Area"]

        three_pct_val = f"{threes['shot_made'].mean():.1%}" if len(threes) >= 5 else "—"
        rim_fg_val    = f"{rim['shot_made'].mean():.1%}"    if len(rim)    >= 5 else "—"
        avg_diff_val  = f"{pdata['p_make'].mean():.1%}"     if has_pm       else "—"
        pae_val       = (f"{row['shrunk_pae_per100']:+.1f}"
                         if "shrunk_pae_per100" in row.index and pd.notna(row["shrunk_pae_per100"])
                         else "—")

        ft_val = "—"
        ft_sub = "Free throw %"
        if box_scores is not None and not box_scores.empty:
            brow = box_scores[box_scores["player_name"] == player]
            if not brow.empty:
                _b = brow.iloc[0]
                _ft = _b.get("ft_pct")
                _fta = _b.get("ft_attempted")
                if _ft is not None:
                    ft_val = f"{_ft:.1%}"
                    ft_sub = f"{int(_fta)} FTA" if _fta else "Box score"

        quick = [
            ("FGA",            f"{int(row['total_shots']):,}", "Total attempts"),
            ("FG%",            f"{row['fg_pct']:.1%}",        "Overall"),
            ("3P%",            three_pct_val,                  f"{len(threes)} att"),
            ("FT%",            ft_val,                        ft_sub),
            ("3PAr",           f"{row['pct_3pt']:.1%}",       "3PT attempt rate"),
            ("At-Rim FG%",     rim_fg_val,                    f"{len(rim)} rim att"),
            ("Avg Shot Dist",  f"{row['avg_shot_dist']:.1f} ft", "Mean distance"),
            ("Avg xFG%",       avg_diff_val,                  "Shot difficulty proxy"),
            ("Shrunk PAE/100", pae_val,                       "Shot quality index"),
        ]

        st.markdown("<br>", unsafe_allow_html=True)
        for lbl, val, sub in quick:
            ca, cb = st.columns([1.6, 1])
            ca.markdown(f'<div style="color:#555;font-size:12px;">{lbl}'
                        f'<br><span style="color:#aaa;font-size:10px;">{sub}</span></div>'
                        f'<div style="height:10px"></div>',
                        unsafe_allow_html=True)
            cb.markdown(f'<div style="color:#1a1a2e;font-size:16px;'
                        f'font-weight:700;margin-top:2px;">{val}</div>'
                        f'<div style="height:10px"></div>',
                        unsafe_allow_html=True)

    # ── Shot Making & Range ───────────────────────────────────────────────────
    section_header("Shot Making  ·  Range")
    ca, cb = st.columns(2)

    with ca:
        if "shrunk_pae_per100" in summary.columns:
            all_pae = summary["shrunk_pae_per100"].dropna()
            p_pae   = row.get("shrunk_pae_per100")
            hl = [(player, float(p_pae), "#f0b429")] if pd.notna(p_pae) else []
            fig = _pae_dist_fig(all_pae, hl)
            st.plotly_chart(fig, use_container_width=True)

    with cb:
        z = (pdata.groupby("shot_zone_basic")
             .agg(att=("shot_attempted","sum"), made=("shot_made","sum"))
             .reindex(ZONE_ORDER, fill_value=0))
        z["fg_pct"] = (z["made"] / z["att"].clip(lower=1)).where(z["att"] >= 3)
        z["freq"]   = z["att"] / z["att"].sum()

        fig, ax = plt.subplots(figsize=(5, 3.5), facecolor="white")
        ax.set_facecolor("#f5f7fa")
        x = np.arange(len(ZONE_ORDER))
        ax.bar(x, z["freq"].values, color="#f0b429", alpha=0.85, width=0.6)
        for i, (fr, fg) in enumerate(zip(z["freq"].values, z["fg_pct"].values)):
            if fr > 0.01:
                ax.text(i, fr + 0.003, f"{fg:.0%}" if pd.notna(fg) else "—",
                        ha="center", va="bottom", color="#333", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels([s[:10] for s in ZONE_ORDER],
                           rotation=30, ha="right", color="#333", fontsize=7)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.tick_params(colors="#333"); ax.spines[:].set_color("#dee2e6")
        ax.set_title("Zone Frequency  (FG% above bar)", color="#1a1a2e", fontsize=10)
        fig.patch.set_facecolor("white"); plt.tight_layout()
        st.pyplot(fig); plt.close(fig)

    # ── Shot Diet & Consistency ───────────────────────────────────────────────
    section_header("Shot Diet  ·  Consistency")
    cc, cd = st.columns(2)
    has_pm = "p_make" in pdata.columns and pdata["p_make"].notna().any()

    with cc:
        if has_pm:
            zq = (pdata.groupby("shot_zone_basic")
                  .agg(actual=("shot_made","mean"), xfg=("p_make","mean"),
                       fga=("shot_attempted","sum"))
                  .reset_index())
            zq = zq[zq["fga"] >= 5].copy()
            zq["delta"] = zq["actual"] - zq["xfg"]
            zq = zq.sort_values("delta")

            fig, ax = plt.subplots(figsize=(5, 3.5), facecolor="white")
            ax.set_facecolor("#f5f7fa")
            colors = ["#dc3545" if v < 0 else "#2d9a4f" for v in zq["delta"]]
            bars   = ax.barh(zq["shot_zone_basic"], zq["delta"], color=colors)
            ax.axvline(0, color="#aaa", lw=1)
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.0%}"))
            ax.tick_params(colors="#333")
            ax.set_yticklabels(zq["shot_zone_basic"], color="#333", fontsize=8)
            ax.spines[:].set_color("#dee2e6")
            for bar, val in zip(bars, zq["delta"]):
                ax.text(bar.get_width() + (0.003 if val >= 0 else -0.003),
                        bar.get_y() + bar.get_height() / 2,
                        f"{val:+.1%}", va="center",
                        ha="left" if val >= 0 else "right",
                        color="#333", fontsize=7)
            ax.set_title("Actual − xFG% by Zone  (green = outperforming)",
                         color="#1a1a2e", fontsize=10)
            fig.patch.set_facecolor("white"); plt.tight_layout()
            st.pyplot(fig); plt.close(fig)
        else:
            st.info("Run `python models/xpts_model.py` to populate shot difficulty data.")

    with cd:
        if has_pm:
            pv  = pdata["shot_type"].map({"3PT Field Goal": 3}).fillna(2)
            pae_shots = pdata["shot_made"].astype(int) * pv - pdata["p_make"] * pv

            shots_cur = shots[shots["season"] == TARGET_SEASON]
            pv2 = shots_cur["shot_type"].map({"3PT Field Goal": 3}).fillna(2)
            ncaa_pae = (shots_cur["shot_made"].astype(int) * pv2
                        - shots_cur["p_make"] * pv2).dropna()

            fig, ax = plt.subplots(figsize=(5, 3.5), facecolor="white")
            ax.set_facecolor("#f5f7fa")
            bins = np.linspace(-3, 3, 30)
            ax.hist(ncaa_pae.clip(-3, 3), bins=bins, density=True,
                    alpha=0.4, color="#4a90d9", label="Class")
            ax.hist(pae_shots.clip(-3, 3), bins=bins, density=True,
                    alpha=0.75, color="#f0b429", label=player.split()[0])
            ax.axvline(0, color="#aaa", lw=1)
            ax.axvline(pae_shots.mean(), color="#f0b429", lw=2, ls="--")
            ax.set_xlabel("PAE per shot", color="#333", fontsize=9)
            ax.tick_params(colors="#333"); ax.spines[:].set_color("#dee2e6")
            ax.legend(facecolor="white", fontsize=8)
            ax.set_title("PAE Distribution  (Consistency)", color="#1a1a2e", fontsize=10)
            fig.patch.set_facecolor("white"); plt.tight_layout()
            st.pyplot(fig); plt.close(fig)
        else:
            st.info("Run `python models/xpts_model.py` for consistency data.")



# ── New Page: Draft Board ─────────────────────────────────────────────────────

def page_draft_board(shots: pd.DataFrame, summary: pd.DataFrame,
                     bios: dict, all_scores: pd.DataFrame,
                     box_scores: pd.DataFrame | None = None,
                     combine: dict | None = None) -> None:
    st.markdown("## 📋 2026 NBA Draft Board")

    if summary.empty:
        st.warning("No player data available.")
        return

    sort_col = "shrunk_pae_per100" if "shrunk_pae_per100" in summary.columns else "pts_above_exp"

    # ── Controls row ──────────────────────────────────────────────────────────
    ctrl1, ctrl2, ctrl3 = st.columns([2, 2, 2])

    with ctrl1:
        sort_options = {
            "Draft Order": "__rank__",
            "PAE/100 (Shot Quality)": sort_col,
            "FG%": "fg_pct",
            "3PT Attempt Rate": "pct_3pt",
            "Avg Shot Distance": "avg_shot_dist",
            "Total FGA": "total_shots",
        }
        sort_label = st.selectbox("Sort by", list(sort_options.keys()), key="db_sort")
        sort_by = sort_options[sort_label]

    with ctrl2:
        all_positions = sorted({
            PROSPECT_META.get(n, {}).get("position", "")
            for n in summary["player_name"].tolist()
            if PROSPECT_META.get(n, {}).get("position", "")
        })
        pos_filter = st.multiselect("Position", all_positions,
                                    default=all_positions, key="db_pos")

    with ctrl3:
        max_fga = int(summary["total_shots"].max()) if not summary.empty else 500
        min_fga = st.slider("Min FGA", min_value=10, max_value=max(max_fga // 2, 20),
                            value=20, step=5, key="db_minfga")

    # ── Build display table ───────────────────────────────────────────────────
    df = summary[summary["total_shots"] >= min_fga].copy()

    if pos_filter:
        df = df[df["player_name"].apply(
            lambda n: PROSPECT_META.get(n, {}).get("position", "") in pos_filter
        )]

    if sort_by == "__rank__":
        df["__rank__"] = df["player_name"].map(
            lambda n: PROSPECT_META.get(n, {}).get("rank", 9999)
        )
        df = df.sort_values("__rank__", ascending=True, na_position="last")
        df = df.drop(columns=["__rank__"])
    elif sort_by in df.columns:
        ascending = sort_by == "avg_shot_dist"
        df = df.sort_values(sort_by, ascending=ascending, na_position="last")
    df = df.reset_index(drop=True)

    # FT% from box scores
    ft_map: dict = {}
    if box_scores is not None and not box_scores.empty:
        for _, brow in box_scores.iterrows():
            ft_v = brow.get("ft_pct")
            if ft_v is not None:
                ft_map[brow["player_name"]] = ft_v

    # Threes FG% from shots
    shots_cur = shots[shots["season"] == TARGET_SEASON]
    three_made = (
        shots_cur[shots_cur["shot_type"] == "3PT Field Goal"]
        .groupby("player_name")
        .agg(three_made=("shot_made", "sum"), three_att=("shot_attempted", "sum"))
    )
    three_made["three_pct"] = three_made["three_made"] / three_made["three_att"].clip(lower=1)

    _GRADE_ICON = {
        "A+": "🟢", "A": "🟢", "B+": "🔵", "B": "🔵",
        "C+": "🟡", "C": "🟡", "D": "🔴", "F": "🔴", "—": "⚪",
    }

    rows_out = []
    name_order = []   # keep insertion order for row→name lookup
    for rank_i, (_, r) in enumerate(df.iterrows(), start=1):
        pname = r["player_name"]
        meta  = PROSPECT_META.get(pname, {})
        bio   = bios.get(pname, {})
        school = meta.get("team", bio.get("team", ""))
        pos    = meta.get("position", bio.get("position", ""))

        three_pct_v = (three_made.loc[pname, "three_pct"]
                       if pname in three_made.index else None)
        ft_pct_v    = ft_map.get(pname)

        score_row  = all_scores[all_scores["player_name"] == pname]
        making_pct = score_row.iloc[0].get("making") if not score_row.empty else None
        grade_lbl, _ = _grade(making_pct)
        pae_val = (float(r["shrunk_pae_per100"])
                   if "shrunk_pae_per100" in r.index and pd.notna(r.get("shrunk_pae_per100"))
                   else None)

        name_order.append(pname)
        rows_out.append({
            "Rank":    rank_i,
            "Player":  pname,
            "School":  school,
            "Pos":     pos,
            "FGA":     int(r.get("total_shots", 0)),
            "FG%":     f"{r['fg_pct']:.1%}" if pd.notna(r.get("fg_pct")) else "—",
            "3P%":     f"{three_pct_v:.1%}" if three_pct_v is not None and pd.notna(three_pct_v) else "—",
            "FT%":     f"{ft_pct_v:.1%}" if ft_pct_v is not None and pd.notna(ft_pct_v) else "—",
            "PAE/100": f"{pae_val:+.1f}" if pae_val is not None else "—",
            "Grade":   f"{_GRADE_ICON.get(grade_lbl, '⚪')} {grade_lbl}",
        })

    tbl_df = pd.DataFrame(rows_out)
    if tbl_df.empty:
        st.info("No players match the current filters.")
    else:
        event = st.dataframe(
            tbl_df.set_index("Rank"),
            column_config={
                "FGA":     st.column_config.NumberColumn("FGA",    format="%d"),
                "Player":  st.column_config.TextColumn("Player",  width="medium"),
                "School":  st.column_config.TextColumn("School",  width="medium"),
                "FG%":     st.column_config.TextColumn("FG%",     width="small"),
                "3P%":     st.column_config.TextColumn("3P%",     width="small"),
                "FT%":     st.column_config.TextColumn("FT%",     width="small"),
                "PAE/100": st.column_config.TextColumn("PAE/100", width="small"),
                "Grade":   st.column_config.TextColumn("Grade",   width="small"),
            },
            use_container_width=True,
            height=520,
            selection_mode="single-row",
            on_select="rerun",
            key="board_table",
        )

        # Row click → jump to Dossier
        selected_rows = event.selection.rows if event and event.selection else []
        if selected_rows:
            clicked_name = name_order[selected_rows[0]]
            st.session_state["dossier_player"] = clicked_name
            st.session_state["_nav_request"] = "👤 Player Dossier"
            st.rerun()

    # ── Combine Shooting Comparison ──────────────────────────────────────────
    if combine:
        st.markdown("---")
        st.markdown("### 🏋️ Combine Shooting Drills — All Prospects")

        # Build combine rows for all players in current filtered view
        combine_rows = []
        for pname in name_order:
            cdata = combine.get(pname, {})
            cs   = cdata.get("COLLEGE_CORNER_LEFT_PCT")
            od   = cdata.get("OFF_DRIB_COLLEGE_BREAK_LEFT_PCT")
            star = cdata.get("ON_MOVE_COLLEGE_PCT")
            side = cdata.get("THREE_PT_SIDE_PCT")
            ft   = cdata.get("FREETHROW_PCT")

            # Game 3PT% for this player
            g3 = (three_made.loc[pname, "three_pct"]
                  if pname in three_made.index else None)
            # gap: combine C&S vs game 3PT (positive = combine better)
            gap = round((cs - g3) * 100, 1) if (cs is not None and g3 is not None) else None

            combine_rows.append({
                "Player":          pname,
                "C&S%":            f"{cs:.0%}"   if cs   is not None else "—",
                "Off-Drib%":       f"{od:.0%}"   if od   is not None else "—",
                "3PT Star%":       f"{star:.0%}" if star is not None else "—",
                "3PT Side%":       f"{side:.0%}" if side is not None else "—",
                "FT%":             f"{ft:.0%}"   if ft   is not None else "—",
                "Game 3PT%":       f"{g3:.0%}"   if g3   is not None and pd.notna(g3) else "—",
                "C&S vs Game (pp)": f"{gap:+.1f}" if gap is not None else "—",
            })

        cdf = pd.DataFrame(combine_rows)
        st.dataframe(
            cdf.set_index("Player"),
            column_config={
                "C&S%":             st.column_config.TextColumn("C&S%",            width="small"),
                "Off-Drib%":        st.column_config.TextColumn("Off-Drib%",        width="small"),
                "3PT Star%":        st.column_config.TextColumn("3PT Star%",        width="small"),
                "3PT Side%":        st.column_config.TextColumn("3PT Side%",        width="small"),
                "FT%":              st.column_config.TextColumn("FT%",             width="small"),
                "Game 3PT%":        st.column_config.TextColumn("Game 3PT%",        width="small"),
                "C&S vs Game (pp)": st.column_config.TextColumn("C&S vs Game (pp)", width="small",
                                    help="Combine C&S% minus actual game 3PT% (percentage points). "
                                         "Large positive = combine > game → untapped potential."),
            },
            use_container_width=True,
            height=420,
        )
        st.caption(
            "**C&S%** — Combine catch-and-shoot (stationary, college 3PT).  ·  "
            "**Off-Drib%** — Off-dribble pull-up.  ·  "
            "**3PT Star%** — Star-pattern run & shoot.  ·  "
            "**3PT Side%** — Side 3PT spot-up.  ·  "
            "**C&S vs Game (pp)** — Combine C&S% minus game 3PT% in percentage points. "
            "Large positive = mechanics exist in isolation; game-speed execution is the gap."
        )

    # ── Metric glossary ───────────────────────────────────────────────────────
    st.markdown("---")
    st.caption(
        "**PAE/100** — Points Above Expected per 100 shot attempts. "
        "Compares each shot's actual outcome to the xPTS model's prediction given shot location, "
        "distance, and shot type. Positive = shooting above expectation. "
        "**Shrunk** = Bayesian regression toward the mean to reduce noise for players with fewer attempts.  ·  "
        "**FG%** — Field goal percentage (all shots).  ·  "
        "**3P%** — 3-point field goal percentage.  ·  "
        "**FT%** — Free throw percentage.  ·  "
        "**3PAr** — 3-point attempt rate: share of all FGA taken from beyond the arc.  ·  "
        "**Grade** — Shot-making letter grade derived from PAE/100 percentile rank within the prospect pool."
    )


# ── New Page: Player Dossier ──────────────────────────────────────────────────

def page_player_dossier(shots: pd.DataFrame, summary: pd.DataFrame,
                        all_ncaa_summary: pd.DataFrame, bios: dict,
                        intl_stats: dict, all_scores: pd.DataFrame,
                        box_scores: pd.DataFrame | None = None,
                        combine: dict | None = None) -> None:
    if summary.empty:
        st.warning("No data available.")
        return

    def _draft_rank(name: str) -> int:
        return PROSPECT_META.get(name, {}).get("rank", 9999)

    all_prospect_names = [p["name"] for p in ALL_PROSPECTS] if ALL_PROSPECTS else []
    ordered = sorted(all_prospect_names, key=_draft_rank)
    # Append any intl players not already in the list
    for n in intl_stats:
        if n not in ordered:
            ordered.append(n)

    default_player = st.session_state.get("dossier_player")
    if default_player in ordered:
        default_idx = ordered.index(default_player)
    else:
        default_idx = 0

    with st.sidebar:
        st.markdown("---")
        st.markdown("**Select Prospect**")
        player = st.selectbox("Prospect", ordered, index=default_idx,
                              label_visibility="collapsed", key="dossier_sel")

    # sync session state so Draft Board "View →" stays consistent
    st.session_state["dossier_player"] = player

    # ── International player branch ───────────────────────────────────────────
    if player in intl_stats:
        _page_intl_profile(player, intl_stats[player], bios)
        return

    pdata  = shots[shots["player_name"] == player].copy()
    row_df = summary[summary["player_name"] == player]
    if pdata.empty or row_df.empty:
        st.warning("No shot data for this player.")
        return
    row  = row_df.iloc[0]
    ncaa = shots

    render_bio_card(player, bios, row)

    # ── Scores ────────────────────────────────────────────────────────────────
    score_row = all_scores[all_scores["player_name"] == player]
    s = score_row.iloc[0] if not score_row.empty else None

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🎯 Shot Analysis", "🔍 Comps", "🏋️ Combine"])

    # ── Tab 1: Overview ───────────────────────────────────────────────────────
    with tab1:
        # Core metrics strip
        section_header("Core Metrics")
        threes    = pdata[pdata["shot_type"] == "3PT Field Goal"]
        twos      = pdata[pdata["shot_type"] != "3PT Field Goal"]
        three_pct = threes["shot_made"].mean() if len(threes) > 0 else 0.0
        two_pct   = twos["shot_made"].mean()   if len(twos)   > 0 else 0.0

        ft_pct_val = "N/A"
        ft_pct_sub = "Not in shot-chart API"
        if box_scores is not None and not box_scores.empty:
            brow = box_scores[box_scores["player_name"] == player]
            if not brow.empty:
                _b = brow.iloc[0]
                _ft = _b.get("ft_pct")
                _fta = _b.get("ft_attempted")
                if _ft is not None:
                    ft_pct_val = f"{_ft:.1%}"
                    ft_pct_sub = f"{int(_fta)} FTA" if _fta else "Box score"

        cols = st.columns(6)
        metric_card(cols[0], "FGA",  f"{int(row.total_shots):,}", "Total attempts")
        metric_card(cols[1], "FG%",  f"{row.fg_pct:.1%}",        "Overall")
        metric_card(cols[2], "2P%",  f"{two_pct:.1%}",           f"{len(twos)} att")
        metric_card(cols[3], "3P%",  f"{three_pct:.1%}",         f"{len(threes)} att")
        metric_card(cols[4], "3PAr", f"{row.pct_3pt:.1%}",       "3PT attempt rate")
        metric_card(cols[5], "FT%",  ft_pct_val,                 ft_pct_sub)
        st.markdown("<br>", unsafe_allow_html=True)

        # Shooting grades
        if s is not None:
            section_header("Shooting Grades")
            dim_labels = ["Shot Making",  "Outside Range",  "At-Rim Finish", "Shot Diet",          "Consistency"]
            dim_keys   = ["making",       "range",          "at_rim",        "shot_diet",          "consistency"]
            dim_descs  = ["Shrunk PAE/100", "3PAr × 3P%",   "RA FG%",        "Hardness of diet",   "PAE variance"]

            gcols = st.columns(5)
            for col, lbl, key, desc in zip(gcols, dim_labels, dim_keys, dim_descs):
                pct   = s.get(key)
                grade, color = _grade(pct)
                pct_s = f"{pct:.0f}th" if (pct is not None and not np.isnan(float(pct))) else "—"
                with col:
                    st.markdown(f"""
                    <div class="metric-card" style="border-top:3px solid {color};">
                      <div class="metric-label">{lbl}</div>
                      <div class="metric-value" style="color:{color};font-size:30px;">{grade}</div>
                      <div class="metric-sub">{pct_s} percentile</div>
                      <div style="color:#aaa;font-size:10px;margin-top:4px;">{desc}</div>
                    </div>""", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

        # Radar + quick stats
        section_header("Shooting Profile")
        col_r, col_m = st.columns([1, 1])

        dim_keys = ["making", "range", "at_rim", "shot_diet", "consistency"]
        with col_r:
            if s is not None:
                fig = _radar_fig({k: s.get(k) for k in dim_keys}, player)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Insufficient data to generate radar chart.")
            st.markdown("""
<table style="width:100%;border-collapse:collapse;font-size:11px;color:#555;margin-top:-8px;">
  <thead>
    <tr style="border-bottom:1px solid #e0e0e0;">
      <th style="text-align:left;padding:4px 6px;color:#1a1a2e;font-size:11px;">Dimension</th>
      <th style="text-align:left;padding:4px 6px;color:#1a1a2e;font-size:11px;">Metric</th>
      <th style="text-align:left;padding:4px 6px;color:#1a1a2e;font-size:11px;">Interpretation</th>
    </tr>
  </thead>
  <tbody>
    <tr><td style="padding:3px 6px;font-weight:600;">Making</td>
        <td style="padding:3px 6px;">Shrunk PAE/100</td>
        <td style="padding:3px 6px;color:#888;">Pts above model expectation per 100 shots, Bayes-shrunk by sample size</td></tr>
    <tr style="background:#f8f9fa;"><td style="padding:3px 6px;font-weight:600;">Range</td>
        <td style="padding:3px 6px;">3PAr × 3P%</td>
        <td style="padding:3px 6px;color:#888;">Perimeter volume (% of shots from 3) × 3-point efficiency composite</td></tr>
    <tr><td style="padding:3px 6px;font-weight:600;">At-Rim</td>
        <td style="padding:3px 6px;">RA FG%</td>
        <td style="padding:3px 6px;color:#888;">Restricted Area field goal % — rim finishing ability</td></tr>
    <tr style="background:#f8f9fa;"><td style="padding:3px 6px;font-weight:600;">Shot Diet</td>
        <td style="padding:3px 6px;">Avg xFG% (inv.)</td>
        <td style="padding:3px 6px;color:#888;">Average shot difficulty; inverted so taking harder shots scores higher</td></tr>
    <tr><td style="padding:3px 6px;font-weight:600;">Consistency</td>
        <td style="padding:3px 6px;">PAE std dev (inv.)</td>
        <td style="padding:3px 6px;color:#888;">Shot-to-shot variance in PAE; inverted so lower variance scores higher</td></tr>
  </tbody>
  <tfoot>
    <tr><td colspan="3" style="padding:5px 6px;color:#aaa;font-size:10px;border-top:1px solid #eee;">
      All scores are percentile ranks within the 2026 draft class (0th = worst, 100th = best)
    </td></tr>
  </tfoot>
</table>""", unsafe_allow_html=True)

        with col_m:
            has_pm = "p_make" in pdata.columns and pdata["p_make"].notna().any()
            rim    = pdata[pdata["shot_zone_basic"] == "Restricted Area"]

            three_pct_val = f"{threes['shot_made'].mean():.1%}" if len(threes) >= 5 else "—"
            rim_fg_val    = f"{rim['shot_made'].mean():.1%}"    if len(rim)    >= 5 else "—"
            avg_diff_val  = f"{pdata['p_make'].mean():.1%}"     if has_pm       else "—"
            pae_val       = (f"{row['shrunk_pae_per100']:+.1f}"
                             if "shrunk_pae_per100" in row.index and pd.notna(row["shrunk_pae_per100"])
                             else "—")

            ft_val = "—"
            ft_sub = "Free throw %"
            if box_scores is not None and not box_scores.empty:
                brow2 = box_scores[box_scores["player_name"] == player]
                if not brow2.empty:
                    _b2 = brow2.iloc[0]
                    _ft2 = _b2.get("ft_pct")
                    _fta2 = _b2.get("ft_attempted")
                    if _ft2 is not None:
                        ft_val = f"{_ft2:.1%}"
                        ft_sub = f"{int(_fta2)} FTA" if _fta2 else "Box score"

            quick = [
                ("FGA",            f"{int(row['total_shots']):,}", "Total attempts"),
                ("FG%",            f"{row['fg_pct']:.1%}",        "Overall"),
                ("3P%",            three_pct_val,                  f"{len(threes)} att"),
                ("FT%",            ft_val,                        ft_sub),
                ("3PAr",           f"{row['pct_3pt']:.1%}",       "3PT attempt rate"),
                ("At-Rim FG%",     rim_fg_val,                    f"{len(rim)} rim att"),
                ("Avg Shot Dist",  f"{row['avg_shot_dist']:.1f} ft", "Mean distance"),
                ("Avg xFG%",       avg_diff_val,                  "Shot difficulty proxy"),
                ("Shrunk PAE/100", pae_val,                       "Shot quality index"),
            ]

            st.markdown("<br>", unsafe_allow_html=True)
            for lbl, val, sub in quick:
                ca, cb = st.columns([1.6, 1])
                ca.markdown(f'<div style="color:#555;font-size:12px;">{lbl}'
                            f'<br><span style="color:#aaa;font-size:10px;">{sub}</span></div>'
                            f'<div style="height:10px"></div>',
                            unsafe_allow_html=True)
                cb.markdown(f'<div style="color:#1a1a2e;font-size:16px;'
                            f'font-weight:700;margin-top:2px;">{val}</div>'
                            f'<div style="height:10px"></div>',
                            unsafe_allow_html=True)

    # ── Tab 2: Shot Analysis ──────────────────────────────────────────────────
    with tab2:
        has_pmake = "p_make" in pdata.columns and pdata["p_make"].notna().any()

        # ── Row 1: Zone Court  |  Zone Scatter ───────────────────────────────
        row1_l, row1_r = st.columns(2)

        with row1_l:
            section_header("Zone Court")
            fig = _zone_court_fig(pdata, player, ncaa, title_color="#f0b429", show_cbar=True)
            st.plotly_chart(fig, use_container_width=True)

        with row1_r:
            section_header("Shot Quality by Zone")
            if has_pmake:
                zs = pdata.groupby("shot_zone_basic").agg(
                    actual_fg=("shot_made",    "mean"),
                    xfg=      ("p_make",       "mean"),
                    fga=      ("shot_attempted","sum"),
                ).reset_index()
                zs = zs[zs["fga"] >= 5].copy()
                zs["delta"] = zs["actual_fg"] - zs["xfg"]

                # bubble size: map FGA to pixel area
                max_fga = zs["fga"].max()
                zs["size"] = (zs["fga"] / max_fga * 55 + 12).round(1)

                # color: green above diagonal, red below
                zs["color"] = zs["delta"].apply(
                    lambda d: "#2d9a4f" if d >= 0 else "#dc3545")

                xy_min = min(zs["xfg"].min(), zs["actual_fg"].min()) - 0.03
                xy_max = max(zs["xfg"].max(), zs["actual_fg"].max()) + 0.03

                scatter_fig = go.Figure()

                # reference diagonal y = x
                scatter_fig.add_trace(go.Scatter(
                    x=[xy_min, xy_max], y=[xy_min, xy_max],
                    mode="lines",
                    line=dict(color="#cccccc", width=1.5, dash="dot"),
                    showlegend=False, hoverinfo="skip",
                ))

                # one trace per zone for distinct hover
                for _, zr in zs.iterrows():
                    delta_str = f"{zr['delta']:+.1%}"
                    label = ("above expectation" if zr["delta"] >= 0
                             else "below expectation")
                    scatter_fig.add_trace(go.Scatter(
                        x=[zr["xfg"]], y=[zr["actual_fg"]],
                        mode="markers+text",
                        marker=dict(
                            size=zr["size"], color=zr["color"],
                            opacity=0.82,
                            line=dict(color="white", width=1.5),
                        ),
                        text=[zr["shot_zone_basic"].replace(
                            "In The Paint (Non-RA)", "Paint (Non-RA)"
                        ).replace("Above the Break 3", "ATB 3")],
                        textposition="top center",
                        textfont=dict(size=9, color="#333"),
                        name=zr["shot_zone_basic"],
                        hovertemplate=(
                            f"<b>{zr['shot_zone_basic']}</b><br>"
                            f"xFG% (difficulty): {zr['xfg']:.1%}<br>"
                            f"Actual FG%: {zr['actual_fg']:.1%}<br>"
                            f"Δ vs model: <b>{delta_str}</b> {label}<br>"
                            f"FGA: {int(zr['fga'])}"
                            "<extra></extra>"
                        ),
                        showlegend=False,
                    ))

                scatter_fig.update_layout(
                    xaxis=dict(
                        title="xFG%  (shot difficulty →  harder)",
                        tickformat=".0%", gridcolor="#eeeeee",
                        range=[xy_min, xy_max],
                    ),
                    yaxis=dict(
                        title="Actual FG%",
                        tickformat=".0%", gridcolor="#eeeeee",
                        range=[xy_min, xy_max],
                    ),
                    height=400,
                    margin=dict(t=10, b=50, l=60, r=20),
                    paper_bgcolor="white", plot_bgcolor="#fafafa",
                    hoverlabel=dict(bgcolor="white", font=dict(size=12)),
                    annotations=[dict(
                        x=xy_min + 0.01, y=xy_max - 0.01,
                        text="← above expectation",
                        showarrow=False,
                        font=dict(size=9, color="#2d9a4f"),
                        xanchor="left",
                    ), dict(
                        x=xy_max - 0.01, y=xy_min + 0.01,
                        text="below expectation →",
                        showarrow=False,
                        font=dict(size=9, color="#dc3545"),
                        xanchor="right",
                    )],
                )
                st.plotly_chart(scatter_fig, use_container_width=True)
                st.caption("Bubble size = FGA volume · Diagonal = meeting model expectation exactly")
            else:
                st.info("Run `python models/xpts_model.py` to populate shot difficulty data.")

        # ── Row 2: PAE Class Distribution  |  Consistency ────────────────────
        row2_l, row2_r = st.columns(2)

        with row2_l:
            section_header("PAE/100  —  Class Context")
            if "shrunk_pae_per100" in summary.columns:
                all_pae = summary["shrunk_pae_per100"].dropna()
                p_pae   = row.get("shrunk_pae_per100")
                hl = [(player, float(p_pae), "#f0b429")] if pd.notna(p_pae) else []
                fig = _pae_dist_fig(all_pae, hl)
                st.plotly_chart(fig, use_container_width=True)

        with row2_r:
            section_header("Consistency  —  PAE per Shot")
            if has_pmake:
                from scipy.stats import gaussian_kde as _gkde2
                pv        = pdata["shot_type"].map({"3PT Field Goal": 3}).fillna(2)
                pae_shots = (pdata["shot_made"].astype(int) * pv
                             - pdata["p_make"] * pv).clip(-3, 3)

                shots_cur = shots[shots["season"] == TARGET_SEASON]
                pv2       = shots_cur["shot_type"].map({"3PT Field Goal": 3}).fillna(2)
                ncaa_pae  = ((shots_cur["shot_made"].astype(int) * pv2
                              - shots_cur["p_make"] * pv2)
                             .dropna().clip(-3, 3))

                x_range = np.linspace(-3, 3, 200)
                kde_p   = _gkde2(pae_shots,  bw_method=0.4)(x_range)
                kde_c   = _gkde2(ncaa_pae,   bw_method=0.4)(x_range)
                p_mean  = float(pae_shots.mean())

                cons_fig = go.Figure()
                cons_fig.add_trace(go.Scatter(
                    x=x_range, y=kde_c, mode="lines", name="2026 Class",
                    line=dict(color="#4a90d9", width=1.5, dash="dot"),
                    fill="tozeroy", fillcolor="rgba(74,144,217,0.08)",
                ))
                cons_fig.add_trace(go.Scatter(
                    x=x_range, y=kde_p, mode="lines",
                    name=player.split()[0],
                    line=dict(color="#f0b429", width=2.5),
                    fill="tozeroy", fillcolor="rgba(240,180,41,0.15)",
                ))
                cons_fig.add_vline(x=0, line=dict(color="#aaa", width=1, dash="dot"))
                cons_fig.add_vline(
                    x=p_mean,
                    line=dict(color="#f0b429", width=2, dash="dash"),
                    annotation_text=f"avg {p_mean:+.2f}",
                    annotation_position="top right",
                    annotation_font=dict(size=10, color="#c8921a"),
                )
                cons_fig.update_layout(
                    xaxis=dict(title="PAE per shot", gridcolor="#eeeeee",
                               tickformat="+.1f"),
                    yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                    height=300,
                    margin=dict(t=10, b=45, l=10, r=20),
                    paper_bgcolor="white", plot_bgcolor="#fafafa",
                    legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center",
                                font=dict(size=11)),
                    hoverlabel=dict(bgcolor="white", font=dict(size=11)),
                )
                st.plotly_chart(cons_fig, use_container_width=True)
                st.caption("Narrower curve = more consistent shot quality · dashed = player average")
            else:
                st.info("Run `python models/xpts_model.py` for consistency data.")

    # ── Tab 3: Comps ──────────────────────────────────────────────────────────
    with tab3:
        section_header("Historical NCAA Player Comps")
        st.caption(
            "Most similar players from the past 3 NCAA seasons (2022-23 to 2024-25) "
            "by shooting profile — FG%, 3PT%, avg shot distance, Shrunk PAE/100. "
            "Cosine similarity on standardised features."
        )

        comps = compute_comps(player, all_ncaa_summary, n=5)
        if comps.empty:
            st.info("Not enough data to compute comps.")
        else:
            for i, comp_row in comps.iterrows():
                sim_pct = comp_row["similarity"] * 100
                fg      = f"{comp_row['fg_pct']:.1%}" if pd.notna(comp_row.get("fg_pct")) else "—"
                pae     = (f"{comp_row['shrunk_pae_per100']:+.1f}"
                           if pd.notna(comp_row.get("shrunk_pae_per100")) else "—")
                dist    = (f"{comp_row['avg_shot_dist']:.1f} ft"
                           if pd.notna(comp_row.get("avg_shot_dist")) else "—")
                three   = f"{comp_row['pct_3pt']:.1%}" if pd.notna(comp_row.get("pct_3pt")) else "—"

                st.markdown(f"""
                <div class="comp-card">
                  <div class="comp-rank">#{i+1}</div>
                  <div>
                    <div class="comp-name">{comp_row['player_name']}</div>
                    <div class="comp-team">{comp_row['season']} · FG% {fg} · 3PT% {three} · Dist {dist} · PAE/100 {pae}</div>
                  </div>
                  <div class="comp-sim" style="margin-left:auto">{sim_pct:.0f}% match</div>
                </div>
                """, unsafe_allow_html=True)

    # ── Tab 4: Combine ────────────────────────────────────────────────────────
    with tab4:
        combine = combine or {}
        cdata = combine.get(player)
        if cdata is None:
            st.info("No 2026 NBA Draft Combine data for this player.")
        else:
            def _v(key, fmt="{}", fallback="—"):
                val = cdata.get(key)
                return fmt.format(val) if val is not None else fallback

            # ── Physical Measurements ─────────────────────────────────────────
            section_header("Physical Measurements")
            p1, p2, p3, p4, p5 = st.columns(5)
            metric_card(p1, "Height", _v("HEIGHT_WO_SHOES_FT_IN"), "Without shoes")
            metric_card(p2, "Wingspan", _v("WINGSPAN_FT_IN"),
                        _combine_rank(combine, "WINGSPAN", cdata.get("WINGSPAN")))
            metric_card(p3, "Standing Reach", _v("STANDING_REACH_FT_IN"),
                        _combine_rank(combine, "STANDING_REACH", cdata.get("STANDING_REACH")))
            metric_card(p4, "Weight",
                        f"{cdata['WEIGHT']:.0f} lbs" if cdata.get("WEIGHT") else "—", "lbs")
            hl = cdata.get("HAND_LENGTH")
            hw = cdata.get("HAND_WIDTH")
            metric_card(p5, "Hand Size",
                        f'{hl:.2f}" × {hw:.2f}"' if hl and hw else "—",
                        "Length × Width")

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Athleticism ───────────────────────────────────────────────────
            section_header("Athleticism")
            a1, a2, a3, a4, a5 = st.columns(5)
            metric_card(a1, "Standing Vert",
                        f'{cdata["STANDING_VERTICAL_LEAP"]:.1f}"' if cdata.get("STANDING_VERTICAL_LEAP") else "—",
                        _combine_rank(combine, "STANDING_VERTICAL_LEAP", cdata.get("STANDING_VERTICAL_LEAP")))
            metric_card(a2, "Max Vertical",
                        f'{cdata["MAX_VERTICAL_LEAP"]:.1f}"' if cdata.get("MAX_VERTICAL_LEAP") else "—",
                        _combine_rank(combine, "MAX_VERTICAL_LEAP", cdata.get("MAX_VERTICAL_LEAP")))
            metric_card(a3, "Lane Agility",
                        f'{cdata["LANE_AGILITY_TIME"]:.2f}s' if cdata.get("LANE_AGILITY_TIME") else "—",
                        _combine_rank(combine, "LANE_AGILITY_TIME", cdata.get("LANE_AGILITY_TIME"), lower_is_better=True))
            metric_card(a4, "Mod. Agility",
                        f'{cdata["MODIFIED_LANE_AGILITY_TIME"]:.2f}s' if cdata.get("MODIFIED_LANE_AGILITY_TIME") else "—",
                        _combine_rank(combine, "MODIFIED_LANE_AGILITY_TIME", cdata.get("MODIFIED_LANE_AGILITY_TIME"), lower_is_better=True))
            metric_card(a5, "¾ Sprint",
                        f'{cdata["THREE_QUARTER_SPRINT"]:.2f}s' if cdata.get("THREE_QUARTER_SPRINT") else "—",
                        _combine_rank(combine, "THREE_QUARTER_SPRINT", cdata.get("THREE_QUARTER_SPRINT"), lower_is_better=True))

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Combine Shooting Drills ───────────────────────────────────────
            cs_pct   = cdata.get("COLLEGE_CORNER_LEFT_PCT")
            od_pct   = cdata.get("OFF_DRIB_COLLEGE_BREAK_LEFT_PCT")
            om_pct   = cdata.get("ON_MOVE_COLLEGE_PCT")
            side_pct = cdata.get("THREE_PT_SIDE_PCT")
            ft_pct   = cdata.get("FREETHROW_PCT")

            if any(v is not None for v in [cs_pct, od_pct, om_pct]):
                section_header("Combine Shooting Drills")
                d1, d2, d3, d4 = st.columns(4)
                metric_card(d1, "Catch & Shoot",
                            f"{cs_pct:.0%}" if cs_pct is not None else "—",
                            _combine_rank(combine, "COLLEGE_CORNER_LEFT_PCT", cs_pct))
                metric_card(d2, "Off Dribble",
                            f"{od_pct:.0%}" if od_pct is not None else "—",
                            _combine_rank(combine, "OFF_DRIB_COLLEGE_BREAK_LEFT_PCT", od_pct))
                metric_card(d3, "3PT Star (On Move)",
                            f"{om_pct:.0%}" if om_pct is not None else "—",
                            _combine_rank(combine, "ON_MOVE_COLLEGE_PCT", om_pct))

                # C&S Premium — coachability indicator
                if cs_pct is not None and od_pct is not None:
                    premium = cs_pct - od_pct
                    if premium > 0.15:
                        label = "Shot Creator (needs work)"
                    elif premium > 0.05:
                        label = "C&S Specialist"
                    else:
                        label = "Versatile Shooter"
                    metric_card(d4, "C&S Premium",
                                f"{premium:+.0%}",
                                label)
                else:
                    metric_card(d4, "C&S Premium", "—", "")

                e1, e2, e3, e4 = st.columns(4)
                metric_card(e1, "3PT Side",
                            f"{side_pct:.0%}" if side_pct is not None else "—",
                            _combine_rank(combine, "THREE_PT_SIDE_PCT", side_pct))
                metric_card(e2, "Free Throw",
                            f"{ft_pct:.0%}" if ft_pct is not None else "—",
                            _combine_rank(combine, "FREETHROW_PCT", ft_pct))

                st.caption(
                    "All shooting drills at college 3PT distance. "
                    "**C&S Premium** = Catch-&-Shoot% − Off-Dribble%: "
                    "large positive → good mechanics, shot creation needs coaching."
                )

            st.markdown("<br>", unsafe_allow_html=True)




# ── New Page: Head-to-Head ────────────────────────────────────────────────────

def page_h2h(shots: pd.DataFrame, summary: pd.DataFrame,
             bios: dict, intl_stats: dict,
             all_scores: pd.DataFrame,
             box_scores: pd.DataFrame | None = None,
             combine: dict | None = None) -> None:
    st.markdown("## ⚖️ Head-to-Head")

    def _h2h_rank(name: str) -> int:
        return PROSPECT_META.get(name, {}).get("rank", 9999)

    all_ordered = sorted(
        ([p["name"] for p in ALL_PROSPECTS] if ALL_PROSPECTS else
         list(set(summary["player_name"].tolist()) | set(intl_stats.keys()))),
        key=_h2h_rank,
    )
    for n in intl_stats:
        if n not in all_ordered:
            all_ordered.append(n)

    def pos_of(name): return PROSPECT_META.get(name, {}).get("position", "")

    # ── Position filter ───────────────────────────────────────────────────────
    st.markdown("**Position Group**")
    pos_filter = st.radio("pos", ["All", "G", "F", "C"],
                          horizontal=True, label_visibility="collapsed",
                          key="h2h_pos")

    filtered = [n for n in all_ordered
                if pos_filter == "All" or pos_of(n) == pos_filter]
    if len(filtered) < 2:
        st.warning(f"Not enough {pos_filter} players in the prospect pool.")
        return

    # ── Player selectors ──────────────────────────────────────────────────────
    st.markdown("---")
    sel_a, sel_gap, sel_b = st.columns([5, 1, 5])

    with sel_a:
        st.markdown('<p style="color:#f0b429;font-weight:700;margin:0 0 4px;">Player A</p>',
                    unsafe_allow_html=True)
        player_a = st.selectbox("A", filtered, key="h2h_a",
                                label_visibility="collapsed")

    with sel_gap:
        st.markdown(
            '<div style="text-align:center;padding-top:28px;font-size:18px;'
            'font-weight:800;color:#ccc;">VS</div>',
            unsafe_allow_html=True)

    a_pos  = pos_of(player_a)
    b_pool = [n for n in filtered if n != player_a and
              (pos_filter != "All" or pos_of(n) == a_pos or not a_pos)]
    if len(b_pool) == 0:
        b_pool = [n for n in filtered if n != player_a]

    with sel_b:
        st.markdown('<p style="color:#4a90d9;font-weight:700;margin:0 0 4px;">Player B</p>',
                    unsafe_allow_html=True)
        player_b = st.selectbox("B", b_pool, key="h2h_b",
                                label_visibility="collapsed")

    st.markdown("---")

    ma = _player_metrics(player_a, shots, summary, intl_stats, box_scores)
    mb = _player_metrics(player_b, shots, summary, intl_stats, box_scores)

    # ── Bio cards ─────────────────────────────────────────────────────────────
    _empty = type("", (), {})()
    ca, cm, cb = st.columns([10, 1, 10])
    with ca:
        row_a = ma.get("_row")
        render_bio_card(player_a, bios, row_a if row_a is not None else _empty)
    with cm:
        st.markdown(
            '<div style="text-align:center;padding-top:36px;font-size:22px;'
            'font-weight:800;color:#f0b429;">VS</div>',
            unsafe_allow_html=True)
    with cb:
        row_b = mb.get("_row")
        render_bio_card(player_b, bios, row_b if row_b is not None else _empty)

    # ── Head-to-head + radar ──────────────────────────────────────────────────
    section_header("Head-to-Head")
    col_tbl, col_radar = st.columns([1, 1])

    with col_tbl:
        _h2h_table(ma, mb, player_a, player_b)

    with col_radar:
        sa_row   = all_scores[all_scores["player_name"] == player_a]
        sb_row   = all_scores[all_scores["player_name"] == player_b]
        scores_a = sa_row.iloc[0].to_dict() if not sa_row.empty else {}
        scores_b = sb_row.iloc[0].to_dict() if not sb_row.empty else {}
        if scores_a or scores_b:
            fig = _radar_compare(scores_a, scores_b, player_a, player_b)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("""
<table style="width:100%;border-collapse:collapse;font-size:11px;color:#555;">
  <thead><tr style="border-bottom:1px solid #e0e0e0;">
    <th style="padding:3px 6px;color:#1a1a2e;">Dimension</th>
    <th style="padding:3px 6px;color:#1a1a2e;">Metric</th>
    <th style="padding:3px 6px;color:#1a1a2e;">Interpretation</th>
  </tr></thead>
  <tbody>
    <tr><td style="padding:3px 6px;font-weight:600;">Making</td><td style="padding:3px 6px;">Shrunk PAE/100</td><td style="padding:3px 6px;color:#888;">Pts above model per 100 shots, Bayes-shrunk</td></tr>
    <tr style="background:#f8f9fa;"><td style="padding:3px 6px;font-weight:600;">Range</td><td style="padding:3px 6px;">3PAr × 3P%</td><td style="padding:3px 6px;color:#888;">Perimeter volume × 3-point efficiency</td></tr>
    <tr><td style="padding:3px 6px;font-weight:600;">At-Rim</td><td style="padding:3px 6px;">RA FG%</td><td style="padding:3px 6px;color:#888;">Restricted Area FG% — rim finishing</td></tr>
    <tr style="background:#f8f9fa;"><td style="padding:3px 6px;font-weight:600;">Shot Diet</td><td style="padding:3px 6px;">Avg xFG% (inv.)</td><td style="padding:3px 6px;color:#888;">Shot difficulty; inverted — harder diet scores higher</td></tr>
    <tr><td style="padding:3px 6px;font-weight:600;">Consistency</td><td style="padding:3px 6px;">PAE std dev (inv.)</td><td style="padding:3px 6px;color:#888;">PAE variance; inverted — lower variance scores higher</td></tr>
  </tbody>
  <tfoot><tr><td colspan="3" style="padding:5px 6px;color:#aaa;font-size:10px;border-top:1px solid #eee;">
    All axes = percentile rank within 2026 class (0 = worst, 100 = best)
  </td></tr></tfoot>
</table>""", unsafe_allow_html=True)
        else:
            st.info("Radar unavailable — run xPTS model first.")

    # ── PAE class distribution ────────────────────────────────────────────────
    if "shrunk_pae_per100" in summary.columns:
        section_header("PAE/100  —  Class Context")
        all_pae = summary["shrunk_pae_per100"].dropna()
        pa_val  = ma.get("PAE/100"); pb_val = mb.get("PAE/100")
        hl = []
        if pa_val is not None: hl.append((player_a, pa_val, "#f0b429"))
        if pb_val is not None: hl.append((player_b, pb_val, "#4a90d9"))
        fig = _pae_dist_fig(all_pae, hl)
        st.plotly_chart(fig, use_container_width=True)

    # ── Combine: Physical & Athleticism ───────────────────────────────────────
    combine = combine or {}
    ca_data = combine.get(player_a)
    cb_data = combine.get(player_b)
    if ca_data or cb_data:
        section_header("NBA Draft Combine — Physical & Athleticism")

        def _cv(d, key, fmt):
            val = (d or {}).get(key)
            return fmt.format(val) if val is not None else "—"

        rows = [
            ("Height (no shoes)",  "HEIGHT_WO_SHOES_FT_IN", "{}",        False),
            ("Wingspan",           "WINGSPAN_FT_IN",         "{}",        False),
            ("Standing Reach",     "STANDING_REACH_FT_IN",   "{}",        False),
            ("Weight (lbs)",       "WEIGHT",                 "{:.0f}",    False),
            ("Standing Vertical",  "STANDING_VERTICAL_LEAP", '{:.1f}"',   False),
            ("Max Vertical",       "MAX_VERTICAL_LEAP",      '{:.1f}"',   False),
            ("Lane Agility",       "LANE_AGILITY_TIME",      "{:.2f}s",   True),
            ("¾ Sprint",           "THREE_QUARTER_SPRINT",   "{:.2f}s",   True),
            ("Catch & Shoot 3PT%",  "COLLEGE_CORNER_LEFT_PCT",         "{:.0%}", False),
            ("Off-Drib 3PT%",      "OFF_DRIB_COLLEGE_BREAK_LEFT_PCT", "{:.0%}", False),
            ("3PT Star (On Move)", "ON_MOVE_COLLEGE_PCT",             "{:.0%}", False),
            ("3PT Side",           "THREE_PT_SIDE_PCT",               "{:.0%}", False),
            ("Free Throw%",        "FREETHROW_PCT",                   "{:.0%}", False),
        ]

        html = f"""
        <table class="h2h-table">
        <thead><tr>
          <th style="color:#f0b429;padding:6px 4px;text-align:right;width:38%">{player_a.split()[-1]}</th>
          <th style="padding:6px 8px;text-align:center;color:#999;font-size:11px;width:24%">Metric</th>
          <th style="color:#4a90d9;padding:6px 4px;text-align:left;width:38%">{player_b.split()[-1]}</th>
        </tr></thead><tbody>
        """
        for label, key, fmt, lower_better in rows:
            va_raw = (ca_data or {}).get(key)
            vb_raw = (cb_data or {}).get(key)
            va_s = fmt.format(va_raw) if va_raw is not None else "—"
            vb_s = fmt.format(vb_raw) if vb_raw is not None else "—"
            if va_raw is not None and vb_raw is not None:
                a_wins = (va_raw < vb_raw) if lower_better else (va_raw > vb_raw)
                col_a = "#f0b429" if a_wins else "#555"
                col_b = "#4a90d9" if not a_wins else "#555"
            else:
                col_a = col_b = "#555"
            html += f"""<tr>
              <td style="text-align:right;font-size:15px;font-weight:700;color:{col_a};padding:5px 4px">{va_s}</td>
              <td style="text-align:center;color:#999;font-size:11px;padding:5px 8px">{label}</td>
              <td style="text-align:left;font-size:15px;font-weight:700;color:{col_b};padding:5px 4px">{vb_s}</td>
            </tr>"""
        html += "</tbody></table>"
        st.markdown(html, unsafe_allow_html=True)


# ── Page: Shot Mechanics ──────────────────────────────────────────────────────

VIDEO_DIR = Path(__file__).resolve().parent.parent / "video"

# Map display name → video folder name (only players with pose data)
MECHANICS_PLAYERS: dict[str, str | None] = {
    "Cameron Boozer":    "1_Cameron_Boozer",
    "Darryn Peterson":   "2_Darryn_Peterson",
    "AJ Dybantsa":       "3_AJ_Dybantsa",
    "Caleb Wilson":      "4_Caleb_Wilson",
    "Mikel Brown Jr.":   "5_Mikel_Brown_Jr.",
    "Kingston Flemings": "6_Kingston_Flemings",
    "Bennett Stirtz":    "10_Bennett_Stirtz",
    # < 10 pose clips — excluded until more footage reviewed
    "Hannes Steinbach":  None,
    "Koa Peat":          None,
    "Labaron Philon":    None,
}

METRICS_META = {
    "knee_angle":  {"label": "Knee Angle (°)",  "color": "#e8854c"},
    "elbow_angle": {"label": "Elbow Angle (°)", "color": "#4c9ee8"},
    "body_lean":   {"label": "Body Lean (°)",   "color": "#c84ce8"},
}

METRIC_DESCRIPTIONS = {
    "knee_angle":  "Shooting-side knee angle (hip→knee→ankle). Drops during load, extends explosively at takeoff. Most predictive of shot quality.",
    "elbow_angle": "Shooting-side elbow angle (shoulder→elbow→wrist). Shows arm cocking during load and extension toward release.",
    "body_lean":   "Torso tilt from vertical. Negative = leaning left, positive = leaning right. Ideally stays near 0° (upright) at release.",
}


def _clean_source_name(filename: str) -> str:
    """Turn raw video filename into a readable label."""
    stem = Path(filename).stem
    # YTDown format: YTDown_YouTube_<Title-With-Dashes>_Media_<id>_NNN_1080p
    if stem.startswith("YTDown_YouTube_"):
        parts = stem.split("_Media_")[0]
        title = parts.replace("YTDown_YouTube_", "").replace("-", " ")
        return title.title()
    # Remove common suffixes like _1080p, _720p, _workout
    for suf in ("_1080p", "_720p", "_workout", "_highlights"):
        stem = stem.replace(suf, "")
    return stem.replace("_", " ").strip()


@st.cache_data(show_spinner=False)
def load_pose_clips(folder: str) -> list[dict]:
    """Load all pose JSON files for a player, including source video info."""
    clips_dir = VIDEO_DIR / folder / "jump_shot_clips"
    clips = []
    for p in sorted(clips_dir.glob("*_pose.json")):
        if "summary" in p.name:
            continue
        data = json.loads(p.read_text())
        if not data.get("trajectory"):
            continue
        clip_stem = p.stem.replace("_pose", "")
        # Load source info from tracking JSON
        source_video, anchor_sec, clip_start_sec = "", 0.0, 0.0
        release_frame_off = None
        tracking_path = clips_dir / f"{clip_stem}_tracking.json"
        if tracking_path.exists():
            tj = json.loads(tracking_path.read_text())
            source_video      = tj.get("source_video", "")
            anchor_sec        = tj.get("anchor_sec", 0.0)
            fps               = tj.get("fps", 60.0)
            clip_start_sec    = tj.get("clip_start_frame", 0) / fps
            # Use manually annotated release_frame if available
            # release_frame is already a clip-frame offset (0-based within clip)
            if tj.get("release_frame") is not None:
                release_frame_off = tj["release_frame"]
        trajectory = data["trajectory"]

        # Anchor x-axis: manual annotation > action-classifier frame (rel_frame=0)
        if release_frame_off is not None:
            for r in trajectory:
                r["release_rel_frame"] = r["frame_offset"] - release_frame_off
        else:
            for r in trajectory:
                r["release_rel_frame"] = r["rel_frame"]

        valid_frames = sum(1 for f in trajectory if f.get("knee_angle") is not None)
        clips.append({
            "clip":           clip_stem,
            "trajectory":     trajectory,
            "stats":          data.get("stats", {}),
            "source_video":   source_video,
            "source_label":   _clean_source_name(source_video),
            "anchor_sec":     anchor_sec,
            "clip_start_sec": clip_start_sec,
            "valid_frames":   valid_frames,
        })
    # Most complete clips first so the default view is the best quality
    clips.sort(key=lambda c: -c["valid_frames"])
    return clips


def _trajectory_fig(clips: list[dict], metric: str, fps: float = 60.0) -> go.Figure:
    meta   = METRICS_META[metric]
    fig    = go.Figure()
    colors = meta["color"]

    # Per-clip lines (semi-transparent)
    all_x, all_y_by_x = [], {}
    for clip in clips:
        xs = [r["release_rel_frame"] / fps for r in clip["trajectory"] if r.get(metric) is not None]
        ys = [r[metric]                     for r in clip["trajectory"] if r.get(metric) is not None]
        if not xs:
            continue
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="lines",
            name=clip["clip"],
            line=dict(color=colors, width=1.5),
            opacity=0.35,
            showlegend=True,
            hovertemplate=f"{clip['clip']}<br>t=%{{x:.2f}}s  {metric}=%{{y:.1f}}<extra></extra>",
        ))
        for x, y in zip(xs, ys):
            all_y_by_x.setdefault(round(x, 3), []).append(y)
            all_x.append(x)

    # Mean line
    if all_y_by_x:
        mx = sorted(all_y_by_x)
        my = [float(np.mean(all_y_by_x[x])) for x in mx]
        fig.add_trace(go.Scatter(
            x=mx, y=my,
            mode="lines",
            name="Mean",
            line=dict(color=colors, width=3),
            opacity=1.0,
            showlegend=True,
            hovertemplate=f"Mean<br>t=%{{x:.2f}}s  {metric}=%{{y:.1f}}<extra></extra>",
        ))

    # Anchor line
    fig.add_vline(x=0, line_dash="dash", line_color="#aaaaaa", line_width=1)
    fig.add_annotation(x=0, y=1, yref="paper", text="release", showarrow=False,
                       font=dict(size=10, color="#aaaaaa"), yanchor="top")

    fig.update_layout(
        title=dict(text=meta["label"], font=dict(size=14)),
        xaxis=dict(title="Time from release (s)", zeroline=False,
                   tickformat=".1f"),
        yaxis=dict(title=meta["label"]),
        height=320,
        margin=dict(l=50, r=20, t=40, b=40),
        legend=dict(font=dict(size=10), orientation="h",
                    yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="#fafafa",
        paper_bgcolor="#ffffff",
        hovermode="x unified",
    )
    return fig


def _single_clip_fig(clip: dict, all_clips: list[dict],
                     metric: str, fps: float = 60.0) -> go.Figure:
    """One clip highlighted, all-clip mean as grey reference."""
    meta  = METRICS_META[metric]
    color = meta["color"]
    fig   = go.Figure()

    # Grey reference lines for all other clips
    for c in all_clips:
        if c["clip"] == clip["clip"]:
            continue
        xs = [r["release_rel_frame"] / fps for r in c["trajectory"] if r.get(metric) is not None]
        ys = [r[metric]                     for r in c["trajectory"] if r.get(metric) is not None]
        if not xs:
            continue
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines",
            line=dict(color="#cccccc", width=1),
            opacity=0.5, showlegend=False,
            hoverinfo="skip",
        ))

    # Mean line
    all_y_by_x: dict[float, list] = {}
    for c in all_clips:
        for r in c["trajectory"]:
            if r.get(metric) is not None:
                k = round(r["release_rel_frame"] / fps, 3)
                all_y_by_x.setdefault(k, []).append(r[metric])
    if all_y_by_x:
        mx = sorted(all_y_by_x)
        my = [float(np.mean(all_y_by_x[x])) for x in mx]
        fig.add_trace(go.Scatter(
            x=mx, y=my, mode="lines", name="All-clip mean",
            line=dict(color="#888888", width=2, dash="dot"),
            opacity=0.8, showlegend=True,
        ))

    # Selected clip — highlighted
    xs = [r["release_rel_frame"] / fps for r in clip["trajectory"] if r.get(metric) is not None]
    ys = [r[metric]                     for r in clip["trajectory"] if r.get(metric) is not None]
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines+markers", name=clip["clip"],
        line=dict(color=color, width=2.5),
        marker=dict(size=4, color=color),
        showlegend=True,
        hovertemplate=f"t=%{{x:.2f}}s<br>{metric}=%{{y:.1f}}<extra></extra>",
    ))

    fig.add_vline(x=0, line_dash="dash", line_color="#aaaaaa", line_width=1)
    fig.add_annotation(x=0, y=1, yref="paper", text="release", showarrow=False,
                       font=dict(size=10, color="#aaaaaa"), yanchor="top")

    fig.update_layout(
        title=dict(text=meta["label"], font=dict(size=13)),
        xaxis=dict(title="Time from release (s)", zeroline=False, tickformat=".1f"),
        yaxis=dict(title=meta["label"]),
        height=260,
        margin=dict(l=50, r=10, t=36, b=36),
        legend=dict(font=dict(size=10), orientation="h",
                    yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="#fafafa", paper_bgcolor="#ffffff",
        hovermode="x",
    )
    return fig


def _snapshot_metrics(clips: list[dict]) -> dict:
    """Compute scouting snapshot cards across all clips for a player."""
    load_depths, arm_setups, release_leans = [], [], []

    for c in clips:
        traj = c["trajectory"]

        knee_indexed  = [(i, r["knee_angle"])  for i, r in enumerate(traj) if r.get("knee_angle")  is not None]
        elbow_indexed = [(i, r["elbow_angle"]) for i, r in enumerate(traj) if r.get("elbow_angle") is not None]
        lean_indexed  = [(i, r["body_lean"])   for i, r in enumerate(traj) if r.get("body_lean")   is not None]

        if knee_indexed:
            min_knee = min(v for _, v in knee_indexed)
            load_depths.append(min_knee)

            # Loading window = frames within 15° of the deepest knee bend
            load_idxs = {i for i, v in knee_indexed if v <= min_knee + 15}

            # Arm setup = mean elbow angle during loading window
            load_elbow = [v for i, v in elbow_indexed if i in load_idxs]
            if load_elbow:
                arm_setups.append(float(np.mean(load_elbow)))

        # Release lean = mean body lean in ±3 frames around the re-anchored release (frame 0)
        near_release = [r["body_lean"] for r in traj
                        if r.get("body_lean") is not None
                        and abs(r.get("release_rel_frame", 9999)) <= 3]
        if near_release:
            release_leans.append(float(np.mean(near_release)))

    return {
        "load_depth_mean":   round(float(np.mean(load_depths)),   1) if load_depths   else None,
        "load_depth_std":    round(float(np.std(load_depths)),    1) if len(load_depths) > 1   else None,
        "arm_setup_mean":    round(float(np.mean(arm_setups)),    1) if arm_setups    else None,
        "release_lean_mean": round(float(np.mean(release_leans)), 1) if release_leans else None,
        "n_clips":           len(clips),
    }


def page_shot_mechanics(bios: dict) -> None:
    st.markdown("## 🎯 Shot Mechanics")

    # Only show players ranked ≤ 10 who have pose data
    mechanics_names = [
        name for name, folder in MECHANICS_PLAYERS.items()
        if folder is not None and PROSPECT_META.get(name, {}).get("rank", 9999) <= 10
    ]
    mechanics_names.sort(key=lambda n: PROSPECT_META.get(n, {}).get("rank", 9999))

    with st.sidebar:
        st.markdown("### Player")
        player_name = st.selectbox(
            "Select player", mechanics_names,
            label_visibility="collapsed",
        )

    folder = MECHANICS_PLAYERS.get(player_name)
    if folder is None:
        st.info(f"Pose data for **{player_name}** not yet processed.")
        return

    clips = load_pose_clips(folder)
    if not clips:
        st.warning(f"No pose clips found for {player_name}.")
        return

    clips_dir = VIDEO_DIR / folder / "jump_shot_clips"

    # ── Profile header + scouting snapshot ───────────────────────────────────
    snap = _snapshot_metrics(clips)
    bio  = bios.get(player_name, {})
    meta = PROSPECT_META.get(player_name, {})

    headshot = bio.get("headshot_url", "")
    position = bio.get("position") or meta.get("position", "")
    team     = meta.get("team", bio.get("team", ""))
    rank     = meta.get("rank", "")
    height   = bio.get("height", "")
    weight   = bio.get("weight", "")

    prof_col, snap_col = st.columns([1, 2])

    with prof_col:
        img_html = ""
        if headshot:
            img_html = (f'<img src="{headshot}" style="width:90px;height:90px;'
                        f'border-radius:50%;object-fit:cover;'
                        f'border:3px solid #f0b429;background:#1a1a2e;'
                        f'float:left;margin-right:14px;" '
                        f'onerror="this.style.display:\'none\'">')
        pills = []
        if rank:     pills.append(f"#{rank} Overall")
        if position: pills.append(position)
        if team:     pills.append(team)
        if height:   pills.append(height)
        if weight:   pills.append(weight)
        pills_html = " · ".join(f"<span style='color:#f0b429;font-weight:600'>{p}</span>" for p in pills[:2])
        rest_html  = " · ".join(pills[2:])
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:14px;padding:8px 0">'
            f'{img_html}'
            f'<div><div style="font-size:22px;font-weight:700;color:#1a1a2e">{player_name}</div>'
            f'<div style="margin-top:4px;font-size:13px">{pills_html}</div>'
            f'<div style="margin-top:2px;font-size:12px;color:#666">{rest_html}</div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )
        st.caption(f"Based on {snap['n_clips']} clips")

    with snap_col:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(
            "Load Depth",
            f"{snap['load_depth_mean']:.0f}°" if snap["load_depth_mean"] is not None else "—",
            help="Mean minimum knee angle across clips — lower = deeper squat, more leg drive",
        )
        m2.metric(
            "Consistency",
            f"±{snap['load_depth_std']:.1f}°" if snap["load_depth_std"] is not None else "—",
            help="Std of load depth across clips — lower = more repeatable mechanics",
        )
        m3.metric(
            "Arm Setup",
            f"{snap['arm_setup_mean']:.0f}°" if snap["arm_setup_mean"] is not None else "—",
            help="Mean elbow angle during the loading phase — lower = more compact arm set",
        )
        m4.metric(
            "Release Lean",
            f"{snap['release_lean_mean']:+.1f}°" if snap["release_lean_mean"] is not None else "—",
            help="Mean body lean at release — negative = leaning back, positive = leaning forward",
        )

    st.markdown("---")

    # ── Clip navigation ───────────────────────────────────────────────────────
    idx_key = f"mech_clip_idx_{player_name}"
    st.session_state.setdefault(idx_key, 0)
    n_clips = len(clips)

    st.markdown("""
<style>
div[data-testid="stHorizontalBlock"]:has(video) {
    gap: 0.25rem !important;
    align-items: center !important;
}
div[data-testid="stHorizontalBlock"]:has(video) > div[data-testid="column"]:first-child,
div[data-testid="stHorizontalBlock"]:has(video) > div[data-testid="column"]:last-child {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    padding: 0 !important;
}
div[data-testid="stHorizontalBlock"]:has(video) button[data-testid="baseButton-secondary"] {
    font-size: 2rem !important;
    height: 5rem !important;
    width: 100% !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: #ccc !important;
}
div[data-testid="stHorizontalBlock"]:has(video) button[data-testid="baseButton-secondary"]:hover:not(:disabled) {
    color: #fff !important;
    background: transparent !important;
}
div[data-testid="stHorizontalBlock"]:has(video) button[data-testid="baseButton-secondary"]:disabled {
    opacity: 0.2 !important;
}
</style>
""", unsafe_allow_html=True)

    selected_idx = st.session_state[idx_key]
    clip = clips[selected_idx]

    stem       = Path(clip['clip']).stem
    web_path   = clips_dir / f"{stem}_pose_preview_web.mp4"
    raw_path   = clips_dir / f"{stem}_pose_preview.mp4"
    video_path = web_path if web_path.exists() else raw_path

    # ── Video + nav buttons ───────────────────────────────────────────────────
    nav_l, vid_col, nav_r = st.columns([0.5, 5, 0.5])

    with nav_l:
        if st.button("◀", key="mech_prev", use_container_width=True,
                     disabled=(selected_idx == 0)):
            st.session_state[idx_key] = selected_idx - 1
            st.rerun()
    with vid_col:
        if video_path.exists():
            st.video(str(video_path), autoplay=True)
        else:
            st.warning("Video not found")
        st.caption(f"📁 {clip['source_label']}")
    with nav_r:
        if st.button("▶", key="mech_next", use_container_width=True,
                     disabled=(selected_idx >= n_clips - 1)):
            st.session_state[idx_key] = selected_idx + 1
            st.rerun()

    # ── Trajectory charts ─────────────────────────────────────────────────────
    st.caption(
        "**Colored line** = this clip · **dotted grey** = player mean · "
        "faint grey = other clips · dashed vertical = release point"
    )
    c1, c2, c3 = st.columns(3)
    for col, metric in [(c1, "knee_angle"), (c2, "elbow_angle"), (c3, "body_lean")]:
        fig = _single_clip_fig(clip, clips, metric)
        col.plotly_chart(fig, use_container_width=True)
        col.caption(METRIC_DESCRIPTIONS[metric])



# ── App entry point ───────────────────────────────────────────────────────────

def main() -> None:
    st.session_state.setdefault("nav_page", "📋 Draft Board")

    # Consume programmatic nav requests before any widget is created,
    # so the radio key can be updated without a conflict.
    if "_nav_request" in st.session_state:
        st.session_state["nav_page"] = st.session_state.pop("_nav_request")

    if not DB_PATH.exists():
        st.error("Database not found. Run `python pipeline/ingest.py` first.")
        st.stop()

    shots            = load_shots("NCAA")
    summary          = load_summary(season=TARGET_SEASON)
    all_ncaa_summary = load_summary(league="NCAA")
    bios             = load_bios()
    intl_stats       = load_intl_stats()
    box_scores       = load_box_scores()
    combine          = load_combine()
    all_scores       = compute_all_report_scores(shots, summary)

    with st.sidebar:
        st.markdown("# 🏀 2026 Draft")
        st.caption("Shot Quality Scouting Portal")
        st.markdown("---")
        page = st.radio("Navigate",
                        ["📋 Draft Board", "👤 Player Dossier", "⚖️ Head-to-Head", "🎯 Shot Mechanics"],
                        label_visibility="collapsed",
                        key="nav_page")

    if page == "📋 Draft Board":
        page_draft_board(shots, summary, bios, all_scores, box_scores, combine)
    elif page == "👤 Player Dossier":
        page_player_dossier(shots, summary, all_ncaa_summary, bios, intl_stats, all_scores, box_scores, combine)
    elif page == "⚖️ Head-to-Head":
        page_h2h(shots, summary, bios, intl_stats, all_scores, box_scores, combine)
    else:
        page_shot_mechanics(bios)


if __name__ == "__main__":
    main()
