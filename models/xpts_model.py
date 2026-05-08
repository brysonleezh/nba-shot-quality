"""
Expected Points (xPTS) Model
------------------------------
Trains a LightGBM classifier on historical NCAA shots (2022-23 to 2024-25),
then scores 2025-26 prospects and all historical players. Derives:

    xPTS  = P(make) × point_value     (2 or 3)
    PAE   = actual_pts - expected_pts  (Points Above Expectation)

Baseline is NCAA-calibrated (not NBA), so PAE reflects performance
relative to average NCAA shooting from the same locations.

Usage:
    python models/xpts_model.py
    python models/xpts_model.py --cv                     # show cross-val AUC
    python models/xpts_model.py --train-seasons 2022-23 2023-24 2024-25
    python models/xpts_model.py --score-season 2025-26
"""

import argparse
import sqlite3
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DB_PATH   = Path(__file__).resolve().parent.parent / "data" / "nba_shots.db"
MODEL_DIR = Path(__file__).resolve().parent.parent / "data"

# ── Feature engineering ───────────────────────────────────────────────────────

CATEGORICAL = ["shot_type", "shot_zone_basic", "shot_zone_area",
               "shot_zone_range", "action_type"]
NUMERIC     = ["shot_distance", "loc_x", "loc_y",
               "period", "minutes_remaining", "seconds_remaining"]
TARGET      = "shot_made"


def load_shots(conn: sqlite3.Connection,
               season: str | None = None,
               seasons: list[str] | None = None,
               league: str | None = None) -> pd.DataFrame:
    conditions = []
    params: list = []
    if season:
        conditions.append("season = ?")
        params.append(season)
    elif seasons:
        placeholders = ", ".join(["?"] * len(seasons))
        conditions.append(f"season IN ({placeholders})")
        params.extend(seasons)
    if league:
        conditions.append("league = ?")
        params.append(league)
    q = "SELECT * FROM shots"
    if conditions:
        q += " WHERE " + " AND ".join(conditions)
    return pd.read_sql_query(q, conn, params=params)


def build_features(df: pd.DataFrame,
                   encoders: dict | None = None,
                   fit: bool = True) -> tuple[pd.DataFrame, dict]:
    """One-hot encode categoricals; return feature matrix + fitted encoders."""
    df = df.copy()

    if encoders is None:
        encoders = {}

    encoded_frames = []
    for col in CATEGORICAL:
        df[col] = df[col].fillna("Unknown").astype(str)
        if fit:
            le = LabelEncoder()
            df[col + "_enc"] = le.fit_transform(df[col])
            encoders[col] = le
        else:
            le = encoders[col]
            df[col + "_enc"] = df[col].map(
                lambda x, le=le: le.transform([x])[0]
                if x in le.classes_ else -1
            )
        encoded_frames.append(col + "_enc")

    feature_cols = NUMERIC + encoded_frames
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    return df[feature_cols], encoders


def point_value(shot_type: pd.Series) -> pd.Series:
    return shot_type.map({"3PT Field Goal": 3}).fillna(2)


# ── Model training & scoring ──────────────────────────────────────────────────

def train(df: pd.DataFrame, run_cv: bool = False) -> tuple[LGBMClassifier, dict, float]:
    X, encoders = build_features(df, fit=True)
    y = df[TARGET].astype(int)

    model = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        colsample_bytree=0.8,
        subsample=0.8,
        random_state=42,
        verbose=-1,
    )

    auc = None
    if run_cv:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        auc = scores.mean()
        log.info("5-fold CV AUC: %.4f ± %.4f", auc, scores.std())

    model.fit(X, y)
    log.info("Model trained on %d shots", len(df))
    return model, encoders, auc


def score_shots(df: pd.DataFrame, model: LGBMClassifier,
                encoders: dict) -> pd.DataFrame:
    df = df.copy()
    X, _ = build_features(df, encoders=encoders, fit=False)
    df["p_make"]       = model.predict_proba(X)[:, 1]
    df["pts_value"]    = point_value(df["shot_type"])
    df["actual_pts"]   = df["shot_made"].astype(int) * df["pts_value"]
    df["expected_pts"] = df["p_make"] * df["pts_value"]
    df["pae_shot"]     = df["actual_pts"] - df["expected_pts"]   # per-shot PAE
    return df


# ── Summarisation → player_season_summary ────────────────────────────────────

def build_summary(df_scored: pd.DataFrame) -> pd.DataFrame:
    grp = df_scored.groupby(["player_id", "player_name", "season"])
    summary = grp.agg(
        total_shots  =("shot_attempted", "sum"),
        shots_made   =("shot_made", "sum"),
        total_pts    =("actual_pts", "sum"),
        expected_pts =("expected_pts", "sum"),
        avg_shot_dist=("shot_distance", "mean"),
    ).reset_index()

    summary["fg_pct"]        = summary["shots_made"] / summary["total_shots"]
    summary["pts_above_exp"] = summary["total_pts"] - summary["expected_pts"]

    three_pt = df_scored[df_scored["shot_type"] == "3PT Field Goal"]
    pct_3pt  = (three_pt.groupby(["player_id", "season"])["shot_attempted"]
                .sum()
                .reset_index()
                .rename(columns={"shot_attempted": "threes"}))
    summary = summary.merge(pct_3pt, on=["player_id", "season"], how="left")
    summary["pct_3pt"] = summary["threes"] / summary["total_shots"]
    summary = summary.drop(columns=["threes"])

    # Carry league through so shrinkage is applied per-league
    if "league" in df_scored.columns:
        player_league = (df_scored[["player_id", "league"]]
                         .drop_duplicates("player_id"))
        summary = summary.merge(player_league, on="player_id", how="left")

    summary = apply_shrinkage(df_scored, summary)
    return summary


def apply_shrinkage(df_scored: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    """
    Empirical Bayes partial pooling on PAE/shot within each (league, season) group.

    For each player i:
        y_i      = mean PAE/shot  (= pts_above_exp / total_shots)
        se²_i    = Var(pae_shot_j) / n_i   (variance of the sample mean)

    Group-level parameters estimated via method of moments:
        μ        = weighted mean of y_i
        τ²       = max(0, Var(y_i) - mean(se²_i))   (between-player variance)

    Shrinkage factor:
        λ_i      = τ² / (τ² + se²_i)
        θ̂_i     = λ_i * y_i + (1 - λ_i) * μ

    Output column: shrunk_pae_per100 = θ̂_i * 100
    """
    # Per-shot PAE variance per player-season
    shot_var = (df_scored
                .groupby(["player_id", "season"])["pae_shot"]
                .var()
                .reset_index()
                .rename(columns={"pae_shot": "pae_var"}))

    summary = summary.merge(shot_var, on=["player_id", "season"], how="left")
    summary["pae_var"] = summary["pae_var"].fillna(0)

    # Standard error² of the mean PAE/shot
    summary["se2"] = summary["pae_var"] / summary["total_shots"].clip(lower=1)
    # Raw PAE/shot
    summary["y"]   = summary["pts_above_exp"] / summary["total_shots"]

    # Apply shrinkage per (league, season) group
    league_col = "league" if "league" in summary.columns else None

    def shrink_group(grp: pd.DataFrame) -> pd.DataFrame:
        y   = grp["y"].values
        se2 = grp["se2"].values
        n   = len(grp)

        # Weighted mean (weight by 1/se²; fall back to simple mean if se² = 0)
        w = np.where(se2 > 0, 1.0 / se2, 1.0)
        mu = np.average(y, weights=w)

        # Between-player variance via method of moments
        var_y    = np.var(y, ddof=1) if n > 1 else 0.0
        mean_se2 = np.mean(se2)
        tau2     = max(0.0, var_y - mean_se2)

        # Shrinkage factor
        lam = np.where(tau2 + se2 > 0, tau2 / (tau2 + se2), 0.0)
        grp = grp.copy()
        grp["shrunk_pae_per100"] = (lam * y + (1 - lam) * mu) * 100
        grp["shrinkage_factor"]  = lam
        return grp

    group_keys = ["league", "season"] if league_col else ["season"]
    # Only group by league if the column exists in summary
    if league_col and league_col not in summary.columns:
        group_keys = ["season"]

    summary = summary.groupby(group_keys).apply(shrink_group, include_groups=False).reset_index()
    drop = ["pae_var", "se2", "y"] + [c for c in summary.columns if c.startswith("level_")]
    summary = summary.drop(columns=[c for c in drop if c in summary.columns])

    log.info("Shrinkage applied. Avg λ = %.3f", summary["shrinkage_factor"].mean())
    return summary


def write_shot_scores(conn: sqlite3.Connection, df_scored: pd.DataFrame) -> None:
    """Write p_make back to the shots table."""
    # ALTER TABLE is a no-op if the column already exists (caught and ignored)
    try:
        conn.execute("ALTER TABLE shots ADD COLUMN p_make REAL")
    except Exception:
        pass

    rows = df_scored[["shot_id", "p_make"]].itertuples(index=False, name=None)
    conn.executemany("UPDATE shots SET p_make = ? WHERE shot_id = ?",
                     ((p, sid) for sid, p in rows))
    conn.commit()
    log.info("p_make written back to shots table for %d rows", len(df_scored))


def write_summary(conn: sqlite3.Connection, summary: pd.DataFrame) -> None:
    summary.to_sql("player_season_summary", conn, if_exists="replace",
                   index=False, method="multi")
    conn.commit()
    log.info("player_season_summary updated: %d rows", len(summary))


# ── Main ──────────────────────────────────────────────────────────────────────

TRAIN_SEASONS = ["2022-23", "2023-24", "2024-25"]
SCORE_SEASON  = "2025-26"


def run(train_seasons: list[str] = TRAIN_SEASONS,
        score_season: str = SCORE_SEASON,
        run_cv: bool = False) -> None:
    conn = sqlite3.connect(DB_PATH)

    # ── Step 1: Train on historical NCAA data only ──
    df_train = load_shots(conn, seasons=train_seasons, league="NCAA")
    if df_train.empty:
        log.error(
            "No historical NCAA training data found for seasons %s. "
            "Run pipeline/ingest_ncaa_historical.py first.", train_seasons)
        conn.close()
        return
    log.info("Training on %d historical NCAA shots (%s)",
             len(df_train), ", ".join(train_seasons))

    model, encoders, _ = train(df_train, run_cv=run_cv)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model,    MODEL_DIR / "xpts_model.pkl")
    joblib.dump(encoders, MODEL_DIR / "xpts_encoders.pkl")
    log.info("Model saved to %s", MODEL_DIR)

    # ── Step 2: Score ALL NCAA shots (historical + target season) ──
    df_all_ncaa = load_shots(conn, league="NCAA")
    log.info("Scoring %d total NCAA shots", len(df_all_ncaa))

    df_scored = score_shots(df_all_ncaa, model, encoders)
    write_shot_scores(conn, df_scored)

    # ── Step 3: Build summary for all NCAA players ──
    # Dashboard filters to score_season; historical players power the comps feature
    summary = build_summary(df_scored)
    write_summary(conn, summary)
    log.info("Summaries written: %d player-seasons", len(summary))
    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NCAA xPTS model and score prospects")
    parser.add_argument(
        "--train-seasons", nargs="+", default=TRAIN_SEASONS,
        help="NCAA seasons to train on (e.g. 2022-23 2023-24 2024-25)")
    parser.add_argument(
        "--score-season", default=SCORE_SEASON,
        help="Season to evaluate prospects (default: 2025-26)")
    parser.add_argument("--cv", action="store_true", help="Run 5-fold CV and print AUC")
    args = parser.parse_args()
    run(train_seasons=args.train_seasons,
        score_season=args.score_season,
        run_cv=args.cv)
