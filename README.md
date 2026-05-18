# NBA Draft 2026 — Shot Quality Scouting Portal

A scouting tool for the **2026 NBA Draft** that evaluates prospects through two lenses:

1. **Shot quality** — statistical analysis of every field goal attempt using an xPTS model trained on NCAA shot-chart data
2. **Shot mechanics** — frame-level joint angle extraction from game film using a custom YOLOv11 action classifier and RTMPose

> *Is a prospect a shot-maker, or just a shot-taker?*

---

## Core Metric: Points Above Expectation (PAE)

Each shot is assigned an expected point value by a LightGBM model trained on shot location, zone, shot type, and action type. PAE measures how many points a player generates **above** what an average shooter would score from the same attempts.

```
PAE/100 = (Actual Points − Expected Points) per 100 shot attempts
```

Raw PAE is shrunk toward the class mean using **Empirical Bayes** to handle small sample sizes:

```
Shrunk PAE = λ × raw_PAE + (1 − λ) × class_mean
λ = n / (n + 150)     (data trust; ≈0.67 at 300 FGA, ≈0.80 at 600 FGA)
```

---

## Project Structure

```
nba-shot-quality/
├── pipeline/
│   ├── prospects_2026.py           # Master prospect list (62 NCAA + 3 intl, with draft rank)
│   ├── ingest_prospects.py         # NCAA shot data → SQLite (cbbd PlaysApi)
│   ├── ingest_box_scores.py        # Full box scores → player_season_box table
│   ├── ingest_ncaa_historical.py   # Historical shot data for comp-player search
│   ├── fetch_prospect_bios.py      # Player bios → prospect_bios.json
│   ├── schema.sql                  # SQLite schema
│   └── train_action_classifier.py  # YOLOv11 action classifier training script
├── models/
│   ├── xpts_model.py               # LightGBM xPTS model + Empirical Bayes shrinkage
│   └── action_classifier/
│       └── weights/best.pt         # Trained YOLOv11n action classifier (mAP50=0.828)
├── dashboard/
│   └── app.py                      # Streamlit scouting portal (4 pages)
├── video/
│   ├── extract_shots_pipeline.py   # Game film → jump-shot clips (uses action classifier)
│   ├── review_clips.py             # Interactive OpenCV clip review (keep / delete)
│   ├── analyze_pose.py             # RTMPose joint angle extraction per clip
│   ├── review_pose.py              # Interactive OpenCV pose review (keep / delete)
│   ├── annotate_release.py         # Manual release-frame annotation tool
│   ├── make_zoom_preview.py        # Static-crop pose preview generator
│   ├── delete_clips.py             # Batch clip deletion utility
│   ├── detect_players.py           # Standalone YOLO player detection utility
│   └── <N>_<Player>/
│       └── jump_shot_clips/
│           ├── clip_*.mp4                    # Raw jump-shot clip
│           ├── clip_*_preview.mp4            # Annotated with tracking bbox
│           ├── clip_*_tracking.json          # Per-frame player bbox + source video info
│           ├── clip_*_pose.json              # Per-frame mechanics trajectory
│           ├── clip_*_pose_preview.mp4       # RTMPose skeleton overlay
│           ├── clip_*_pose_preview_web.mp4   # H.264-re-encoded for browser playback
│           └── index.json                    # Clip registry + review state
├── data/
│   ├── nba_shots.db                # SQLite: shots, player_season_summary, player_season_box
│   ├── combine_2026.json           # NBA Draft Combine measurements
│   ├── prospect_bios.json          # Player metadata (headshot, height, weight, team)
│   ├── international_stats.json    # Manually curated stats for intl prospects
│   ├── xpts_model.pkl              # Trained LightGBM model
│   └── xpts_encoders.pkl           # Feature encoders for xPTS model
├── yolo11n.pt                      # YOLOv11n base weights (used by train_action_classifier.py)
├── yolov8n.pt                      # YOLOv8n weights (used by detect_players.py)
├── train_colab.ipynb               # Colab notebook for action classifier training
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Environment setup

```bash
conda create -n nba python=3.11 -y
conda activate nba
pip install -r requirements.txt
export CBBD_API_KEY="your_key_here"
```

### 2. Ingest data

```bash
python pipeline/ingest_prospects.py        # NCAA shot-by-shot data (cbbd PlaysApi)
python pipeline/ingest_box_scores.py       # Full season box scores
python pipeline/ingest_ncaa_historical.py  # Historical comps (optional)
python pipeline/fetch_prospect_bios.py     # Player bios
```

### 3. Train the xPTS model

```bash
python models/xpts_model.py
```

Trains LightGBM on all ingested shots, computes PAE, applies Empirical Bayes shrinkage, and populates `player_season_summary`.

### 4. Launch the dashboard

```bash
streamlit run dashboard/app.py
```

---

## Dashboard Pages

### Draft Board
Sortable leaderboard of all prospects. Default order is **draft rank**. Columns: PAE/100, FG%, 3P%, FT%, 3PAr, shot-making Grade. Click any row to jump to that player's dossier. Metric glossary shown at the bottom.

### Player Dossier
Deep-dive profile for a single prospect:
- Bio strip (school, position, draft rank, height/weight)
- Season stats (PPG, RPG, APG, FT%, TS%, USG%)
- Shot quality cards (FGA, FG%, PAE/100, data trust λ)
- 5-dimension Plotly radar (Shot Making, Outside Range, At-Rim, Shot Diet, Consistency)
- Interactive half-court zone diagram (PAE vs class average by zone)
- Comparable player search (cosine similarity on 4-feature shooting profile)
- PAE/100 class distribution histogram
- NBA Draft Combine physical/athleticism data

### Head-to-Head
Side-by-side comparison of any two prospects, with optional position filter:
- Back-to-back horizontal bar chart (bars proportional to actual values within realistic metric domains)
- Radar overlay (both players + class mean reference)
- Combine physical comparison

### Shot Mechanics
Film-based mechanics analysis for the top 7 prospects (≥10 reviewed clips each):
- Player snapshot: Load Depth, Consistency (std dev), Arm Setup, Release Lean
- Video player with ◀ ▶ navigation (auto-play, sorted best clips first)
- Three trajectory charts: Knee Angle, Elbow Angle, Body Lean — this clip vs player mean vs all clips, aligned to release frame

---

## Shot Mechanics Pipeline

### Status (as of May 2026)

| # | Player | Clips Reviewed | Pose Clips | In Dashboard |
|---|--------|:-:|:-:|:-:|
| 1 | Cameron Boozer | ✅ | ✅ 39 clips | ✅ |
| 2 | Darryn Peterson | ✅ | ✅ | ✅ |
| 3 | AJ Dybantsa | ✅ | ✅ | ✅ |
| 4 | Caleb Wilson | ✅ | ✅ | ✅ |
| 5 | Mikel Brown Jr. | ✅ | ✅ | ✅ |
| 6 | Kingston Flemings | ✅ | ✅ | ✅ |
| 7 | Hannes Steinbach | ✅ | 8 clips | — (< 10) |
| 8 | Koa Peat | ✅ | 2 clips | — (< 10) |
| 9 | Labaron Philon | ✅ | 7 clips | — (< 10) |
| 10 | Bennett Stirtz | ✅ | ✅ | ✅ |

### Pipeline flow

```
Game film (1080p 60fps, downloaded via yt-dlp)
       ↓
① extract_shots_pipeline.py
   YOLOv11n action classifier → detects jump-shot events (JUMP_CLASS=6)
   Merges detections within 2s → one shot event per attempt
   Extracts [anchor−1s, anchor+2s] clip
   Output: clip.mp4 + clip_preview.mp4 (with bbox) + clip_tracking.json
       ↓
② review_clips.py --player video/1_Cameron_Boozer
   OpenCV window — Space=pause, k=keep, d=delete, u=undo, ←/→=frame
   Progress saved to index.json after each decision; resumes across runs
       ↓
③ analyze_pose.py --player video/1_Cameron_Boozer
   RTMPose (rtmlib, COCO-17 keypoints) on each frame in ±0.5s window
   Closest-to-center person selection; stability filter (IoU < 0.30 = skip)
   Block outlier removal + 3-frame median smoothing
   Auto-deletes clips with < 10 valid pose frames
   Output: clip_pose.json + clip_pose_preview.mp4
       ↓
④ review_pose.py --player video/1_Cameron_Boozer
   Same OpenCV controls as review_clips; Left/Right arrows for frame stepping
       ↓
⑤ Re-encode pose previews for browser (H.264 faststart)
   for f in video/1_Cameron_Boozer/jump_shot_clips/*_pose_preview.mp4; do
     ffmpeg -y -i "$f" -vcodec libx264 -crf 23 -preset fast \
       -movflags +faststart -an "${f%.mp4}_web.mp4"
   done
```

### Mechanics metrics (per-frame trajectories across shot window)

| Metric | Keypoints | Interpretation |
|---|---|---|
| **Knee Angle** | Hip → Knee → Ankle | Drops during load, extends explosively at takeoff — most predictive of leg drive |
| **Elbow Angle** | Shoulder → Elbow → Wrist | Shows arm cocking during load and extension toward release |
| **Body Lean** | Shoulder midpoint → Hip midpoint | Torso tilt from vertical; ideally near 0° (upright) at release |

### Action classifier

- Architecture: YOLOv11 Nano, `imgsz=640`, trained 59 epochs
- Dataset: 12 basketball action classes (COCO format); see `basketball-player-detection-2-Forked on 8-19-2025.coco/`
- Weights: `models/action_classifier/weights/best.pt` (mAP50 = 0.828)
- Retrain: `python pipeline/train_action_classifier.py` or `train_colab.ipynb` (Google Colab)

---

## Five Shooting Dimensions (Radar)

| Dimension | What it measures |
|---|---|
| **Shot Making** | Shrunk PAE/100 percentile vs prospect class |
| **Outside Range** | 3PAr × 3P% composite |
| **At-Rim** | Restricted Area FG% |
| **Shot Diet** | Avg shot difficulty (higher xFG% threshold = tougher diet) |
| **Consistency** | PAE std dev — inverted so lower variance = higher score |

---

## Prospect Coverage

**62 NCAA prospects** (consensus top picks through mid-round, FanSided + No Ceilings boards, April 2026)

**3 international prospects** with manually curated stats:
- Karim Lopez — NBL, New Zealand Breakers
- Dash Daniels — NBL, Melbourne United
- Sergio De Larrea — ACB + EuroLeague, Valencia Basket
