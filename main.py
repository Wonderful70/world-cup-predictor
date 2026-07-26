import os
import csv
import subprocess
import sys
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from scipy.stats import poisson
import requests

# ---------------------------------------------------------------
# Paths
# ---------------------------------------------------------------
RESULTS_CSV     = r"C:\Users\wonse\OneDrive\Documents\World Cup AI\archive\results.csv"
ML_DATASET_PATH = r"C:\Users\wonse\OneDrive\Documents\World Cup AI\archive\ml_dataset.csv"
MODEL_PATH      = r"C:\Users\wonse\OneDrive\Documents\World Cup AI\archive\wc_model.joblib"
PIPELINE_SCRIPT = r"C:\Users\wonse\OneDrive\Documents\World Cup AI\worldcup_full_pipeline.py"

# FIX 3: safe startup — won't crash if files don't exist yet
try:
    MODEL = joblib.load(MODEL_PATH)
    DF    = pd.read_csv(ML_DATASET_PATH, parse_dates=["date"])
    print("Model and dataset loaded successfully.")
except FileNotFoundError as e:
    print(f"WARNING: {e}")
    print("Run --build then --train before making predictions.")
    MODEL = None
    DF    = None

# ---------------------------------------------------------------
# App setup
# ---------------------------------------------------------------
app = FastAPI(title="World Cup Predictor")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------
def _require_data():
    """Raise a clear 503 if model/dataset aren't loaded yet."""
    if MODEL is None or DF is None:
        raise HTTPException(
            status_code=503,
            detail="Model not ready. Run --build then --train first.",
        )


def get_team_stats(team):
    h = DF[DF["home_team"] == team]
    a = DF[DF["away_team"] == team]
    if h.empty and a.empty:
        raise HTTPException(404, f"Team not found: {team}")
    if not h.empty:
        r = h.sort_values("date").iloc[-1]
        return {
            "elo":       round(float(r["home_elo"]),    1),
            "form_att":  round(float(r["home_form_att"]), 2),
            "form_def":  round(float(r["home_form_def"]), 2),
        }
    r = a.sort_values("date").iloc[-1]
    return {
        "elo":       round(float(r["away_elo"]),    1),
        "form_att":  round(float(r["away_form_att"]), 2),
        "form_def":  round(float(r["away_form_def"]), 2),
    }


def get_last_n_matches(team, n=10):
    """Return the last n matches for a team, most recent first."""
    h = DF[DF["home_team"] == team].copy()
    h["opponent"]   = h["away_team"]
    h["team_score"] = h["home_score"]
    h["opp_score"]  = h["away_score"]
    h["venue"]      = "Home"

    a = DF[DF["away_team"] == team].copy()
    a["opponent"]   = a["home_team"]
    a["team_score"] = a["away_score"]
    a["opp_score"]  = a["home_score"]
    a["venue"]      = "Away"

    combined = pd.concat([h, a]).sort_values("date", ascending=False).head(n)

    matches = []
    for _, row in combined.iterrows():
        ts, os_ = row["team_score"], row["opp_score"]
        result = "W" if ts > os_ else ("L" if ts < os_ else "D")
        matches.append({
            "date":       row["date"].strftime("%Y-%m-%d"),
            "opponent":   row["opponent"],
            "score":      f"{int(ts)}-{int(os_)}",
            "result":     result,
            "venue":      row["venue"],
            "tournament": row["tournament"],
        })
    return matches


def _slugify(s):
    return s.lower().replace(" ", "-").replace(".", "")


def _parse_outcomes(m):
    import json as _json
    outcomes = m.get("outcomes")
    prices   = m.get("outcomePrices")
    if isinstance(outcomes, str):
        try:
            outcomes = _json.loads(outcomes)
        except Exception:
            outcomes = None
    if isinstance(prices, str):
        try:
            prices = _json.loads(prices)
        except Exception:
            prices = None
    out = []
    if outcomes and prices:
        for o, p in zip(outcomes, prices):
            try:
                out.append({"outcome": o, "implied_pct": round(float(p) * 100, 1)})
            except Exception:
                continue
    return out


# ---------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------
class Req(BaseModel):
    home_team: str
    away_team: str
    neutral: bool = True


class NewMatch(BaseModel):
    date:       str
    home_team:  str
    away_team:  str
    home_score: int
    away_score: int
    tournament: str  = "FIFA World Cup"
    city:       str  = ""
    country:    str  = ""
    neutral:    bool = True


# ---------------------------------------------------------------
# Routes
# ---------------------------------------------------------------
@app.get("/")
def root():
    return {"status": "ok"}


@app.get("/teams")
def teams():
    _require_data()
    return {"teams": sorted(set(DF["home_team"].dropna()) | set(DF["away_team"].dropna()))}


@app.get("/last10/{team}")
def last10(team: str):
    _require_data()
    matches = get_last_n_matches(team, 10)
    if not matches:
        raise HTTPException(404, f"No match history for {team}")
    return {"team": team, "matches": matches}


@app.post("/add_match")
def add_match(m: NewMatch):
    """
    Append a new match result to results.csv, then rebuild the dataset
    and retrain the model so it's reflected immediately in predictions.
    """
    try:
        with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                m.date, m.home_team, m.away_team,
                m.home_score, m.away_score,
                m.tournament, m.city, m.country,
                "True" if m.neutral else "False",
            ])
    except Exception as e:
        raise HTTPException(500, f"Could not write to results.csv: {e}")

    python_exe = sys.executable

    try:
        build_proc = subprocess.run(
            [python_exe, PIPELINE_SCRIPT, "--build"],
            capture_output=True, text=True, timeout=120,
        )
        if build_proc.returncode != 0:
            raise HTTPException(500, f"Build step failed:\n{build_proc.stderr[-2000:]}")

        train_proc = subprocess.run(
            [python_exe, PIPELINE_SCRIPT, "--train"],
            capture_output=True, text=True, timeout=180,
        )
        if train_proc.returncode != 0:
            raise HTTPException(500, f"Train step failed:\n{train_proc.stderr[-2000:]}")
    except subprocess.TimeoutExpired:
        raise HTTPException(500, "Rebuild/retrain timed out.")

    # Reload globals after retrain
    global DF, MODEL
    DF    = pd.read_csv(ML_DATASET_PATH, parse_dates=["date"])
    MODEL = joblib.load(MODEL_PATH)

    return {
        "status":  "ok",
        "message": (
            f"Match added: {m.home_team} {m.home_score}-{m.away_score} {m.away_team}. "
            "Dataset rebuilt and model retrained."
        ),
        "train_output_tail": train_proc.stdout[-800:],
    }


@app.get("/polymarket/{team_a}/{team_b}")
def polymarket_odds(team_a: str, team_b: str):
    """
    Try to find a Polymarket event for this matchup using the Gamma API.
    Tries multiple slug patterns and both team orderings.
    """
    a_slug = _slugify(team_a)
    b_slug = _slugify(team_b)

    candidate_slugs = [
        f"{a_slug}-vs-{b_slug}",
        f"{b_slug}-vs-{a_slug}",
        f"world-cup-{a_slug}-vs-{b_slug}",
        f"world-cup-{b_slug}-vs-{a_slug}",
    ]

    for slug in candidate_slugs:
        try:
            resp = requests.get(
                "https://gamma-api.polymarket.com/events",
                params={"slug": slug},
                timeout=6,
            )
            resp.raise_for_status()
            events = resp.json()
        except Exception:
            continue

        if isinstance(events, dict):
            events = events.get("data", [])

        if events:
            event        = events[0]
            markets      = event.get("markets", [])
            outcomes_data = []
            for mkt in markets:
                outcomes_data.extend(_parse_outcomes(mkt))
            if outcomes_data:
                return {
                    "found":    True,
                    "title":    event.get("title"),
                    "url":      f"https://polymarket.com/event/{event.get('slug', '')}",
                    "outcomes": outcomes_data,
                }

    return {
        "found":   False,
        "message": (
            f"No active Polymarket market found for {team_a} vs {team_b}. "
            "It may not exist yet, may use a different slug, or may have closed."
        ),
    }


@app.post("/predict")
def predict(req: Req):
    _require_data()
    h = get_team_stats(req.home_team)
    a = get_team_stats(req.away_team)

    X = pd.DataFrame([{
        "elo_delta":       h["elo"] - a["elo"],
        "home_form_att":   h["form_att"],
        "away_form_att":   a["form_att"],
        "home_form_def":   h["form_def"],
        "away_form_def":   a["form_def"],
        "form_att_delta":  h["form_att"] - a["form_att"],
        "form_def_delta":  h["form_def"] - a["form_def"],
        "is_neutral":      int(req.neutral),
        "is_world_cup":    1,
    }])

    p = MODEL.predict_proba(X)[0]

    hxg = h["form_att"] * (a["form_def"] / 1.35)
    axg = a["form_att"] * (h["form_def"] / 1.35)

    np.random.seed(0)
    hg = np.random.poisson(hxg, 10000)
    ag = np.random.poisson(axg, 10000)

    sc  = {
        f"{hi}-{ai}": round(poisson.pmf(hi, hxg) * poisson.pmf(ai, axg) * 100, 2)
        for hi in range(7) for ai in range(7)
    }
    top = dict(sorted(sc.items(), key=lambda x: -x[1])[:10])

    last10_home = get_last_n_matches(req.home_team, 10)
    last10_away = get_last_n_matches(req.away_team, 10)

    return {
        "home_team": req.home_team,
        "away_team": req.away_team,
        "xgboost": {
            "home_win_pct": round(float(p[0]) * 100, 1),
            "draw_pct":     round(float(p[1]) * 100, 1),
            "away_win_pct": round(float(p[2]) * 100, 1),
        },
        "poisson": {
            "home_win_pct":   round(float(np.mean(hg > ag)) * 100, 1),
            "draw_pct":       round(float(np.mean(hg == ag)) * 100, 1),
            "away_win_pct":   round(float(np.mean(hg < ag)) * 100, 1),
            "home_xg":        round(hxg, 2),
            "away_xg":        round(axg, 2),
            "top_scorelines": top,
        },
        "home_stats":  h,
        "away_stats":  a,
        "last10_home": last10_home,
        "last10_away": last10_away,
    }
