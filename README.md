# World Cup Match Predictor

A full-stack football match prediction system that combines a calibrated XGBoost classifier and Poisson simulation to forecast international match outcomes with win/draw/loss probabilities and expected goals.

## Features

- **Dual prediction engine** — XGBoost classifier + 10,000-match Poisson simulation run in parallel for robust probability estimates
- **Custom Elo rating system** — built from scratch across 47,000+ international match results with tournament-weighted K-factors (World Cup K=60, Friendly K=20)
- **Rolling form pipeline** — 10-match rolling averages for attacking and defensive output per team, updated dynamically
- **Live REST API** — FastAPI backend with endpoints for predictions, team stats, match history, and real-time model retraining
- **Polymarket integration** — benchmarks model predictions against real-world betting market odds via the Gamma API
- **Self-updating** — adding a new match result triggers automatic dataset rebuild and model retrain

## Tech Stack

- Python, FastAPI, XGBoost, Scikit-learn, Pandas, NumPy, SciPy, Joblib

## Project Structure

## Setup

```bash
pip install -r requirements.txt
```

**Step 1 — Build dataset**
```bash
python worldcup_full_pipeline.py --build
```

**Step 2 — Train model**
```bash
python worldcup_full_pipeline.py --train
```

**Step 3 — Start API**
```bash
uvicorn main:app --reload --port 8000
```

## Usage

**Quick CLI prediction**
```bash
python worldcup_full_pipeline.py --predict "Argentina" "France"
```

**API prediction (POST)**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"home_team": "Argentina", "away_team": "France", "neutral": true}'
```

**Example response**
```json
{
  "home_team": "Argentina",
  "away_team": "France",
  "xgboost": {
    "home_win_pct": 42.3,
    "draw_pct": 24.1,
    "away_win_pct": 33.6
  },
  "poisson": {
    "home_win_pct": 40.1,
    "draw_pct": 25.5,
    "away_win_pct": 34.4,
    "home_xg": 1.48,
    "away_xg": 1.31
  }
}
```

## Model Validation

Model was tested on held-out 2018 and 2022 World Cup matches to evaluate real-world predictive performance.

## Data

International match results sourced from publicly available football history datasets. Dataset not included in this repo — place your `results.csv` in the `/archive` folder before running the pipeline.

## Author

Won Se — Biology & Pre-Med @ Trevecca Nazarene University  
[LinkedIn](https://www.linkedin.com/in/won-seo-157197425/)
