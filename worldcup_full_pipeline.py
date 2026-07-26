# -*- coding: utf-8 -*-
"""
World Cup Match Predictor - Full Pipeline
Usage:
  python worldcup_full_pipeline.py --build
  python worldcup_full_pipeline.py --train
  python worldcup_full_pipeline.py --predict "Argentina" "France"
"""

import argparse
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, log_loss, classification_report
from xgboost import XGBClassifier
import joblib

# ---------------------------------------------------------------
# Paths - edit DATA_DIR to point to your archive folder
# ---------------------------------------------------------------
DATA_DIR   = r"C:\Users\wonse\OneDrive\Documents\World Cup AI\archive"
RESULTS    = os.path.join(DATA_DIR, "results.csv")
ML_DATASET = os.path.join(DATA_DIR, "ml_dataset.csv")
MODEL_FILE = os.path.join(DATA_DIR, "wc_model.joblib")

# ---------------------------------------------------------------
# K-factor by match importance
# ---------------------------------------------------------------
K_MAP = {
    "FIFA World Cup":           60,
    "Confederations Cup":       50,
    "AFC Asian Cup":            40,
    "UEFA Euro":                40,
    "Copa America":             40,
    "African Cup of Nations":   40,
    "CONCACAF Gold Cup":        40,
    "FIFA World Cup qualification": 35,
    "UEFA Euro qualification":  30,
    "Friendly":                 20,
}
DEFAULT_K = 30
HOME_ADV  = 100   # Elo points added to home side expected score
START_ELO = 1500


def get_k(tournament):
    for key, val in K_MAP.items():
        if key.lower() in str(tournament).lower():
            return val
    return DEFAULT_K


def expected_score(elo_a, elo_b):
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))


# ---------------------------------------------------------------
# PHASE 1: Build dataset
# ---------------------------------------------------------------
def build_dataset():
    print("Loading results.csv ...")
    df = pd.read_csv(RESULTS, parse_dates=["date"])
    df = df.dropna(subset=["home_score", "away_score"])
    df = df.sort_values("date").reset_index(drop=True)
    print(f"  {len(df):,} matches ({df['date'].dt.year.min()}-{df['date'].dt.year.max()})")

    # --- Compute Elo --------------------------------------------------
    print("Computing Elo ratings ...")
    elo = {}
    home_elo_list, away_elo_list = [], []

    for _, row in df.iterrows():
        h, a = row["home_team"], row["away_team"]
        elo.setdefault(h, START_ELO)
        elo.setdefault(a, START_ELO)

        # record pre-match Elo
        home_elo_list.append(elo[h])
        away_elo_list.append(elo[a])

        # determine actual result
        hs, as_ = row["home_score"], row["away_score"]
        if hs > as_:
            s_h, s_a = 1.0, 0.0
        elif hs < as_:
            s_h, s_a = 0.0, 1.0
        else:
            s_h = s_a = 0.5

        is_neutral = bool(row.get("neutral", False))
        adj = 0 if is_neutral else HOME_ADV

        e_h = expected_score(elo[h] + adj, elo[a])
        e_a = 1.0 - e_h
        k   = get_k(row["tournament"])

        elo[h] += k * (s_h - e_h)
        elo[a] += k * (s_a - e_a)

    df["home_elo"]    = home_elo_list
    df["away_elo"]    = away_elo_list
    df["elo_delta"]   = df["home_elo"] - df["away_elo"]

    # --- Rolling form (goals scored / conceded, last 10 matches) ------
    print("Building rolling form ...")

    long = pd.concat([
        df[["date","home_team","home_score","away_score"]].rename(
            columns={"home_team":"team","home_score":"gf","away_score":"ga"}),
        df[["date","away_team","away_score","home_score"]].rename(
            columns={"away_team":"team","away_score":"gf","home_score":"ga"}),
    ]).sort_values("date").reset_index(drop=True)

    long["form_att"] = (
        long.groupby("team")["gf"]
            .transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())
    )
    long["form_def"] = (
        long.groupby("team")["ga"]
            .transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())
    )

    # keep the last record per (team, date) for the merge
    form = long[["date","team","form_att","form_def"]].dropna()
    form = form.sort_values("date").reset_index(drop=True)

    df = df.sort_values("date").reset_index(drop=True)

    df = pd.merge_asof(
        df, form.rename(columns={"team":"home_team","form_att":"home_form_att","form_def":"home_form_def"}),
        on="date", left_by="home_team", right_by="home_team"
    )
    df = pd.merge_asof(
        df, form.rename(columns={"team":"away_team","form_att":"away_form_att","form_def":"away_form_def"}),
        on="date", left_by="away_team", right_by="away_team"
    )

    df = df.dropna(subset=["home_form_att","away_form_att"])

    # --- Target label ------------------------------------------------
    def outcome(row):
        if   row["home_score"] > row["away_score"]: return 0   # home win
        elif row["home_score"] < row["away_score"]: return 2   # away win
        else:                                        return 1   # draw

    df["match_outcome"] = df.apply(outcome, axis=1)

    # --- Extra features ----------------------------------------------
    df["is_neutral"]   = df["neutral"].astype(int)
    df["is_world_cup"] = df["tournament"].str.contains("FIFA World Cup", case=False, na=False).astype(int)
    df["form_att_delta"] = df["home_form_att"] - df["away_form_att"]
    df["form_def_delta"] = df["home_form_def"] - df["away_form_def"]

    print(f"  Features engineered. {len(df):,} usable rows.")
    print(f"  World Cup matches: {df['is_world_cup'].sum()}")

    df.to_csv(ML_DATASET, index=False)
    print(f"Saved to {ML_DATASET}")

    sample_cols = ["date","home_team","away_team","home_elo","away_elo",
                   "elo_delta","home_form_att","away_form_att","match_outcome"]
    print("\nSample output:")
    print(df[sample_cols].tail(5).to_string(index=False))


# ---------------------------------------------------------------
# PHASE 2: Train model
# ---------------------------------------------------------------
FEATURES = [
    "elo_delta","home_form_att","away_form_att",
    "home_form_def","away_form_def","form_att_delta",
    "form_def_delta","is_neutral","is_world_cup"
]
LABEL_MAP = {0: "Home Win", 1: "Draw", 2: "Away Win"}


def train_model():
    print("Loading ml_dataset.csv ...")
    df = pd.read_csv(ML_DATASET, parse_dates=["date"])
    print(f"  {len(df):,} rows loaded")

    df = df.dropna(subset=FEATURES + ["match_outcome"])

    test_years  = df["date"].dt.year.isin([2018, 2022])
    is_wc       = df["is_world_cup"] == 1
    train_mask  = ~(test_years & is_wc)
    test_mask   = test_years & is_wc

    X_train, y_train = df.loc[train_mask, FEATURES], df.loc[train_mask, "match_outcome"]
    X_test,  y_test  = df.loc[test_mask,  FEATURES], df.loc[test_mask,  "match_outcome"]

    print(f"Training on {len(X_train):,} matches")
    print(f"Testing  on {len(X_test):,} World Cup matches (2018 + 2022)")

    dist = y_train.value_counts().sort_index()
    labels = {0: "Home win", 1: "Draw", 2: "Away win"}
    print("Training set class distribution:")
    for k, v in dist.items():
        print(f"  {labels[k]}: {v:,} ({v/len(y_train)*100:.1f}%)")

    # Upweight draws to improve draw recall
    weight_map = {0: 1.0, 1: 2.2, 2: 1.0}
    sample_weights = y_train.map(weight_map).values

    # FIX 1: removed use_label_encoder=False (dropped in XGBoost 2.0+)
    base = XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=42,
        verbosity=0,
    )

    print("Training (this may take ~30 seconds) ...")
    model = CalibratedClassifierCV(base, method="isotonic", cv=3)
    model.fit(X_train, y_train, sample_weight=sample_weights)
    print("Done.")

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    print(f"Accuracy  : {accuracy_score(y_test, y_pred)*100:.2f}%")
    print(f"Log-loss  : {log_loss(y_test, y_proba):.4f}  (lower = better calibrated)")
    print(classification_report(y_test, y_pred,
                                target_names=["Home Win","Draw","Away Win"]))

    # FIX 2: access feature importances through calibrated model internals
    try:
        fi = model.calibrated_classifiers_[0].estimator.feature_importances_
        print("Feature importances:")
        pairs = sorted(zip(FEATURES, fi), key=lambda x: -x[1])
        max_fi = pairs[0][1]
        for feat, imp in pairs:
            bar = "#" * int(imp / max_fi * 80)
            print(f"  {feat:<20} {imp:.4f}  {bar}")
    except Exception:
        pass

    # Save
    joblib.dump(model, MODEL_FILE)
    print(f"Model saved to {MODEL_FILE}")

    # Quick demo prediction
    print("\n-- Sample prediction: Argentina vs France (neutral) --")
    result = predict_match("Argentina", "France", neutral=True)
    print(f"  Argentina win : {result['xgb']['home_win']*100:.1f}%")
    print(f"  Draw          : {result['xgb']['draw']*100:.1f}%")
    print(f"  France win    : {result['xgb']['away_win']*100:.1f}%")
    print(f"  Poisson xG    : Argentina {result['poisson']['home_xg']:.2f}  France {result['poisson']['away_xg']:.2f}")


# ---------------------------------------------------------------
# PHASE 3: Predict a match
# ---------------------------------------------------------------
def get_team_stats(team):
    """Return the latest Elo + form stats for a team from ml_dataset."""
    df = pd.read_csv(ML_DATASET, parse_dates=["date"])
    home_rows = df[df["home_team"] == team].copy()
    away_rows = df[df["away_team"] == team].copy()

    if home_rows.empty and away_rows.empty:
        raise ValueError(f"Team '{team}' not found in dataset.")

    # Most recent home appearance
    if not home_rows.empty:
        latest_home = home_rows.sort_values("date").iloc[-1]
        elo   = latest_home["home_elo"]
        f_att = latest_home["home_form_att"]
        f_def = latest_home["home_form_def"]
    else:
        latest_away = away_rows.sort_values("date").iloc[-1]
        elo   = latest_away["away_elo"]
        f_att = latest_away["away_form_att"]
        f_def = latest_away["away_form_def"]

    return {"elo": elo, "form_att": f_att, "form_def": f_def}


def simulate_poisson(home_xg, away_xg, n=10_000):
    """Simulate n matches using Poisson distributions."""
    h_goals = np.random.poisson(home_xg, n)
    a_goals = np.random.poisson(away_xg, n)
    home_wins = np.sum(h_goals > a_goals) / n
    draws     = np.sum(h_goals == a_goals) / n
    away_wins = np.sum(h_goals < a_goals) / n
    return home_wins, draws, away_wins


def predict_match(home_team, away_team, neutral=True):
    model = joblib.load(MODEL_FILE)

    h = get_team_stats(home_team)
    a = get_team_stats(away_team)

    elo_delta      = h["elo"] - a["elo"]
    form_att_delta = h["form_att"] - a["form_att"]
    form_def_delta = h["form_def"] - a["form_def"]
    is_neutral     = int(neutral)
    is_world_cup   = 1

    X = pd.DataFrame([{
        "elo_delta":       elo_delta,
        "home_form_att":   h["form_att"],
        "away_form_att":   a["form_att"],
        "home_form_def":   h["form_def"],
        "away_form_def":   a["form_def"],
        "form_att_delta":  form_att_delta,
        "form_def_delta":  form_def_delta,
        "is_neutral":      is_neutral,
        "is_world_cup":    is_world_cup,
    }])

    proba = model.predict_proba(X)[0]   # [home_win, draw, away_win]

    # Poisson expected goals: attack * (opponent's average goals conceded / league avg)
    league_avg_goals = 1.35
    home_xg = h["form_att"] * (a["form_def"] / league_avg_goals)
    away_xg = a["form_att"] * (h["form_def"] / league_avg_goals)
    ph, pd_, pa = simulate_poisson(home_xg, away_xg)

    return {
        "xgb": {
            "home_win": float(proba[0]),
            "draw":     float(proba[1]),
            "away_win": float(proba[2]),
        },
        "poisson": {
            "home_win": ph,
            "draw":     pd_,
            "away_win": pa,
            "home_xg":  round(home_xg, 2),
            "away_xg":  round(away_xg, 2),
        },
    }


# ---------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="World Cup Match Predictor")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--build",   action="store_true", help="Build ml_dataset.csv from results.csv")
    group.add_argument("--train",   action="store_true", help="Train the XGBoost model")
    group.add_argument("--predict", nargs=2, metavar=("HOME", "AWAY"), help="Predict a match")
    args = parser.parse_args()

    if args.build:
        build_dataset()
    elif args.train:
        train_model()
    elif args.predict:
        home_team, away_team = args.predict
        print(f"Predicting: {home_team} vs {away_team}")
        result = predict_match(home_team, away_team, neutral=True)
        xgb = result["xgb"]
        poi = result["poisson"]
        print()
        print(f"  --- XGBoost (calibrated) ---")
        print(f"  {home_team} win : {xgb['home_win']*100:.1f}%")
        print(f"  Draw            : {xgb['draw']*100:.1f}%")
        print(f"  {away_team} win : {xgb['away_win']*100:.1f}%")
        print()
        print(f"  --- Poisson simulation (10,000 matches) ---")
        print(f"  {home_team} win : {poi['home_win']*100:.1f}%")
        print(f"  Draw            : {poi['draw']*100:.1f}%")
        print(f"  {away_team} win : {poi['away_win']*100:.1f}%")
        print(f"  Expected goals  : {home_team} {poi['home_xg']}  {away_team} {poi['away_xg']}")


if __name__ == "__main__":
    main()
