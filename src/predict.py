"""
Predicts one upcoming fixture using the trained models.

Usage:
    python src/predict.py "Arsenal" "Chelsea" "A Taylor"

This looks up each team's current rolling form and the referee's
career-to-date profile from the most recent data on file, builds the
same feature row used in training, and returns:
  - Win / Draw / Loss probabilities
  - Over/Under 2.5 goals probability
  - Over/Under 3.5 cards probability (referee-driven)
"""
import sys
import os

import joblib
import numpy as np
import pandas as pd

import features
from features import build_feature_table

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def latest_team_row(df: pd.DataFrame, team: str, prefix: str) -> pd.Series:
    """Grab a team's most recent rolling-form snapshot regardless of venue."""
    home_cols = [c for c in df.columns if c.startswith("Home_")]
    away_cols = [c for c in df.columns if c.startswith("Away_")]
    generic = [c.replace("Home_", "") for c in home_cols]

    home_rows = df[df["HomeTeam"] == team][["Date"] + home_cols].rename(
        columns=dict(zip(home_cols, generic))
    )
    away_rows = df[df["AwayTeam"] == team][["Date"] + away_cols].rename(
        columns=dict(zip(away_cols, generic))
    )
    combined = pd.concat([home_rows, away_rows]).sort_values("Date")
    if combined.empty:
        raise ValueError(f"No historical rows found for team '{team}'")
    latest = combined.iloc[-1]
    return latest.rename(lambda c: f"{prefix}_{c}" if c != "Date" else c)


def latest_referee_profile(df: pd.DataFrame, referee: str) -> dict:
    ref_rows = df[df["Referee"].str.lower() == referee.lower()]
    if ref_rows.empty:
        # fall back to league average of the most recent season on file
        latest_season = df["season"].max()
        recent = df[df["season"] == latest_season]
        return {
            "Ref_AvgCards": recent["Ref_AvgCards"].mean(),
            "Ref_AvgFouls": recent["Ref_AvgFouls"].mean(),
            "Ref_AvgReds": recent["Ref_AvgReds"].mean(),
            "Ref_CareerGames": 0,
        }
    latest = ref_rows.sort_values("Date").iloc[-1]
    return {
        "Ref_AvgCards": latest["Ref_AvgCards"],
        "Ref_AvgFouls": latest["Ref_AvgFouls"],
        "Ref_AvgReds": latest["Ref_AvgReds"],
        "Ref_CareerGames": latest["Ref_CareerGames"],
    }


def predict_fixture(home_team: str, away_team: str, referee: str, df: pd.DataFrame | None = None):
    if df is None:
        df = build_feature_table()

    home_feats = latest_team_row(df, home_team, "Home")
    away_feats = latest_team_row(df, away_team, "Away")
    ref_feats = latest_referee_profile(df, referee)

    row = {}
    row.update(home_feats.drop("Date").to_dict())
    row.update(away_feats.drop("Date").to_dict())
    row.update(ref_feats)
    row["H2H_HomePoints_avg"] = df["H2H_HomePoints_avg"].mean()   # neutral fallback
    row["H2H_TotalGoals_avg"] = df["H2H_TotalGoals_avg"].mean()

    feature_cols = features.FEATURE_COLUMNS
    X = pd.DataFrame([row])[feature_cols]

    m_1x2 = joblib.load(os.path.join(MODEL_DIR, "model_1x2.joblib"))
    m_over25 = joblib.load(os.path.join(MODEL_DIR, "model_over25.joblib"))
    m_cards35 = joblib.load(os.path.join(MODEL_DIR, "model_cards35.joblib"))

    # fill missing features with the SAME per-column values used at training
    # time (saved alongside each model), not medians recomputed from
    # whatever data happens to be loaded now -- keeps predictions consistent
    # with what the model actually learned.
    def _fill_path(model_name):
        return os.path.join(MODEL_DIR, f"{model_name}_fillvalues.joblib")

    fv_1x2 = joblib.load(_fill_path("model_1x2"))
    fv_over25 = joblib.load(_fill_path("model_over25"))
    fv_cards35 = joblib.load(_fill_path("model_cards35"))

    p_1x2 = m_1x2.predict_proba(X.fillna(fv_1x2))[0]
    p_over25 = m_over25.predict_proba(X.fillna(fv_over25))[0, 1]
    p_cards35 = m_cards35.predict_proba(X.fillna(fv_cards35))[0, 1]

    return {
        "fixture": f"{home_team} vs {away_team}",
        "referee": referee,
        "referee_avg_cards_per_game": round(float(ref_feats["Ref_AvgCards"]), 2),
        "home_win_pct": round(float(p_1x2[0]) * 100, 1),
        "draw_pct": round(float(p_1x2[1]) * 100, 1),
        "away_win_pct": round(float(p_1x2[2]) * 100, 1),
        "over_2.5_goals_pct": round(float(p_over25) * 100, 1),
        "under_2.5_goals_pct": round((1 - float(p_over25)) * 100, 1),
        "over_3.5_cards_pct": round(float(p_cards35) * 100, 1),
        "under_3.5_cards_pct": round((1 - float(p_cards35)) * 100, 1),
    }


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) != 3:
        print('Usage: python src/predict.py "HomeTeam" "AwayTeam" "Referee"')
        sys.exit(1)
    result = predict_fixture(*args)
    for k, v in result.items():
        print(f"{k:28s}: {v}")
