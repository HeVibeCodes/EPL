"""
Predicts one upcoming fixture using the trained models.

Usage:
    python src/predict.py "Arsenal" "Chelsea" "A Taylor"

Looks up each team's current rolling form and the referee's career-to-date
profile from the most recent data on file, builds the same feature row used
in training, and returns:
  - Win / Draw / Loss probabilities
  - Every over/under market registered in features.MARKET_DEFS (goals at
    0.5/1.5/2.5/3.5/4.5, cards at 3.5/4.5, corners at 8.5/9.5/10.5, BTTS)
  - Match context: each team's last 5 results, the last 5 head-to-head
    meetings, and days of rest since each team's last match
"""
import os
import sys

import joblib
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


def _load_market_model(model_name: str):
    """Returns (pipeline, fill_values, threshold) for one market, or None if
    that market hasn't been trained yet (e.g. an older models/ folder)."""
    path = os.path.join(MODEL_DIR, f"{model_name}.joblib")
    if not os.path.exists(path):
        return None
    pipe = joblib.load(path)
    fill_values = joblib.load(os.path.join(MODEL_DIR, f"{model_name}_fillvalues.joblib"))
    threshold_path = os.path.join(MODEL_DIR, f"{model_name}_threshold.joblib")
    threshold = joblib.load(threshold_path) if os.path.exists(threshold_path) else 0.5
    return pipe, fill_values, threshold


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

    # --- 1X2 -----------------------------------------------------------
    m_1x2 = joblib.load(os.path.join(MODEL_DIR, "model_1x2.joblib"))
    fv_1x2 = joblib.load(os.path.join(MODEL_DIR, "model_1x2_fillvalues.joblib"))
    p_1x2 = m_1x2.predict_proba(X.fillna(fv_1x2))[0]

    outcome = {
        "home_win_pct": round(float(p_1x2[0]) * 100, 1),
        "draw_pct": round(float(p_1x2[1]) * 100, 1),
        "away_win_pct": round(float(p_1x2[2]) * 100, 1),
    }

    # --- every registered over/under + BTTS market ----------------------
    markets = {"goals": [], "cards": [], "corners": [], "btts": None}
    for market in features.MARKET_DEFS:
        loaded = _load_market_model(market["model_name"])
        if loaded is None:
            continue  # model not trained yet -- skip rather than error
        pipe, fill_values, threshold = loaded
        proba_over = float(pipe.predict_proba(X.fillna(fill_values))[0, 1])
        entry = {
            "line": market["line"],
            "label": market["label"],
            "over_pct": round(proba_over * 100, 1),
            "under_pct": round((1 - proba_over) * 100, 1),
            "decision_threshold": threshold,
        }
        if market["group"] == "btts":
            markets["btts"] = {
                "yes_pct": entry["over_pct"],
                "no_pct": entry["under_pct"],
            }
        else:
            markets[market["group"]].append(entry)

    # --- match context: form, head-to-head, rest days -------------------
    context = {
        "home_form": features.recent_results(df, home_team, n=5),
        "away_form": features.recent_results(df, away_team, n=5),
        "h2h": features.head_to_head_matches(df, home_team, away_team, n=5),
        "home_rest_days": None if pd.isna(home_feats.get("Home_DaysSinceLast")) else int(home_feats["Home_DaysSinceLast"]),
        "away_rest_days": None if pd.isna(away_feats.get("Away_DaysSinceLast")) else int(away_feats["Away_DaysSinceLast"]),
    }

    return {
        "fixture": f"{home_team} vs {away_team}",
        "referee": referee,
        "referee_avg_cards_per_game": round(float(ref_feats["Ref_AvgCards"]), 2),
        "outcome": outcome,
        "markets": markets,
        "context": context,
    }


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) != 3:
        print('Usage: python src/predict.py "HomeTeam" "AwayTeam" "Referee"')
        sys.exit(1)
    import json
    print(json.dumps(predict_fixture(*args), indent=2, default=str))
