"""
Turns raw football-data.co.uk season CSVs into a model-ready feature table.

Feature groups built here:
  1. Rolling team form       - points, goals for/against, shots, shot accuracy,
                                corners, over recent matches (pre-match only,
                                so nothing here leaks the result being predicted)
  2. Head-to-head             - recent history between the two exact teams
  3. Referee profile          - each referee's career-to-date average cards,
                                fouls, and penalty-for-red-card rate, computed
                                on a walk-forward basis (only using games the
                                ref had already officiated before this match)
  4. Squad-strength proxy     - a stand-in for "who's playing" (see note below
                                on the real player-level data gap)
  5. Market targets           - FTR (H/D/A), total-goals over/under lines,
                                and total cards for a cards over/under line
"""
from __future__ import annotations

import glob
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="DataFrame is highly fragmented")

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

RESULT_MAP = {"H": 0, "D": 1, "A": 2}  # class indices for the 1X2 model


# --------------------------------------------------------------------------- #
# 1. Load & tidy
# --------------------------------------------------------------------------- #
def _parse_date_column(s: pd.Series) -> pd.Series:
    """
    Parse one file's Date column using WHICHEVER single format explains
    almost all of its rows, tried in order: ISO (yyyy-mm-dd), then UK-style
    dd/mm/yyyy, then dd/mm/yy.

    This matters because pandas' format="mixed" heuristic, applied across a
    column that blends formats, can silently swap day/month for ambiguous
    dates (any day <= 12) -- e.g. misreading 2025-11-01 (1 Nov) as if it
    were 2025-01-11 (11 Jan). Detecting the format PER FILE first (each
    football-data.co.uk file uses one consistent format internally) avoids
    that ambiguity entirely instead of guessing row-by-row.
    """
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d/%m/%y"):
        parsed = pd.to_datetime(s, format=fmt, errors="coerce")
        if parsed.notna().mean() > 0.95:  # this format explains almost every row
            return parsed
    # no single format fit well -- fall back to pandas' best guess
    return pd.to_datetime(s, errors="coerce", dayfirst=True)


def load_all_seasons(raw_dir: str = RAW_DIR) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(raw_dir, "season-*.csv")))
    if not files:
        raise FileNotFoundError(
            f"No season CSVs found in {raw_dir}. Run src/download_data.py first."
        )
    frames = []
    for f in files:
        df = None
        # older football-data.co.uk files are Windows-1252, newer ones are
        # UTF-8 -- try both instead of assuming and dropping the file.
        for encoding in ("utf-8", "cp1252", "latin1"):
            try:
                df = pd.read_csv(f, engine="python", on_bad_lines="skip", encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
            except Exception as exc:
                print(f"  skipping {os.path.basename(f)}: could not parse it ({exc})")
                break
        if df is None:
            print(f"  skipping {os.path.basename(f)}: could not decode this file with utf-8/cp1252/latin1")
            continue
        if df.empty or "FTR" not in df.columns:
            print(f"  skipping {os.path.basename(f)}: no usable rows")
            continue
        df["season"] = os.path.basename(f).replace("season-", "").replace(".csv", "")
        df["Date"] = _parse_date_column(df["Date"])  # parsed per-file, before concatenation
        frames.append(df)

    if not frames:
        raise ValueError(f"None of the files in {raw_dir} could be parsed.")

    df = pd.concat(frames, ignore_index=True, sort=False)
    df = df.dropna(subset=["Date", "FTR"]).sort_values("Date").reset_index(drop=True)

    # normalise column presence -- very old seasons lack some stat columns
    for col in ["HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC", "HY", "AY", "HR", "AR"]:
        if col not in df.columns:
            df[col] = np.nan

    df["TotalGoals"] = df["FTHG"] + df["FTAG"]
    df["TotalCards"] = df[["HY", "AY", "HR", "AR"]].sum(axis=1)
    df["TotalFouls"] = df[["HF", "AF"]].sum(axis=1)
    df["TotalCorners"] = df[["HC", "AC"]].sum(axis=1)
    df["TotalShots"] = df[["HS", "AS"]].sum(axis=1)
    df["BTTS"] = ((df["FTHG"] > 0) & (df["FTAG"] > 0)).astype(int)
    return df


# --------------------------------------------------------------------------- #
# 2. Rolling team form (walk-forward, no leakage)
# --------------------------------------------------------------------------- #
def _team_match_log(df: pd.DataFrame) -> pd.DataFrame:
    """One row per team per match (long format) to make rolling windows easy."""
    home = df[["Date", "season", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
               "HS", "AS", "HST", "AST", "HC", "AC", "HF", "HY", "HR"]].copy()
    home = home.rename(columns={
        "HomeTeam": "Team", "AwayTeam": "Opponent",
        "FTHG": "GF", "FTAG": "GA", "HS": "Shots", "AS": "ShotsAgainst",
        "HST": "ShotsOnTarget", "AST": "ShotsOnTargetAgainst",
        "HC": "Corners", "AC": "CornersAgainst", "HF": "Fouls",
        "HY": "Yellows", "HR": "Reds",
    })
    home["Venue"] = "H"
    home["Points"] = home["FTR"].map({"H": 3, "D": 1, "A": 0})

    away = df[["Date", "season", "AwayTeam", "HomeTeam", "FTAG", "FTHG", "FTR",
               "AS", "HS", "AST", "HST", "AC", "HC", "AF", "AY", "AR"]].copy()
    away = away.rename(columns={
        "AwayTeam": "Team", "HomeTeam": "Opponent",
        "FTAG": "GF", "FTHG": "GA", "AS": "Shots", "HS": "ShotsAgainst",
        "AST": "ShotsOnTarget", "HST": "ShotsOnTargetAgainst",
        "AC": "Corners", "HC": "CornersAgainst", "AF": "Fouls",
        "AY": "Yellows", "AR": "Reds",
    })
    away["Venue"] = "A"
    away["Points"] = away["FTR"].map({"H": 0, "D": 1, "A": 3})

    long_df = pd.concat([home, away], ignore_index=True).sort_values(["Team", "Date"])
    return long_df


def team_match_log(df: pd.DataFrame) -> pd.DataFrame:
    """Public wrapper around _team_match_log for callers outside this module
    (the website uses this for recent-form and head-to-head display)."""
    return _team_match_log(df)


def recent_results(df: pd.DataFrame, team: str, n: int = 5) -> list[dict]:
    """Last n matches for a team, most recent first, as plain W/D/L rows for
    display (not model input -- rolling form features already cover that)."""
    log = _team_match_log(df)
    rows = log[log["Team"] == team].sort_values("Date", ascending=False).head(n)
    out = []
    for _, r in rows.iterrows():
        result = "W" if r["GF"] > r["GA"] else ("L" if r["GF"] < r["GA"] else "D")
        out.append({
            "date": r["Date"].strftime("%d %b %Y"),
            "opponent": r["Opponent"],
            "venue": r["Venue"],
            "score": f"{int(r['GF'])}-{int(r['GA'])}",
            "result": result,
        })
    return out


def head_to_head_matches(df: pd.DataFrame, home_team: str, away_team: str, n: int = 5) -> list[dict]:
    """Last n meetings between these two exact teams, most recent first."""
    mask = (
        ((df["HomeTeam"] == home_team) & (df["AwayTeam"] == away_team))
        | ((df["HomeTeam"] == away_team) & (df["AwayTeam"] == home_team))
    )
    rows = df[mask].sort_values("Date", ascending=False).head(n)
    out = []
    for _, r in rows.iterrows():
        out.append({
            "date": r["Date"].strftime("%d %b %Y"),
            "home": r["HomeTeam"],
            "away": r["AwayTeam"],
            "score": f"{int(r['FTHG'])}-{int(r['FTAG'])}",
        })
    return out


def add_rolling_form(df: pd.DataFrame, windows=(5, 10)) -> pd.DataFrame:
    """
    For each match, attach the home & away team's rolling stats from their
    PRIOR matches only (shift(1) before rolling -> no leakage from the
    match we're trying to predict).
    """
    long_df = _team_match_log(df)
    stat_cols = ["GF", "GA", "Points", "Shots", "ShotsOnTarget", "Corners", "Fouls", "Yellows", "Reds"]

    for w in windows:
        grp = long_df.groupby("Team", group_keys=False)
        for col in stat_cols:
            long_df[f"{col}_r{w}"] = grp[col].apply(
                lambda s: s.shift(1).rolling(w, min_periods=1).mean()
            )
        long_df[f"GD_r{w}"] = long_df[f"GF_r{w}"] - long_df[f"GA_r{w}"]

    # overall season-to-date rest (days since last match) as a fatigue proxy
    long_df["DaysSinceLast"] = long_df.groupby("Team")["Date"].diff().dt.days

    keep = ["Date", "Team", "Opponent"] + [c for c in long_df.columns if "_r" in c] + ["DaysSinceLast"]
    home_feats = long_df[long_df["Venue"] == "H"][keep].rename(
        columns={c: f"Home_{c}" for c in keep if c not in ("Date", "Team", "Opponent")}
    ).rename(columns={"Team": "HomeTeam", "Opponent": "AwayTeam"})
    away_feats = long_df[long_df["Venue"] == "A"][keep].rename(
        columns={c: f"Away_{c}" for c in keep if c not in ("Date", "Team", "Opponent")}
    ).rename(columns={"Team": "AwayTeam", "Opponent": "HomeTeam"})

    out = df.merge(home_feats, on=["Date", "HomeTeam", "AwayTeam"], how="left")
    out = out.merge(away_feats, on=["Date", "HomeTeam", "AwayTeam"], how="left")
    return out


# --------------------------------------------------------------------------- #
# 3. Head-to-head
# --------------------------------------------------------------------------- #
def add_h2h(df: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
    df = df.sort_values("Date").reset_index(drop=True)
    h2h_home_pts, h2h_avg_goals = [], []
    history: dict[frozenset, list] = {}

    for _, row in df.iterrows():
        key = frozenset([row["HomeTeam"], row["AwayTeam"]])
        past = history.get(key, [])[-lookback:]
        if past:
            pts = np.mean([
                3 if (m["winner"] == row["HomeTeam"]) else (1 if m["winner"] is None else 0)
                for m in past
            ])
            goals = np.mean([m["total_goals"] for m in past])
        else:
            pts, goals = np.nan, np.nan
        h2h_home_pts.append(pts)
        h2h_avg_goals.append(goals)

        winner = row["HomeTeam"] if row["FTR"] == "H" else (row["AwayTeam"] if row["FTR"] == "A" else None)
        history.setdefault(key, []).append({"winner": winner, "total_goals": row["TotalGoals"]})

    df["H2H_HomePoints_avg"] = h2h_home_pts
    df["H2H_TotalGoals_avg"] = h2h_avg_goals
    return df


# --------------------------------------------------------------------------- #
# 4. Referee profile (walk-forward: only uses the ref's PAST matches)
# --------------------------------------------------------------------------- #
def add_referee_profile(df: pd.DataFrame, min_games: int = 5) -> pd.DataFrame:
    df = df.sort_values("Date").reset_index(drop=True)
    df["Referee"] = df["Referee"].fillna("Unknown").str.strip()

    cards_hist, fouls_hist, reds_hist, games_hist = {}, {}, {}, {}
    ref_cards, ref_fouls, ref_reds, ref_games = [], [], [], []

    for _, row in df.iterrows():
        ref = row["Referee"]
        past_cards = cards_hist.get(ref, [])
        past_fouls = fouls_hist.get(ref, [])
        past_reds = reds_hist.get(ref, [])
        n = len(past_cards)

        ref_games.append(n)
        ref_cards.append(np.mean(past_cards) if n >= min_games else np.nan)
        ref_fouls.append(np.mean(past_fouls) if n >= min_games else np.nan)
        ref_reds.append(np.mean(past_reds) if n >= min_games else np.nan)

        cards_hist.setdefault(ref, []).append(row["TotalCards"])
        fouls_hist.setdefault(ref, []).append(row["TotalFouls"])
        reds_hist.setdefault(ref, []).append(row["HR"] + row["AR"])

    df["Ref_CareerGames"] = ref_games
    df["Ref_AvgCards"] = ref_cards
    df["Ref_AvgFouls"] = ref_fouls
    df["Ref_AvgReds"] = ref_reds

    # league-average fallback for refs with too small a sample (new/rare refs)
    league_avg_cards = df["TotalCards"].expanding().mean().shift(1)
    league_avg_fouls = df["TotalFouls"].expanding().mean().shift(1)
    league_avg_reds = (df["HR"] + df["AR"]).expanding().mean().shift(1)
    df["Ref_AvgCards"] = df["Ref_AvgCards"].fillna(league_avg_cards)
    df["Ref_AvgFouls"] = df["Ref_AvgFouls"].fillna(league_avg_fouls)
    df["Ref_AvgReds"] = df["Ref_AvgReds"].fillna(league_avg_reds)

    # simple tercile label for human-readable reporting ("strict" vs "lenient")
    df["Ref_StrictnessTertile"] = pd.qcut(
        df["Ref_AvgCards"].rank(method="first"), 3, labels=["lenient", "average", "strict"]
    )
    return df


def referee_leaderboard(df: pd.DataFrame, min_games: int = 15) -> pd.DataFrame:
    """Human-readable summary table: one row per referee, career-to-date (latest)."""
    g = df.groupby("Referee").agg(
        games=("TotalCards", "count"),
        avg_cards=("TotalCards", "mean"),
        avg_yellows=("TotalCards", lambda s: np.nan),  # placeholder, filled below
        avg_fouls=("TotalFouls", "mean"),
        avg_reds=("HR", lambda s: s.mean()),
        home_win_rate=("FTR", lambda s: (s == "H").mean()),
    ).reset_index()
    g["avg_reds"] = df.groupby("Referee").apply(lambda x: (x["HR"] + x["AR"]).mean()).values
    g = g[g["games"] >= min_games].sort_values("avg_cards", ascending=False)
    return g.drop(columns=["avg_yellows"])


# --------------------------------------------------------------------------- #
# 5. Squad-strength proxy (see README for the real player-level data caveat)
# --------------------------------------------------------------------------- #
def add_squad_strength_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """
    football-data.co.uk has no lineups, so there is no free bulk source of
    "who actually started" for 25 seasons of history. As a stand-in we use
    each team's own attacking/defensive scoring rate as a rolling proxy for
    current squad quality -- it moves when key players are missing (goals
    dry up, goals conceded rise) even though it can't name the players.
    See README.md "Player-level data" section for how to wire in real
    lineups (API-Football / Understat) if you have access to one.
    """
    df["Home_AttackStrength"] = df["Home_GF_r10"] / df["Home_GF_r10"].mean()
    df["Away_AttackStrength"] = df["Away_GF_r10"] / df["Away_GF_r10"].mean()
    df["Home_DefenseWeakness"] = df["Home_GA_r10"] / df["Home_GA_r10"].mean()
    df["Away_DefenseWeakness"] = df["Away_GA_r10"] / df["Away_GA_r10"].mean()
    return df


# --------------------------------------------------------------------------- #
# 6. Targets
# --------------------------------------------------------------------------- #
GOAL_LINES = (0.5, 1.5, 2.5, 3.5, 4.5)
CARD_LINES = (3.5, 4.5)
CORNER_LINES = (8.5, 9.5, 10.5)


def add_targets(df: pd.DataFrame, goal_lines=GOAL_LINES, card_lines=CARD_LINES, corner_lines=CORNER_LINES) -> pd.DataFrame:
    df["Target_1X2"] = df["FTR"].map(RESULT_MAP)
    for line in goal_lines:
        df[f"Target_Over{line}"] = (df["TotalGoals"] > line).astype(int)
    for line in card_lines:
        df[f"Target_CardsOver{line}"] = (df["TotalCards"] > line).astype(int)
    for line in corner_lines:
        df[f"Target_CornersOver{line}"] = (df["TotalCorners"] > line).astype(int)
    df["Target_BTTS"] = df["BTTS"]
    return df


# Every binary market the site can predict, and which target column/model
# file each one maps to. train.py loops over this to train each market;
# predict.py and app.py loop over it to serve predictions. Add a market by
# adding one line here -- nothing else needs to change.
MARKET_DEFS = (
    [{"key": f"goals_over_{line}", "target_col": f"Target_Over{line}",
      "model_name": f"model_goals_{str(line).replace('.', '_')}",
      "group": "goals", "line": line, "label": f"Over/Under {line} goals"}
     for line in GOAL_LINES]
    + [{"key": f"cards_over_{line}", "target_col": f"Target_CardsOver{line}",
        "model_name": f"model_cards_{str(line).replace('.', '_')}",
        "group": "cards", "line": line, "label": f"Over/Under {line} cards"}
       for line in CARD_LINES]
    + [{"key": f"corners_over_{line}", "target_col": f"Target_CornersOver{line}",
        "model_name": f"model_corners_{str(line).replace('.', '_')}",
        "group": "corners", "line": line, "label": f"Over/Under {line} corners"}
       for line in CORNER_LINES]
    + [{"key": "btts", "target_col": "Target_BTTS", "model_name": "model_btts",
        "group": "btts", "line": None, "label": "Both teams to score"}]
)


# --------------------------------------------------------------------------- #
# Pipeline entry point
# --------------------------------------------------------------------------- #
FEATURE_COLUMNS = None  # populated by build_feature_table()


def build_feature_table(raw_dir: str = RAW_DIR) -> pd.DataFrame:
    global FEATURE_COLUMNS
    df = load_all_seasons(raw_dir)
    df = add_rolling_form(df)
    df = add_h2h(df)
    df = add_referee_profile(df)
    df = add_squad_strength_proxy(df)
    df = add_targets(df)

    feature_cols = (
        [c for c in df.columns if c.startswith("Home_") or c.startswith("Away_")]
        + ["H2H_HomePoints_avg", "H2H_TotalGoals_avg",
           "Ref_AvgCards", "Ref_AvgFouls", "Ref_AvgReds", "Ref_CareerGames"]
    )
    FEATURE_COLUMNS = feature_cols
    return df


if __name__ == "__main__":
    table = build_feature_table()
    print(table.shape)
    print(table[["Date", "HomeTeam", "AwayTeam", "Referee", "Ref_AvgCards", "Target_1X2"]].tail(10))
