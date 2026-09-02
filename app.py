"""
Website for the EPL predictor.

    pip install -r requirements.txt
    python app.py

Then open http://127.0.0.1:5000

The app loads the feature table and models once at startup and keeps them
in memory. Hitting "Check for new results" on the site (or POSTing to
/api/refresh) re-pulls the current season, retrains if anything changed,
and reloads the in-memory table -- no restart needed.
"""
import os
import sys
import threading
from datetime import datetime

from flask import Flask, jsonify, render_template, request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import update_data  # noqa: E402
from features import build_feature_table, referee_leaderboard  # noqa: E402
from predict import predict_fixture  # noqa: E402

app = Flask(__name__)

STATE = {"df": None, "teams": [], "referees": [], "last_updated": None, "last_checked": None}
_lock = threading.Lock()


def load_state():
    df = build_feature_table()
    teams = sorted(set(df["HomeTeam"].dropna()) | set(df["AwayTeam"].dropna()))
    referees = sorted(df["Referee"].dropna().unique().tolist())
    STATE["df"] = df
    STATE["teams"] = teams
    STATE["referees"] = referees
    STATE["last_updated"] = datetime.now()


load_state()


@app.route("/")
def index():
    return render_template(
        "index.html",
        teams=STATE["teams"],
        referees=STATE["referees"],
        last_updated=STATE["last_updated"],
        n_matches=len(STATE["df"]),
        n_seasons=STATE["df"]["season"].nunique(),
    )


@app.route("/referees")
def referees_page():
    board = referee_leaderboard(STATE["df"], min_games=5)
    rows = board.round(2).to_dict(orient="records")
    return render_template("referees.html", rows=rows, last_updated=STATE["last_updated"])


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True) or {}
    home, away, ref = data.get("home"), data.get("away"), data.get("referee")
    if not home or not away or not ref:
        return jsonify({"error": "Pick a home team, an away team, and a referee."}), 400
    if home == away:
        return jsonify({"error": "Home and away team can't be the same."}), 400
    try:
        result = predict_fixture(home, away, ref, df=STATE["df"])
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(result)


@app.route("/api/refresh", methods=["POST"])
def api_refresh():
    with _lock:
        changed = update_data.refresh(retrain_if_changed=True)
        if changed:
            load_state()
        STATE["last_checked"] = datetime.now()
    return jsonify({
        "changed": changed,
        "last_updated": STATE["last_updated"].strftime("%d %b %Y, %H:%M"),
        "last_checked": STATE["last_checked"].strftime("%d %b %Y, %H:%M"),
    })


@app.route("/api/status")
def api_status():
    return jsonify({
        "last_updated": STATE["last_updated"].strftime("%d %b %Y, %H:%M") if STATE["last_updated"] else None,
        "matches": len(STATE["df"]),
        "seasons": int(STATE["df"]["season"].nunique()),
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("FLASK_DEBUG") == "1")
