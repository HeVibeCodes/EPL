# EPL Match Predictor

Predicts, for any Premier League fixture:
- **1X2**: Home win / Draw / Away win probabilities
- **Goals market**: Over/Under 2.5 total goals (thresholds 1.5/2.5/3.5 all built in)
- **Cards market**: Over/Under 3.5 total cards, driven by the referee's career profile
- Everything is conditioned on **team form** and **the specific referee assigned**

Built on free, public data from [football-data.co.uk](https://www.football-data.co.uk/),
which has covered referee names, fouls, cards, shots and corners for every
Premier League match since the **2000/01 season**.

## Quick start

```bash
pip install -r requirements.txt

# 1. Download full history (2000/01 -> present, ~26 seasons)
python src/download_data.py

# 2. Build features, train all 3 models, print evaluation metrics
python src/train.py

# 3a. Predict a fixture from the console
python src/predict.py "Arsenal" "Chelsea" "A Taylor"

# 3b. ...or launch the website and predict from a browser
python app.py
# then open http://127.0.0.1:5000

# 4. See which referees are strict/lenient
python -c "from src.features import build_feature_table, referee_leaderboard; \
  print(referee_leaderboard(build_feature_table()).to_string(index=False))"
```

## Keeping the data up to date

football-data.co.uk updates the current season's file roughly twice a week
in-season (after midweek and weekend rounds). `src/update_data.py` re-pulls
just the current (and previous, in case of corrections) season file, and
**only if something actually changed**, retrains the models automatically.
The website's "Check for new results" button calls this directly, so you
can refresh on demand without touching the command line.

For it to happen automatically without you visiting the site, schedule it:

**Linux/macOS (cron)** &mdash; refresh every 6 hours:
```
0 */6 * * * cd /path/to/epl_predictor && /usr/bin/python3 src/update_data.py >> logs/update.log 2>&1
```

**Windows (Task Scheduler)**: create a Basic Task that runs
`python.exe src\update_data.py` from the project folder, trigger "Daily",
repeat every 6 hours.

**No cron access?** Run the bundled scheduler as a long-lived process
instead (e.g. inside `tmux`/`screen`, or as a systemd service):
```bash
python src/scheduler.py --hours 6
```

If you deploy `app.py` on a server (see "Deploying the website" below), the
scheduler process should run alongside it so the site's in-memory data stays
current between restarts too -- `app.py` itself only reloads when a request
hits `/api/refresh`.

## The website

`app.py` is a small Flask app:
- `/` &mdash; pick a home team, away team, and referee, get win/draw/loss
  probabilities plus the goals and cards lines, styled as a matchday ticket.
- `/referees` &mdash; the full referee leaderboard (cards, fouls, reds per
  game, home win rate under that ref).
- `/api/predict`, `/api/refresh`, `/api/status` &mdash; JSON endpoints the
  pages call; also usable directly if you want to build another frontend
  against the same models.

Run it locally with `python app.py` (defaults to port 5000, override with
the `PORT` env var). It loads the feature table and models into memory once
at startup and keeps them there; predictions are near-instant.

### Deploying the website

The built-in Flask server is for local use only. To put it somewhere
reachable by other people, run it behind a production WSGI server, e.g.:
```bash
pip install gunicorn
gunicorn -w 2 -b 0.0.0.0:8000 app:app
```
then put nginx or Caddy in front of it for TLS. Any host that runs a
Python process works (a small VPS, Render, Railway, Fly.io, PythonAnywhere,
etc.) -- there's nothing here that needs a specific provider. Pair it with
the cron/`scheduler.py` job above so the deployed copy keeps itself current
without you redeploying.

## What's already included in this delivery

This sandbox has network restrictions that block direct access to
football-data.co.uk, so I couldn't run `download_data.py` here. Instead I
pulled the 2025/26 season directly via web search/fetch and saved it to
`data/raw/season-2526.csv` so the whole pipeline is proven working end-to-end
on **real data** (see the metrics below). Everything is unit-tested against
that season; run `download_data.py` on your own machine (normal, unrestricted
internet) to pull the complete 2000/01-2025/26 history and get a much
stronger model -- one season alone (380 matches) is a thin sample for a
20+ team league.

Sample output from the current single-season fit (train.py):
```
1X2:            accuracy 44.7%  (baseline "always predict home win" = 41.8%)
Over/Under 2.5: accuracy 60.5%, AUC 0.667
Cards Over 3.5: accuracy 56.6%, AUC 0.488   <- needs more data, see below
```
The cards model in particular needs many more seasons per referee before its
AUC beats a coin flip -- with only ~1 season, most referees have fewer than
30 games on file, which is why `add_referee_profile()` backs off to the
league average until a referee passes `min_games` (default 5, tune this up
once you have full history).

## Project layout

```
epl_predictor/
  app.py                      Flask website (fixture predictor + referee leaderboard)
  templates/, static/         website HTML/CSS
  data/raw/season-XXXX.csv   one file per EPL season (Referee, HS/AS, HF/AF, HY/AY, HR/AR, ...)
  src/download_data.py       pulls every season 2000/01-present from football-data.co.uk
  src/update_data.py          incremental refresh: re-pulls only the current season, retrains if changed
  src/scheduler.py            optional background loop that calls update_data.py on a timer
  src/features.py            feature engineering (see below)
  src/train.py                trains & evaluates 3 model heads, chronological split
  src/predict.py              predicts a single fixture from the console
  models/                     saved model files + metrics.json (created after train.py)
```

## Feature groups (src/features.py)

1. **Rolling team form** -- each team's points, goals for/against, shots,
   shots on target, corners, fouls, cards over their last 5 and 10 matches.
   Computed with `.shift(1)` before the rolling window so a match never sees
   its own result (no leakage).
2. **Head-to-head** -- average points and total goals from the last 5
   meetings between the exact two teams.
3. **Referee profile** -- for every match, the referee's **career-to-date**
   average total cards, fouls, and red cards, computed walk-forward (only
   using games that referee had already worked before this one). This is
   what answers "does this ref book a lot of cards" -- e.g. in the one
   season on file, Attwell averages 4.68 cards/game vs Pawson at 2.76.
4. **Squad-strength proxy** -- see the "Player-level data" section below;
   this is the one part of your original ask I could only approximate.
5. **Targets** -- match result, over/under at 1.5/2.5/3.5 goals, over/under
   at 3.5/4.5 total cards.

## Player-level data: an important caveat

You asked for the model to account for **which players are playing**. I
looked for a free, bulk-downloadable source of historical EPL lineups to
match football-data.co.uk's 25 seasons of results and came up short:
football-data.co.uk itself has no lineup data, and the sites that do
(FBref, Understat) don't offer bulk CSV downloads for historical lineups --
you'd need to scrape them match-by-match (thousands of page loads for 25
seasons) or use a paid API (API-Football, Opta, StatsBomb).

**What's in the pipeline today instead:** `add_squad_strength_proxy()` uses
each team's own rolling goals-for/goals-against as a stand-in for current
squad quality. It's not nothing -- it does move when a team's key attacker
is injured, because their scoring rate drops -- but it can't say "Salah is
out" directly.

**If you want real lineup-level features**, the cleanest path is:
1. Sign up for a free tier of [API-Football](https://www.api-football.com/)
   (100 requests/day free) or use [Understat](https://understat.com/) which
   has player-level xG per match, scrapable with `understatapi` on PyPI.
2. For each historical match, pull the starting XI and a per-player rating
   (or just minutes played by your top-6 rated players) and join it onto
   `data/raw/season-XXXX.csv` on Date + HomeTeam + AwayTeam.
3. Add a feature like `Home_KeyPlayersAvailable` (count of your top-N rated
   players who started) to `features.py` and it'll flow straight through to
   the models -- no other code changes needed.
I'm happy to build that join logic once you've picked a lineup source and
have API access, since I can't reach these providers from this sandbox to
build/test it for you right now.

## Notes on validity

- The 1X2 and goals models are calibrated (Platt/isotonic) so the output
  percentages are genuine probabilities, not just class scores -- useful if
  you want to compare them against bookmaker odds later.
- Train/test split is **chronological** (earliest matches train, most
  recent season tests) specifically so form/table-position drift across a
  season doesn't leak into training -- a random shuffle split would
  overstate accuracy.
- Re-run `train.py` periodically (e.g. weekly in-season) so rolling form
  and referee profiles stay current.
