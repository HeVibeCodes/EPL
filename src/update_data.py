"""
Keeps the dataset (and therefore the models) up to date.

football-data.co.uk updates its CURRENT-season file roughly twice a week
during the season (usually after midweek and weekend fixtures). Past-season
files never change. So "staying up to date" only ever means re-pulling the
one or two most recent season files -- there's no need to re-download the
full 25-season history every time.

What this script does:
  1. Works out which season code is "current" from today's date.
  2. Re-downloads that season's file (and last season's, in case results
     were corrected) and overwrites the local copy if the content changed.
  3. If anything changed, re-runs train.py so the models reflect the new
     matches immediately.

Run this on a schedule (cron / Task Scheduler / the bundled scheduler.py)
so the site always reflects the latest results:
    python src/update_data.py
"""
import hashlib
import os
import subprocess
import sys
import urllib.request
from datetime import date

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
BASE_URL = "https://www.football-data.co.uk/mmz4281/{code}/E0.csv"


def current_season_code(today: date | None = None) -> str:
    """EPL season runs Aug -> May. Aug 2025 - May 2026 is season code '2526'."""
    today = today or date.today()
    start_year = today.year if today.month >= 7 else today.year - 1
    a, b = start_year % 100, (start_year + 1) % 100
    return f"{a:02d}{b:02d}"


def previous_season_code(code: str) -> str:
    start = int(code[:2])
    prev_start = start - 1
    a, b = prev_start % 100, start
    return f"{a:02d}{b:02d}"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch_season(code: str) -> bytes | None:
    url = BASE_URL.format(code=code)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = resp.read()
        if len(data) < 200:
            return None
        return data
    except Exception as exc:
        print(f"  could not fetch season {code}: {exc}")
        return None


def refresh(retrain_if_changed: bool = True) -> bool:
    """Returns True if any local file was updated."""
    cur = current_season_code()
    codes_to_check = [previous_season_code(cur), cur]

    changed = False
    for code in codes_to_check:
        data = fetch_season(code)
        if data is None:
            continue
        out_path = os.path.join(RAW_DIR, f"season-{code}.csv")
        new_hash = _sha256(data)
        old_hash = None
        if os.path.exists(out_path):
            with open(out_path, "rb") as f:
                old_hash = _sha256(f.read())
        if new_hash != old_hash:
            os.makedirs(RAW_DIR, exist_ok=True)
            with open(out_path, "wb") as f:
                f.write(data)
            print(f"  season {code}: updated ({len(data) / 1024:.1f} kB)")
            changed = True
        else:
            print(f"  season {code}: no change")

    if changed and retrain_if_changed:
        print("Data changed -> retraining models...")
        subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "train.py")], check=True)
    elif not changed:
        print("No new results since last check -- models are already current.")

    return changed


if __name__ == "__main__":
    refresh()
