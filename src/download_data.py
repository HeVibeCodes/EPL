"""
Downloads EPL match data (results, referees, cards, fouls, shots, corners)
for every season with full statistics: 2000/01 -> present.

Source: football-data.co.uk (free, public, updated ~twice weekly in-season).
Referee / cards / fouls / shots columns only exist from the 2000/01 season
onward, so that's the earliest season this script pulls.

Run this on a machine with normal internet access:
    python src/download_data.py

It writes one CSV per season into data/raw/season-XXXX.csv
"""
import os
import time
import urllib.request

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
os.makedirs(OUT_DIR, exist_ok=True)

BASE = "https://www.football-data.co.uk/mmz4281/{code}/E0.csv"


def season_codes(start_year: int = 2000, end_year: int = 2026):
    """2000 -> '0001', 2001 -> '0102', ... 2025 -> '2526'."""
    codes = []
    for y in range(start_year, end_year + 1):
        a, b = y % 100, (y + 1) % 100
        codes.append(f"{a:02d}{b:02d}")
    return codes


def download(overwrite: bool = False):
    ok, failed = [], []
    for code in season_codes():
        out_path = os.path.join(OUT_DIR, f"season-{code}.csv")
        if os.path.exists(out_path) and not overwrite:
            ok.append(code)
            continue
        url = BASE.format(code=code)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = resp.read()
            if len(data) < 200:  # season not published yet / bad response
                raise ValueError("response too small, likely no data yet")
            with open(out_path, "wb") as f:
                f.write(data)
            print(f"  saved {code} ({len(data) / 1024:.1f} kB)")
            ok.append(code)
        except Exception as exc:
            print(f"  FAILED {code}: {exc}")
            failed.append(code)
        time.sleep(0.5)  # be polite to the server
    print(f"\nDone. {len(ok)} seasons saved, {len(failed)} failed.")
    if failed:
        print("Failed codes (often just means the season hasn't started yet):", failed)


if __name__ == "__main__":
    download()
