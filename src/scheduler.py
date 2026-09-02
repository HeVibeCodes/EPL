"""
Optional: run this as a long-lived background process to keep data/models
fresh automatically, instead of using cron/Task Scheduler.

    python src/scheduler.py                  # checks every 6 hours (default)
    python src/scheduler.py --hours 12        # checks every 12 hours
    python src/scheduler.py --once            # single check then exit

For a "set and forget" deployment, cron or Task Scheduler calling
update_data.py directly is usually more robust than a long-lived Python
process (it survives reboots without extra setup) -- see README.md for
the exact crontab / Task Scheduler entries. Use this script if you'd
rather not touch cron at all.
"""
import argparse
import time
from datetime import datetime

from update_data import refresh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=float, default=6, help="hours between checks")
    parser.add_argument("--once", action="store_true", help="check once and exit")
    args = parser.parse_args()

    while True:
        print(f"[{datetime.now().isoformat(timespec='seconds')}] checking for new results...")
        try:
            refresh()
        except Exception as exc:
            print(f"  update failed (will retry next cycle): {exc}")
        if args.once:
            break
        time.sleep(args.hours * 3600)


if __name__ == "__main__":
    main()
