"""
One command for a day's slate:   python run_day.py                (today)
                                 python run_day.py --date 2026-10-06 --sims 10000
Grade the log:                   python run_day.py --grade
Backtest:                        python run_day.py --backtest 2026-09-01 2026-10-04
"""
import argparse

import mlb_daily as d

ap = argparse.ArgumentParser()
ap.add_argument("--date", default=None)
ap.add_argument("--sims", type=int, default=6000)
ap.add_argument("--no-calibrate", action="store_true")
ap.add_argument("--grade", action="store_true")
ap.add_argument("--backtest", nargs=2, metavar=("START", "END"))
a = ap.parse_args()

if a.grade:
    d.grade_log()
elif a.backtest:
    d.backtest(a.backtest[0], a.backtest[1], sims=min(a.sims, 3000), calibrate=not a.no_calibrate)
else:
    d.daily_report(a.date, sims=a.sims, calibrate=not a.no_calibrate)
