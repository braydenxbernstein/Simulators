"""
Run the whole week from a terminal:

    python run_week.py --week 5            # refresh data, simulate, pull DraftKings, log, export
    python run_week.py --week 5 --no-dk    # skip DraftKings
    python run_week.py --week 5 --lines fanduel   # free FanDuel lines + prices (no API key)
    python run_week.py --week 5 --lines espn      # free DraftKings lines from ESPN (no prices)
    python run_week.py --grade             # grade logged predictions (results + closing line value)

Outputs land in reports/ (Excel + HTML dashboard) and logs/ (prediction log).
"""
import argparse
import warnings

warnings.filterwarnings("ignore")

from nfl_weekly_v6 import current_season, grade_log, weekly_report

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--week", type=int, help="NFL week to run")
    ap.add_argument("--season", type=int, default=None, help="default: current season")
    ap.add_argument("--sims", type=int, default=10000, help="simulations per game (default 10,000)")
    ap.add_argument("--no-dk", action="store_true", help="don't pull DraftKings lines")
    ap.add_argument("--lines", choices=["auto", "fanduel", "espn", "odds_api"], default="auto",
                    help="where prop lines come from (default auto: Odds API if a key is set up, else FanDuel, else ESPN)")
    ap.add_argument("--no-refresh", action="store_true", help="don't re-download this season's data")
    ap.add_argument("--no-log", action="store_true", help="don't add these predictions to the log")
    ap.add_argument("--grade", action="store_true", help="grade the prediction log and exit")
    a = ap.parse_args()
    if a.grade:
        grade_log(season=a.season)
    elif a.week is None:
        ap.error("--week is required (or use --grade)")
    else:
        weekly_report(week=a.week, season=a.season or current_season(), n_sims=a.sims,
                      refresh=not a.no_refresh, pull_dk=not a.no_dk, log=not a.no_log, lines=a.lines)
