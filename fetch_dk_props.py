"""
Download DraftKings NFL player prop lines from The Odds API (the-odds-api.com).

Setup (once):
  1. Get an API key at https://the-odds-api.com  (free plan: 500 credits/month, current
     lines only; historical lines need a paid plan).
  2. Save the key where only your computer can see it (never in code or chat). Either:
        a) a file named odds_api_key.txt in this folder containing just the key, or
        b) an environment variable:  export ODDS_API_KEY=your_key_here  (Mac/Linux)
     Don't share or upload odds_api_key.txt.

Usage:
  python fetch_dk_props.py current                    # this week's lines -> dk_props_current.csv
  python fetch_dk_props.py current --teams PHI LA     # just one game (~6 credits)
  python fetch_dk_props.py history 2025 --weeks 4-18  # past lines -> dk_props_2025.csv (paid plan)
  python fetch_dk_props.py history 2025 --weeks 4-18 --dry-run   # show credit cost, download nothing

Credit cost (from The Odds API docs):
  current:  1 credit per market per game            (6 markets x 16 games ~ 100 credits/week)
  history: 10 credits per market per game + 1 per events lookup (~60 per game; ~16,000 per season)
"""

import argparse
import csv
import datetime as dt
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from zoneinfo import ZoneInfo

API = "https://api.the-odds-api.com/v4"
SPORT = "americanfootball_nfl"
MARKETS = ["player_pass_yds", "player_rush_yds", "player_reception_yds", "player_receptions",
           "player_pass_tds", "player_anytime_td"]
GAMES_CSV = os.path.join("nfl_data", "games.csv")  # nflverse schedule, downloaded by the notebook
FIELDS = ["season", "week", "event_id", "commence_time", "snapshot", "home_team", "away_team",
          "market", "player", "name", "point", "price"]
TEAM_NAMES = {
    "ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons", "BAL": "Baltimore Ravens", "BUF": "Buffalo Bills",
    "CAR": "Carolina Panthers", "CHI": "Chicago Bears", "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns",
    "DAL": "Dallas Cowboys", "DEN": "Denver Broncos", "DET": "Detroit Lions", "GB": "Green Bay Packers",
    "HOU": "Houston Texans", "IND": "Indianapolis Colts", "JAX": "Jacksonville Jaguars", "KC": "Kansas City Chiefs",
    "LV": "Las Vegas Raiders", "LAC": "Los Angeles Chargers", "LA": "Los Angeles Rams", "MIA": "Miami Dolphins",
    "MIN": "Minnesota Vikings", "NE": "New England Patriots", "NO": "New Orleans Saints", "NYG": "New York Giants",
    "NYJ": "New York Jets", "PHI": "Philadelphia Eagles", "PIT": "Pittsburgh Steelers", "SF": "San Francisco 49ers",
    "SEA": "Seattle Seahawks", "TB": "Tampa Bay Buccaneers", "TEN": "Tennessee Titans", "WAS": "Washington Commanders"}


class OddsAPIError(Exception):
    pass


def api_key():
    key = os.environ.get("ODDS_API_KEY", "").strip()
    if not key and os.path.exists("odds_api_key.txt"):
        key = open("odds_api_key.txt").read().strip()
    if not key:
        raise OddsAPIError("No Odds API key found. Put it in odds_api_key.txt next to the notebook "
                           "(just the key, nothing else) or set ODDS_API_KEY.")
    return key


def get(path, **params):
    key = api_key()
    params["apiKey"] = key
    url = f"{API}{path}?{urllib.parse.urlencode(params)}"
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                left = r.headers.get("x-requests-remaining")
                return json.loads(r.read()), left
        except urllib.error.HTTPError as e:
            if e.code == 429:  # rate limited: wait and retry
                time.sleep(2 + attempt * 3)
                continue
            body = e.read().decode()[:300]
            if e.code in (401, 403):
                raise OddsAPIError(f"API key rejected or plan doesn't include this ({e.code}): {body}")
            print(f"  skipped ({e.code}): {body}")
            return None, None
    return None, None


def rows_from_event(ev, season, week, snapshot):
    out = []
    for bk in ev.get("bookmakers", []):
        if bk.get("key") != "draftkings":
            continue
        for mk in bk.get("markets", []):
            for o in mk.get("outcomes", []):
                out.append({"season": season, "week": week, "event_id": ev["id"],
                            "commence_time": ev["commence_time"], "snapshot": snapshot,
                            "home_team": ev["home_team"], "away_team": ev["away_team"],
                            "market": mk["key"], "player": o.get("description", ""),
                            "name": o.get("name", ""), "point": o.get("point", ""),
                            "price": o.get("price", "")})
    return out


def load_schedule():
    if not os.path.exists(GAMES_CSV):
        raise OddsAPIError(f"{GAMES_CSV} not found. Run the notebook once first (it downloads the schedule).")
    out = []
    with open(GAMES_CSV) as f:
        for g in csv.DictReader(f):
            t = g["gametime"] or "13:00"
            kick = dt.datetime.strptime(f"{g['gameday']} {t}", "%Y-%m-%d %H:%M").replace(
                tzinfo=ZoneInfo("America/New_York")).astimezone(dt.timezone.utc)
            out.append({"season": int(g["season"]), "week": int(g["week"]), "game_type": g["game_type"],
                        "home_team": g["home_team"], "away_team": g["away_team"], "kick_utc": kick})
    return out


def fetch_current(markets=MARKETS, out="dk_props_current.csv", teams=None, verbose=True):
    """Download DraftKings' current lines for upcoming games (next 8 days, not yet started).
    teams: optional list of abbreviations (e.g. ["PHI", "LA"]) to fetch only those games,
    ~1 credit per market per game. Rows for other games already in `out` are kept."""
    evs, left = get(f"/sports/{SPORT}/events")  # the events list itself costs no credits
    if evs is None:
        return 0
    sched = load_schedule()
    abbr = {v: k for k, v in TEAM_NAMES.items()}
    now = dt.datetime.now(dt.timezone.utc)
    rows, fetched = [], set()
    for ev in evs:
        start = dt.datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00"))
        if start < now or start > now + dt.timedelta(days=8):
            continue
        if teams and not ({abbr.get(ev["home_team"]), abbr.get(ev["away_team"])} & set(teams)):
            continue
        g = next((s for s in sched if TEAM_NAMES.get(s["home_team"]) == ev["home_team"]
                  and abs((s["kick_utc"] - start).total_seconds()) < 36 * 3600), None)
        data, left = get(f"/sports/{SPORT}/events/{ev['id']}/odds", regions="us", bookmakers="draftkings",
                         markets=",".join(markets), oddsFormat="decimal")
        if data:
            fetched.add(ev["id"])
            rows += rows_from_event(data, g["season"] if g else "", g["week"] if g else "",
                                    now.isoformat(timespec="seconds"))
            if verbose:
                print(f"  {ev['away_team']} @ {ev['home_team']}: DraftKings lines pulled (credits left {left})")
    # keep other upcoming games already saved
    if os.path.exists(out):
        with open(out) as f:
            for r in csv.DictReader(f):
                start = dt.datetime.fromisoformat(r["commence_time"].replace("Z", "+00:00")) if r["commence_time"] else now
                if r["event_id"] not in fetched and start > now:
                    rows.append(r)
    write(rows, out, verbose)
    # every pull is also appended to a history file: the last snapshot before kickoff is
    # the "closing" line used to grade the prediction log (closing line value)
    fresh = [r for r in rows if r.get("event_id") in fetched]
    if fresh:
        hist = os.path.join(os.path.dirname(out) or ".", "dk_props_history.csv")
        new = not os.path.exists(hist)
        with open(hist, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerows(fresh)
    return len(rows)


def current(args):
    fetch_current(args.markets, args.out or "dk_props_current.csv", args.teams)


def history(args):
    lo, hi = (int(x) for x in args.weeks.split("-")) if args.weeks else (1, 22)
    sched = [g for g in load_schedule() if g["season"] == args.season and lo <= g["week"] <= hi]
    if not sched:
        sys.exit("No games found for that season/weeks in the schedule.")
    days = sorted({g["kick_utc"].date() for g in sched})
    est = len(days) + 10 * len(args.markets) * len(sched)
    print(f"{len(sched)} games on {len(days)} dates. Estimated cost: up to {est:,} credits.")
    if args.dry_run:
        return
    rows = []
    for day in days:
        games_today = [g for g in sched if g["kick_utc"].date() == day]
        first_kick = min(g["kick_utc"] for g in games_today)
        snap = (first_kick - dt.timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
        evs, left = get(f"/historical/sports/{SPORT}/events", date=snap,
                        commenceTimeFrom=(first_kick - dt.timedelta(hours=3)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                        commenceTimeTo=(first_kick + dt.timedelta(hours=20)).strftime("%Y-%m-%dT%H:%M:%SZ"))
        for ev in (evs or {}).get("data", []):
            start = dt.datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00"))
            g = next((s for s in games_today if TEAM_NAMES.get(s["home_team"]) == ev["home_team"]), None)
            if g is None:
                continue
            # lines about 30 minutes before kickoff (close to the closing number)
            at = (start - dt.timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
            data, left = get(f"/historical/sports/{SPORT}/events/{ev['id']}/odds", regions="us",
                             bookmakers="draftkings", markets=",".join(args.markets),
                             oddsFormat="decimal", date=at)
            if data and data.get("data"):
                rows += rows_from_event(data["data"], g["season"], g["week"], data.get("timestamp", at))
                print(f"  wk {g['week']} {ev['away_team']} @ {ev['home_team']}: credits left {left}")
    write(rows, args.out or f"dk_props_{args.season}.csv")


def write(rows, path, verbose=True):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    if verbose:
        print(f"Wrote {len(rows):,} rows to {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    c = sub.add_parser("current")
    c.add_argument("--markets", nargs="+", default=MARKETS)
    c.add_argument("--out")
    c.add_argument("--teams", nargs="+", help="only these teams' games, e.g. --teams PHI LA")
    h = sub.add_parser("history")
    h.add_argument("season", type=int)
    h.add_argument("--weeks", help="e.g. 4-18")
    h.add_argument("--markets", nargs="+", default=MARKETS)
    h.add_argument("--dry-run", action="store_true")
    h.add_argument("--out")
    a = ap.parse_args()
    try:
        current(a) if a.mode == "current" else history(a)
    except OddsAPIError as e:
        sys.exit(str(e))
