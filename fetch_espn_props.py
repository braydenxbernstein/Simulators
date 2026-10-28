"""
DraftKings player prop lines from ESPN's public odds feed. No API key, no credits.

    python fetch_espn_props.py --week 5            # this season, week 5 -> dk_props_espn.csv
    python fetch_espn_props.py --week 5 --season 2026

What you get: DraftKings' main over/under line for passing yards, attempts, completions,
passing TDs, rushing yards, carries, receiving yards and receptions, with opening lines.
What you don't get: prices. ESPN shows DraftKings' lines but not the odds, so both sides
are recorded at -110 (decimal 1.909). Anytime-TD prices aren't in the feed at all.

The output uses the same columns as fetch_dk_props.py, so the notebook reads either file.
Each pull is also appended to dk_props_history.csv for closing-line grading.
"""
import argparse
import csv
import datetime as dt
import json
import os
import urllib.request

from fetch_dk_props import FIELDS, TEAM_NAMES

SITE = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
CORE = "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl"
DK_PROVIDER = "100"            # ESPN's provider id for DraftKings
ASSUMED_PRICE = 1.909          # -110: ESPN doesn't publish DraftKings' prices
ESPN_TEAM_FIX = {"WSH": "WAS", "LAR": "LA", "JAC": "JAX"}
ESPN_MARKETS = {  # ESPN prop type -> The Odds API market key used everywhere else
    "Total Passing Yards (incl. overtime)": "player_pass_yds",
    "Total Passing Attempts (incl. overtime)": "player_pass_attempts",
    "Total Pass Completions (incl. overtime)": "player_pass_completions",
    "Total Passing Touchdowns (incl. overtime)": "player_pass_tds",
    "Total Rushing Yards (incl. overtime)": "player_rush_yds",
    "Total Carries (incl. overtime)": "player_rush_attempts",
    "Total Receiving Yards (incl. overtime)": "player_reception_yds",
    "Total Receptions (incl. overtime)": "player_receptions",
}


class ESPNPropsError(Exception):
    pass


def _get(url):
    last = None
    for ua in (None, "curl/8.5.0"):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": ua} if ua else {})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read())
        except Exception as e:
            last = e
    raise ESPNPropsError(f"ESPN request failed ({last})")


def _athlete_names(season, data_dir="nfl_data"):
    """ESPN athlete id -> name, from the nflverse roster file (no extra requests)."""
    path = os.path.join(data_dir, f"roster_weekly_{season}.parquet")
    if not os.path.exists(path):
        return {}
    import pandas as pd
    r = pd.read_parquet(path, columns=["espn_id", "full_name"]).dropna()
    r["espn_id"] = r.espn_id.astype(str).str.replace(r"\.0$", "", regex=True)
    return dict(zip(r.espn_id, r.full_name))


def fetch_espn_props(week, season, out="dk_props_espn.csv", data_dir="nfl_data", verbose=True):
    """Pull DraftKings' main prop lines for every game in a week from ESPN. Returns rows written."""
    sb = _get(f"{SITE}?week={week}&seasontype=2&dates={season}")
    names = _athlete_names(season, data_dir)
    cache_path = os.path.join(data_dir, "espn_athletes.json")
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    rows, games = [], 0
    for ev in sb.get("events", []):
        comp = ev["competitions"][0]
        if comp.get("status", {}).get("type", {}).get("state") == "post":
            continue   # finished games have no live lines
        teams = {c["homeAway"]: ESPN_TEAM_FIX.get(c["team"]["abbreviation"], c["team"]["abbreviation"])
                 for c in comp["competitors"]}
        home, away = TEAM_NAMES.get(teams.get("home")), TEAM_NAMES.get(teams.get("away"))
        eid = ev["id"]
        try:
            pb = _get(f"{CORE}/events/{eid}/competitions/{eid}/odds/{DK_PROVIDER}/propBets?limit=1000")
        except ESPNPropsError:
            continue
        seen, n = set(), 0
        for it in pb.get("items", []):
            market = ESPN_MARKETS.get(it.get("type", {}).get("name"))
            line = (it.get("current") or {}).get("target", {}).get("value")
            ref = (it.get("athlete") or {}).get("$ref", "")
            if not market or line is None or "/athletes/" not in ref:
                continue
            aid = ref.split("/athletes/")[1].split("?")[0]
            if (aid, market) in seen:     # the feed lists each line twice (over and under)
                continue
            seen.add((aid, market))
            player = names.get(aid) or cache.get(aid)
            if not player:
                try:
                    player = cache[aid] = _get(ref.replace("http://", "https://")).get("displayName", "")
                except ESPNPropsError:
                    continue
            for side in ("Over", "Under"):
                rows.append({"season": season, "week": week, "event_id": f"espn{eid}",
                             "commence_time": ev["date"], "snapshot": now, "home_team": home,
                             "away_team": away, "market": market, "player": player, "name": side,
                             "point": line, "price": ASSUMED_PRICE})
            n += 1
        games += 1
        if verbose and n:
            print(f"  {teams.get('away')} @ {teams.get('home')}: {n} DraftKings lines")
    os.makedirs(data_dir, exist_ok=True)
    json.dump(cache, open(cache_path, "w"))
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    if rows:
        hist = os.path.join(os.path.dirname(out) or ".", "dk_props_history.csv")
        new = not os.path.exists(hist)
        with open(hist, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerows(rows)
    if verbose:
        print(f"Wrote {len(rows) // 2:,} DraftKings lines for {games} games to {out} "
              f"(source: ESPN; prices not included, -110 assumed)")
    return len(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--out", default="dk_props_espn.csv")
    a = ap.parse_args()
    today = dt.date.today()
    season = a.season or (today.year if today.month >= 3 else today.year - 1)
    try:
        fetch_espn_props(a.week, season, a.out)
    except ESPNPropsError as e:
        raise SystemExit(str(e))
