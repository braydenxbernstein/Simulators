"""
FanDuel player prop lines AND prices from FanDuel's public sportsbook feed. No API key.

    python fetch_fanduel_props.py --week 5            # -> fd_props_current.csv
    python fetch_fanduel_props.py --week 5 --state az

What you get: FanDuel's main over/under line and real odds for passing yards, passing TDs,
rushing yards, receiving yards and receptions, plus anytime-TD prices.
What you don't get: pass attempts, completions and carries (FanDuel doesn't post a main
line for those on these pages).

This reads the same public feed FanDuel's own website loads in your browser. It isn't an
official API, so FanDuel can change or block it without notice; keep pulls to a few per day.
The output uses the same columns as fetch_dk_props.py, so the notebook reads either file.
"""
import argparse
import csv
import datetime as dt
import json
import os
import re
import time
import urllib.request

from fetch_dk_props import FIELDS, TEAM_NAMES, load_schedule

STATE = "az"
AK = "FhMFpcPWXMeyZxOx"   # FanDuel's public web-app key (the same one their site sends)
NFL_COMPETITION = 12282733
TABS = ("passing-props", "receiving-props", "rushing-props", "td-scorer-props")
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"
# main-line market types only (FanDuel's ALT_ markets are alternate lines)
FD_MARKETS = [
    (r"^PLAYER_X_PASSING_YARDS(_LOW|_MEDIUM|_HIGH)?$", "player_pass_yds"),
    (r"^PLAYER_X_PASSING_TOUCHDOWNS(_LOW|_MEDIUM|_HIGH)?$", "player_pass_tds"),
    (r"^PLAYER_X_PASSING_ATTEMPTS(_LOW|_MEDIUM|_HIGH)?$", "player_pass_attempts"),
    (r"^PLAYER_X_(PASSING|PASS)_COMPLETIONS(_LOW|_MEDIUM|_HIGH)?$", "player_pass_completions"),
    (r"^PLAYER_X_RUSHING_YARDS(_LOW|_MEDIUM|_HIGH)?$", "player_rush_yds"),
    (r"^PLAYER_X_RUSHING_ATTEMPTS(_LOW|_MEDIUM|_HIGH)?$", "player_rush_attempts"),
    (r"^PLAYER_X_RECEIVING_YARDS(_LOW|_MEDIUM|_HIGH)?$", "player_reception_yds"),
    (r"^PLAYER_X_RECEPTIONS(_LOW|_MEDIUM|_HIGH)?$", "player_receptions"),
    (r"^ANY_TIME_TOUCHDOWN_SCORER$", "player_anytime_td"),
]


class FanDuelError(Exception):
    pass


def _get(url):
    last = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read())
        except Exception as e:
            last = e
            time.sleep(1 + attempt)
    raise FanDuelError(f"FanDuel request failed ({last})")


def _market(mtype):
    for pat, key in FD_MARKETS:
        if re.match(pat, mtype or ""):
            return key
    return None


def fetch_fanduel_props(week, season, out="fd_props_current.csv", state=STATE, verbose=True):
    """FanDuel's main prop lines and prices for every game in `week` that hasn't started."""
    base = f"https://sbapi.{state}.sportsbook.fanduel.com/api"
    page = _get(f"{base}/content-managed-page?page=CUSTOM&customPageId=nfl&_ak={AK}&timezone=America%2FPhoenix")
    events = page.get("attachments", {}).get("events", {}).values()
    sched = [g for g in load_schedule() if g["season"] == season and g["week"] == week]
    full_to_abbr = {v: k for k, v in TEAM_NAMES.items()}
    now = dt.datetime.now(dt.timezone.utc)
    rows, n_games = [], 0
    for ev in events:
        if ev.get("competitionId") != NFL_COMPETITION or " @ " not in ev.get("name", ""):
            continue
        away_full, home_full = [x.strip() for x in ev["name"].split(" @ ")]
        start = dt.datetime.fromisoformat(ev["openDate"].replace("Z", "+00:00"))
        if start < now:
            continue
        g = next((s for s in sched if TEAM_NAMES.get(s["home_team"]) == home_full
                  and abs((s["kick_utc"] - start).total_seconds()) < 36 * 3600), None)
        if g is None:   # neutral-site games can be listed with home/away swapped
            g = next((s for s in sched if {TEAM_NAMES.get(s["home_team"]), TEAM_NAMES.get(s["away_team"])}
                      == {home_full, away_full}), None)
            if g is not None:
                home_full, away_full = TEAM_NAMES[g["home_team"]], TEAM_NAMES[g["away_team"]]
        if g is None:
            continue     # not this week's game
        n = 0
        for tab in TABS:
            try:
                d = _get(f"{base}/event-page?_ak={AK}&eventId={ev['eventId']}&tab={tab}"
                         f"&useCombinedTouchdownsVirtualMarket=true&usePulse=true&useQuickBets=true")
            except FanDuelError:
                continue
            for m in d.get("attachments", {}).get("markets", {}).values():
                key = _market(m.get("marketType"))
                if not key or m.get("marketStatus") != "OPEN":
                    continue
                base_row = {"season": season, "week": week, "event_id": f"fd{ev['eventId']}",
                            "commence_time": ev["openDate"], "snapshot": now.isoformat(timespec="seconds"),
                            "home_team": home_full, "away_team": away_full, "market": key}
                for r in m.get("runners", []):
                    if r.get("runnerStatus") != "ACTIVE":
                        continue
                    price = (r.get("winRunnerOdds", {}).get("trueOdds", {}) or {}).get("decimalOdds")
                    if isinstance(price, dict):   # {"decimalOdds": 2.62}
                        price = price.get("decimalOdds")
                    if not price:
                        continue
                    if key == "player_anytime_td":
                        rows.append({**base_row, "player": r["runnerName"], "name": "Yes", "point": "",
                                     "price": round(price, 4)})
                    else:
                        side = "Over" if r["runnerName"].endswith("Over") else "Under" if r["runnerName"].endswith("Under") else None
                        if side is None:
                            continue
                        player = m.get("marketName", "").split(" - ")[0].strip()
                        rows.append({**base_row, "player": player, "name": side, "point": r.get("handicap"),
                                     "price": round(price, 4)})
                    n += 1
            time.sleep(0.3)
        n_games += 1
        if verbose:
            print(f"  {full_to_abbr.get(away_full)} @ {full_to_abbr.get(home_full)}: {n} FanDuel prices")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    if rows:
        hist = os.path.join(os.path.dirname(out) or ".", "fd_props_history.csv")
        new = not os.path.exists(hist)
        with open(hist, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerows(rows)
    if verbose:
        print(f"Wrote {len(rows):,} FanDuel prices for {n_games} games to {out}")
    return len(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--state", default=STATE, help="FanDuel state site, e.g. az, nj, pa (default az)")
    ap.add_argument("--out", default="fd_props_current.csv")
    a = ap.parse_args()
    today = dt.date.today()
    season = a.season or (today.year if today.month >= 3 else today.year - 1)
    try:
        fetch_fanduel_props(a.week, season, a.out, a.state)
    except FanDuelError as e:
        raise SystemExit(str(e))
