"""
MLB data layer: everything the simulator reads, cached in mlb_data/.

Sources
  MLB Stats API (statsapi.mlb.com)   schedule, scores, probable pitchers, lineups, umpires,
                                     weather at first pitch, boxscores, platoon splits, handedness
  Baseball Savant                    outs above average, catcher framing, sprint speed, xwOBA
  Open-Meteo                         game-time temperature and wind forecast for upcoming games
  ESPN core API                      DraftKings game lines (open / close / current) and DraftKings
                                     player props, for live games and for past games (backtests)
"""

import concurrent.futures as cf
import datetime as dt
import io
import json
import math
import os
import pickle
import re
import time
import unicodedata

import numpy as np
import pandas as pd
import requests

DATA_DIR = "mlb_data"
API = "https://statsapi.mlb.com/api/v1"
ESPN_CORE = "https://sports.core.api.espn.com/v2/sports/baseball/leagues/mlb"
ESPN_SITE = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb"
HEADERS = {"User-Agent": "Mozilla/5.0 (mlb-game-sim)"}
_session = requests.Session()
_session.headers.update(HEADERS)

OUTCOMES = ["K", "BB", "HBP", "1B", "2B", "3B", "HR", "OIP"]   # OIP = out on a ball in play


def _path(name):
    os.makedirs(DATA_DIR, exist_ok=True)
    return os.path.join(DATA_DIR, name)


def get_json(url, params=None, tries=4):
    for i in range(tries):
        try:
            r = _session.get(url, params=params, timeout=40)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.json()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(1.5 * (i + 1))


def get_text(url, params=None):
    r = _session.get(url, params=params, timeout=60)
    r.raise_for_status()
    return r.text


def _pmap(fn, items, workers=16):
    with cf.ThreadPoolExecutor(workers) as ex:
        return list(ex.map(fn, items))


def _cached(name, max_age_hours, build):
    """Pickle cache. max_age_hours=None means never expires."""
    p = _path(name)
    if os.path.exists(p) and (max_age_hours is None or (time.time() - os.path.getmtime(p)) / 3600 < max_age_hours):
        with open(p, "rb") as f:
            return pickle.load(f)
    obj = build()
    with open(p, "wb") as f:
        pickle.dump(obj, f)
    return obj


def norm_name(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", s)
    return re.sub(r"[^a-z]", "", s)


def to_date(d):
    if d is None:
        return dt.date.today()
    if isinstance(d, str):
        return dt.date.fromisoformat(d)
    if isinstance(d, dt.datetime):
        return d.date()
    return d


# ================================================================ teams / people
def teams():
    def build():
        js = get_json(f"{API}/teams", {"sportId": 1})
        rows = [{"team_id": t["id"], "abbr": t["abbreviation"], "name": t["name"],
                 "club": t.get("clubName", t.get("teamName")), "league": t["league"]["name"]}
                for t in js["teams"]]
        return pd.DataFrame(rows)
    return _cached("teams.pkl", 24 * 30, build)


_people_cache = None


def people(ids):
    """Handedness and names, cached across runs."""
    global _people_cache
    p = _path("people.pkl")
    if _people_cache is None:
        _people_cache = pickle.load(open(p, "rb")) if os.path.exists(p) else {}
    need = [int(i) for i in set(ids) if pd.notna(i) and int(i) not in _people_cache]
    chunks = [need[i:i + 150] for i in range(0, len(need), 150)]

    def fetch(ch):
        js = get_json(f"{API}/people", {"personIds": ",".join(map(str, ch))})
        return js.get("people", []) if js else []
    for res in _pmap(fetch, chunks, 6):
        for q in res:
            _people_cache[q["id"]] = {"name": q["fullName"], "bats": q.get("batSide", {}).get("code", "R"),
                                      "throws": "L" if q.get("pitchHand", {}).get("code") == "L" else "R",
                                      "pos": q.get("primaryPosition", {}).get("abbreviation", "")}
    if need:
        pickle.dump(_people_cache, open(p, "wb"))
    return {int(i): _people_cache.get(int(i), {"name": str(i), "bats": "R", "throws": "R", "pos": ""})
            for i in ids if pd.notna(i)}


# ================================================================ schedule
_WIND_DIR = {"out to cf": 1.0, "out to lf": 0.7, "out to rf": 0.7, "in from cf": -1.0,
             "in from lf": -0.7, "in from rf": -0.7}


def parse_weather(w, roof):
    """statsapi weather -> (temp F, wind-out mph component, roof_closed)."""
    w = w or {}
    cond = str(w.get("condition", "")).lower()
    closed = ("dome" in cond) or ("roof closed" in cond) or (str(roof).lower() == "dome")
    try:
        temp = float(w.get("temp"))
    except (TypeError, ValueError):
        temp = np.nan
    wind = str(w.get("wind", "")).lower()
    m = re.match(r"\s*(\d+)\s*mph,?\s*(.*)", wind)
    out = 0.0
    if m:
        mph, d = float(m.group(1)), m.group(2).strip()
        out = mph * _WIND_DIR.get(d, 0.0)
    if closed:
        out = 0.0
    return temp, out, closed


def _parse_game(g):
    t = g["teams"]
    v = g.get("venue", {})
    loc, fi = v.get("location", {}), v.get("fieldInfo", {})
    coords = loc.get("defaultCoordinates", {})
    temp, wind_out, closed = parse_weather(g.get("weather"), fi.get("roofType"))
    hp = [o["official"] for o in g.get("officials", []) if o.get("officialType") == "Home Plate"]
    lu = g.get("lineups", {})
    return {
        "game_pk": g["gamePk"], "date": g.get("officialDate", g["gameDate"][:10]),
        "start": g["gameDate"], "game_type": g["gameType"], "status": g["status"]["detailedState"],
        "abstract": g["status"]["abstractGameState"], "day_night": g.get("dayNight"),
        "dh": g.get("doubleHeader"), "game_num": g.get("gameNumber", 1),
        "series_game": g.get("seriesGameNumber"), "innings_sched": g.get("scheduledInnings", 9),
        "away_id": t["away"]["team"]["id"], "home_id": t["home"]["team"]["id"],
        "away": t["away"]["team"]["name"], "home": t["home"]["team"]["name"],
        "away_score": t["away"].get("score"), "home_score": t["home"].get("score"),
        "away_sp": t["away"].get("probablePitcher", {}).get("id"),
        "home_sp": t["home"].get("probablePitcher", {}).get("id"),
        "venue_id": v.get("id"), "venue": v.get("name"),
        "lat": coords.get("latitude"), "lon": coords.get("longitude"),
        "azimuth": loc.get("azimuthAngle"), "elevation": loc.get("elevation"),
        "roof": fi.get("roofType"), "tz_offset": None,
        "temp": temp, "wind_out": wind_out, "roof_closed": closed,
        "weather_raw": json.dumps(g.get("weather") or {}),
        "hp_ump_id": hp[0]["id"] if hp else None, "hp_ump": hp[0]["fullName"] if hp else None,
        "away_lineup": [p["id"] for p in lu.get("awayPlayers", [])] or None,
        "home_lineup": [p["id"] for p in lu.get("homePlayers", [])] or None,
    }


def schedule(start, end, game_types="R,F,D,L,W"):
    start, end = to_date(start), to_date(end)
    js = get_json(f"{API}/schedule", {"sportId": 1, "startDate": start.isoformat(), "endDate": end.isoformat(),
                                      "gameType": game_types,
                                      "hydrate": "probablePitcher,lineups,officials,venue(location,fieldInfo),weather,linescore,timezone"})
    rows = [_parse_game(g) for d in js.get("dates", []) for g in d["games"]]
    df = pd.DataFrame(rows)
    if len(df):
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df["start"] = pd.to_datetime(df["start"], utc=True)
    return df


def season_schedule(season, refresh_hours=6):
    """Full season (regular + postseason). Past seasons cache forever."""
    live = season >= dt.date.today().year
    return _cached(f"schedule_{season}.pkl", refresh_hours if live else None,
                   lambda: schedule(f"{season}-02-15", f"{season}-11-15"))


def final_games(sched):
    s = sched[(sched.abstract == "Final") & sched.home_score.notna() & (sched.status != "Postponed")]
    s = s[~s.status.str.contains("Cancel|Postpon|Suspend", case=False, na=False)]
    return s.drop_duplicates("game_pk").copy()


# ================================================================ boxscores
def _parse_box(pk):
    js = get_json(f"{API}/game/{pk}/boxscore")
    if not js:
        return None
    out = {"game_pk": pk, "teams": {}}
    for side in ("away", "home"):
        t = js["teams"][side]
        bat = t["teamStats"]["batting"]
        players = t["players"]
        order = []
        batters = {}
        for key, p in players.items():
            pid = p["person"]["id"]
            b = p.get("stats", {}).get("batting", {})
            bo = p.get("battingOrder")
            if bo:
                bo = int(bo)
                if bo % 100 == 0:
                    order.append((bo // 100, pid, p.get("position", {}).get("abbreviation", "")))
            if b and b.get("plateAppearances", 0) > 0:
                batters[pid] = {k: b.get(k, 0) for k in ("plateAppearances", "hits", "doubles", "triples", "homeRuns",
                                                         "baseOnBalls", "strikeOuts", "runs", "rbi", "totalBases",
                                                         "stolenBases", "hitByPitch")}
        order.sort()
        pitchers = []
        for i, pid in enumerate(t.get("pitchers", [])):
            ps = players.get(f"ID{pid}", {}).get("stats", {}).get("pitching", {})
            pitchers.append({"id": pid, "start": i == 0, "pitches": ps.get("numberOfPitches", 0),
                             "outs": ps.get("outs", 0), "bf": ps.get("battersFaced", 0), "k": ps.get("strikeOuts", 0),
                             "bb": ps.get("baseOnBalls", 0), "h": ps.get("hits", 0), "r": ps.get("runs", 0),
                             "er": ps.get("earnedRuns", 0), "hr": ps.get("homeRuns", 0),
                             "win": "(W" in str(ps.get("note", ""))})
        out["teams"][side] = {
            "team_id": t["team"]["id"],
            "bat": {k: bat.get(k, 0) for k in ("plateAppearances", "hits", "doubles", "triples", "homeRuns",
                                              "baseOnBalls", "intentionalWalks", "hitByPitch", "strikeOuts", "runs")},
            "lineup": [pid for _, pid, _ in order[:9]],
            "lineup_pos": [pos for _, _, pos in order[:9]],
            "batters": batters, "pitchers": pitchers}
    return out


def boxscores(game_pks, season):
    """Parsed boxscores keyed by game_pk; cached per season, only missing games fetched."""
    p = _path(f"box_{season}.pkl")
    cache = pickle.load(open(p, "rb")) if os.path.exists(p) else {}
    need = [int(x) for x in game_pks if int(x) not in cache]
    if need:
        for pk, res in zip(need, _pmap(_parse_box, need, 16)):
            if res:
                cache[pk] = res
        pickle.dump(cache, open(p, "wb"))
    return {int(k): cache[int(k)] for k in game_pks if int(k) in cache}


def game_frame(sched, box):
    """One row per team-game with batting totals, starter info, park, weather, ump."""
    rows = []
    for g in sched.itertuples():
        b = box.get(int(g.game_pk))
        if not b:
            continue
        for side, opp in (("away", "home"), ("home", "away")):
            tb, ob = b["teams"][side], b["teams"][opp]
            sp = ob["pitchers"][0] if ob["pitchers"] else {}
            rows.append({"game_pk": g.game_pk, "date": g.date, "game_type": g.game_type, "side": side,
                         "team_id": tb["team_id"], "opp_id": ob["team_id"], "venue_id": g.venue_id,
                         "home": side == "home", "temp": g.temp, "wind_out": g.wind_out, "roof_closed": g.roof_closed,
                         "hp_ump_id": g.hp_ump_id, "opp_sp": sp.get("id"),
                         **{f"b_{k}": v for k, v in tb["bat"].items()}})
    return pd.DataFrame(rows)


# ================================================================ platoon splits
_SPLIT_FIELDS = ["plateAppearances", "hits", "doubles", "triples", "homeRuns", "baseOnBalls", "intentionalWalks",
                 "hitByPitch", "strikeOuts", "groundOuts", "airOuts", "groundIntoDoublePlay", "sacFlies", "sacBunts",
                 "numberOfPitches"]


def _splits(season, group, start=None, end=None):
    params = {"stats": "statSplits", "group": group, "season": season, "sitCodes": "vl,vr",
              "playerPool": "all", "limit": 10000, "sportId": 1}
    if start:
        params.update(startDate=str(start), endDate=str(end))
    js = get_json(f"{API}/stats", params)
    rows = []
    for s in js["stats"][0]["splits"]:
        st = s["stat"]
        pa = st.get("plateAppearances") or st.get("battersFaced") or 0
        r = {"pid": s["player"]["id"], "name": s["player"]["fullName"], "vs": s["split"]["code"][-1].upper(),
             "PA": pa}
        for f in _SPLIT_FIELDS[1:]:
            r[f] = st.get(f, 0) or 0
        rows.append(r)
    df = pd.DataFrame(rows)
    if not len(df):
        return df
    df = df.groupby(["pid", "name", "vs"], as_index=False).sum(numeric_only=True)
    df["1B"] = df.hits - df.doubles - df.triples - df.homeRuns
    df["BB"] = df.baseOnBalls
    df["HBP"] = df.hitByPitch
    df["K"] = df.strikeOuts
    df["2B"], df["3B"], df["HR"] = df.doubles, df.triples, df.homeRuns
    df["OIP"] = (df.PA - df[["K", "BB", "HBP", "1B", "2B", "3B", "HR"]].sum(axis=1)).clip(lower=0)
    df["GO"], df["AO"], df["GIDP"] = df.groundOuts, df.airOuts, df.groundIntoDoublePlay
    df["pitches"] = df.numberOfPitches
    return df[["pid", "name", "vs", "PA"] + OUTCOMES + ["GO", "AO", "GIDP", "pitches"]]


def splits(season, group, asof=None):
    """Platoon splits for a season. If asof is inside the season, only games before asof count."""
    asof = to_date(asof) if asof else None
    if asof and asof.year == season and asof < dt.date.today():
        start = dt.date(season, 3, 1)
        end = asof - dt.timedelta(days=1)
        return _cached(f"splits_{group}_{season}_to_{end}.pkl", None, lambda: _splits(season, group, start, end))
    live = season >= dt.date.today().year
    return _cached(f"splits_{group}_{season}.pkl", 6 if live else None, lambda: _splits(season, group))


# ================================================================ Savant
def _savant_csv(url, name, live):
    def build():
        txt = get_text(url).lstrip("﻿")
        if txt.lstrip().startswith("<"):
            return pd.DataFrame()
        return pd.read_csv(io.StringIO(txt))
    return _cached(name, 12 if live else None, build)


def savant(year):
    live = year >= dt.date.today().year
    S = "https://baseballsavant.mlb.com/leaderboard"
    oaa = _savant_csv(f"{S}/outs_above_average?type=Fielding_Team&startYear={year}&endYear={year}&split=no&team=&range=year&min=q&pos=&roles=&viz=hide&csv=true", f"sv_oaa_team_{year}.pkl", live)
    fr = _savant_csv(f"{S}/catcher-framing?year={year}&team=&min=q&sort=4,1&csv=true", f"sv_frame_{year}.pkl", live)
    sp = _savant_csv(f"{S}/sprint_speed?year={year}&position=&team=&min=0&csv=true", f"sv_speed_{year}.pkl", live)
    xb = _savant_csv(f"{S}/expected_statistics?type=batter&year={year}&position=&team=&min=1&csv=true", f"sv_xb_{year}.pkl", live)
    xp = _savant_csv(f"{S}/expected_statistics?type=pitcher&year={year}&position=&team=&min=1&csv=true", f"sv_xp_{year}.pkl", live)
    return {"oaa": oaa, "framing": fr, "speed": sp, "xbat": xb, "xpit": xp}


# ================================================================ weather forecast
def forecast_weather(lat, lon, start_utc, azimuth):
    """Open-Meteo forecast at first pitch -> (temp F, wind-out mph). Wind-out uses the park's
    home-plate-to-center-field bearing, so 10 mph blowing straight out = +10."""
    try:
        js = get_json("https://api.open-meteo.com/v1/forecast",
                      {"latitude": lat, "longitude": lon, "hourly": "temperature_2m,wind_speed_10m,wind_direction_10m",
                       "wind_speed_unit": "mph", "temperature_unit": "fahrenheit", "timezone": "UTC", "forecast_days": 7})
        h = pd.DataFrame(js["hourly"])
        h["time"] = pd.to_datetime(h.time, utc=True)
        i = (h.time - pd.Timestamp(start_utc)).abs().idxmin()
        row = h.loc[i]
        to_dir = (row.wind_direction_10m + 180) % 360          # meteorological "from" -> blowing toward
        out = row.wind_speed_10m * math.cos(math.radians(to_dir - (azimuth or 0)))
        return float(row.temperature_2m), float(out)
    except Exception:
        return np.nan, 0.0


# ================================================================ DraftKings via ESPN
def _american_to_prob(a):
    a = float(a)
    return 100 / (a + 100) if a > 0 else -a / (-a + 100)


def _num(x):
    try:
        return float(str(x).replace("+", ""))
    except (TypeError, ValueError):
        return np.nan


def espn_events(date):
    date = to_date(date)
    js = get_json(f"{ESPN_SITE}/scoreboard", {"dates": date.strftime("%Y%m%d"), "limit": 50})
    rows = []
    for e in (js or {}).get("events", []):
        c = e["competitions"][0]
        teams_ = {x["homeAway"]: x["team"] for x in c["competitors"]}
        rows.append({"espn_id": e["id"], "start": pd.Timestamp(e["date"]),
                     "away_name": teams_["away"]["displayName"], "home_name": teams_["home"]["displayName"],
                     "away_espn_team": teams_["away"]["id"], "home_espn_team": teams_["home"]["id"]})
    return pd.DataFrame(rows)


def _line_side(team_odds, which):
    """Moneyline / run line american odds for one side at open, close or current."""
    o = team_odds.get(which) or {}
    ml = (o.get("moneyLine") or {}).get("american")
    rl = (o.get("pointSpread") or {}).get("american")
    rlp = (o.get("spread") or {}).get("american")
    return _num(ml), _num(rl), _num(rlp)


def espn_game_odds(espn_id):
    js = get_json(f"{ESPN_CORE}/events/{espn_id}/competitions/{espn_id}/odds")
    items = [i for i in (js or {}).get("items", []) if i.get("provider", {}).get("name") == "DraftKings"]
    if not items:
        return None
    o = items[0]
    out = {"espn_id": espn_id}
    for which in ("open", "close", "current"):
        a_ml, a_rl, a_rlp = _line_side(o.get("awayTeamOdds", {}), which)
        h_ml, h_rl, h_rlp = _line_side(o.get("homeTeamOdds", {}), which)
        tot = o.get(which) or {}
        out.update({f"{which}_away_ml": a_ml, f"{which}_home_ml": h_ml,
                    f"{which}_home_rl": h_rl, f"{which}_home_rl_price": h_rlp, f"{which}_away_rl_price": a_rlp,
                    f"{which}_total": _num(((tot.get("total") or {}).get("alternateDisplayValue")
                                            or (tot.get("total") or {}).get("american") or "").replace("o", "").replace("u", "")),
                    f"{which}_over": _num((tot.get("over") or {}).get("american")),
                    f"{which}_under": _num((tot.get("under") or {}).get("american"))})
    if np.isnan(out["current_total"]):
        out["current_total"] = _num(o.get("overUnder"))
        out["current_over"], out["current_under"] = _num(o.get("overOdds")), _num(o.get("underOdds"))
    if np.isnan(out["current_away_ml"]):
        out["current_away_ml"] = _num(o.get("awayTeamOdds", {}).get("moneyLine"))
        out["current_home_ml"] = _num(o.get("homeTeamOdds", {}).get("moneyLine"))
    return out


_PROP_TYPES = {
    "Total Strikeouts": ("pitcher", "strikeouts"), "Total Outs Recorded": ("pitcher", "outs"),
    "Total Hits Allowed": ("pitcher", "hits_allowed"), "Total Walks Allowed": ("pitcher", "walks_allowed"),
    "Earned Runs Allowed": ("pitcher", "earned_runs"), "To Record Win": ("pitcher", "win"),
    "Total Hits": ("batter", "hits"), "Total Bases": ("batter", "total_bases"),
    "Total Runs Scored": ("batter", "runs"), "Total RBIs": ("batter", "rbi"),
    "Total Hits + Runs + RBIs": ("batter", "hrr"), "Total Singles Hit": ("batter", "singles"),
    "Total Doubles Hit": ("batter", "doubles"), "Total Walks (Batter)": ("batter", "walks"),
    "Home Runs Milestones": ("batter", "home_runs"),
    "Team Total Runs": ("team", "team_total"), "1st 5 Innings Moneyline": ("team", "f5_ml"),
    "1st 5 Innings Run Line": ("team", "f5_rl"), "1st 5 Innings Total Runs": ("game", "f5_total"),
    "First Team to Score": ("team", "first_score"),
}

_athlete_cache = {}


def _ref_id(ref):
    return ref.split("?")[0].rstrip("/").split("/")[-1]


def _espn_athlete(ref):
    aid = _ref_id(ref)
    if aid not in _athlete_cache:
        js = get_json(ref.replace("http:", "https:"))
        _athlete_cache[aid] = (js or {}).get("fullName", aid)
    return _athlete_cache[aid]


def espn_props(espn_id):
    """DraftKings props for one game. Over/under pairs come in order (over first, then under).
    `open` is the opening line (always pre-game); `current` is the latest line ESPN stored."""
    items, page = [], 1
    while True:
        js = get_json(f"{ESPN_CORE}/events/{espn_id}/competitions/{espn_id}/odds/100/propBets",
                      {"limit": 1000, "page": page})      # ESPN caps pages at 1000 items
        if not js:
            break
        items += js.get("items", [])
        if page >= js.get("pageCount", 1):
            break
        page += 1
    rows, seen = [], {}
    for it in items:
        name = it.get("type", {}).get("name")
        if name not in _PROP_TYPES:
            continue
        role, market = _PROP_TYPES[name]
        od = it.get("odds", {})
        line_cur = (it.get("current", {}).get("target") or {}).get("value")
        line_open = (it.get("open", {}).get("target") or {}).get("value")
        if market == "home_runs" and line_cur not in (1, 1.0):
            continue
        who = it.get("athlete", it.get("team", {})).get("$ref", "game")
        key = (who, market)
        seen[key] = seen.get(key, 0) + 1
        idx = seen[key]
        if market in ("home_runs", "f5_ml", "first_score", "f5_rl"):
            side = "yes"
        else:   # over/under pairs, and To Record Win as yes/no, come over (yes) first
            side = "over" if idx % 2 == 1 else "under"
        if market == "win":
            line_cur = line_open = 0.5
        rows.append({"espn_id": espn_id, "role": role, "market": market, "ref": who, "side": side,
                     "line": line_cur if line_cur is not None else _num(od.get("total", {}).get("value")),
                     "line_open": line_open if line_open is not None else _num(od.get("total", {}).get("open")),
                     "price": _num(od.get("american", {}).get("value")),
                     "price_open": _num(od.get("american", {}).get("open"))})
    df = pd.DataFrame(rows)
    if not len(df):
        return df
    ath = df.role.isin(["pitcher", "batter"])
    refs = df.loc[ath, "ref"].unique().tolist()
    names = dict(zip(refs, _pmap(_espn_athlete, refs, 12)))
    df["player"] = df.ref.map(names)
    df["espn_team"] = np.where(df.role == "team", df.ref.map(lambda r: _ref_id(r) if r != "game" else None), None)
    return df


def dk_lines(date, include_props=True):
    """DraftKings game lines (and props) for every MLB game on a date, via ESPN."""
    date = to_date(date)
    live = date >= dt.date.today() - dt.timedelta(days=1)

    def build():
        ev = espn_events(date)
        if not len(ev):
            return {"games": pd.DataFrame(), "props": pd.DataFrame()}
        odds = [x for x in _pmap(espn_game_odds, ev.espn_id.tolist(), 8) if x]
        games = ev.merge(pd.DataFrame(odds), on="espn_id", how="left") if odds else ev
        props = pd.DataFrame()
        if include_props:
            ps = [p for p in _pmap(espn_props, ev.espn_id.tolist(), 6) if p is not None and len(p)]
            props = pd.concat(ps, ignore_index=True) if ps else pd.DataFrame()
        return {"games": games, "props": props}
    return _cached(f"dk_{date}{'' if include_props else '_noprops'}.pkl", 0.5 if live else None, build)


def match_espn(sched_day, ev):
    """Attach espn_id to statsapi games by team names and start time (handles doubleheaders)."""
    if not len(ev):
        sched_day = sched_day.copy()
        sched_day["espn_id"] = None
        return sched_day
    out = []
    for g in sched_day.itertuples():
        c = ev[(ev.home_name.map(norm_name) == norm_name(g.home)) & (ev.away_name.map(norm_name) == norm_name(g.away))]
        if len(c) > 1:
            c = c.iloc[[(c.start - g.start).abs().argmin()]]
        out.append(c.espn_id.iloc[0] if len(c) else None)
    s = sched_day.copy()
    s["espn_id"] = out
    return s
