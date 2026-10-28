"""
Daily MLB report, DraftKings comparison, prediction log with closing line value, and backtests.

    from mlb_daily import *
    report = daily_report()                                   # today's slate, logged and exported
    g = explore_game(report, "NYY", out=["Aaron Judge"])      # rerun one game with changes
    same_game(g, [("Gerrit Cole", "strikeouts", "over", 6.5), ("game", "total", "under", 7.5)])
    grade_log()                                               # logged predictions vs results and closing lines
    bt = backtest("2026-09-01", "2026-10-04")                 # as-of-date backtest vs DraftKings

Betting sections in every report
  Moneyline, run line (+/-1.5), game total, team totals, first 5 innings (moneyline, run line,
  total), first inning (NRFI/YRFI), first team to score, and player props (pitcher strikeouts,
  outs, hits/walks allowed, earned runs, win; batter hits, total bases, home run, runs, RBIs,
  hits+runs+RBIs, singles, doubles, walks). Each is priced by the model and, where DraftKings
  offers it, compared to DraftKings with the vig removed.

Closing line value (CLV): when a prediction is logged, the DraftKings line at that moment is saved.
After the game, the closing line is compared with it. If the market moved toward the model, the
model saw something the market later agreed with. Over many games, CLV is the fastest honest
signal of an edge; win-loss records take thousands of bets to mean anything.
"""

import datetime as dt
import html
import math
import os
import pickle
import time

import numpy as np
import pandas as pd

import mlb_data as md
import mlb_game_sim as ms

LOG_DIR = "logs"
REPORT_DIR = "reports"
MODEL_VERSION = "mlb-v1"
GAME_LOG = os.path.join(LOG_DIR, "game_predictions.csv")
PROP_LOG = os.path.join(LOG_DIR, "prop_predictions.csv")
EDGE_MIN = 0.03          # flag bets where model prob beats DraftKings' no-vig prob by 3+ points

TEAM_ABBR = None


def _abbr(name):
    global TEAM_ABBR
    if TEAM_ABBR is None:
        t = md.teams()
        TEAM_ABBR = dict(zip(t.name, t.abbr))
    return TEAM_ABBR.get(name, name[:3].upper())


# ================================================================== odds math
def am_to_prob(a):
    if a is None or not np.isfinite(a):
        return np.nan
    return 100 / (a + 100) if a > 0 else -a / (-a + 100)


def am_to_dec(a):
    if a is None or not np.isfinite(a):
        return np.nan
    return 1 + (a / 100 if a > 0 else 100 / -a)


def prob_to_am(p):
    if p is None or not np.isfinite(p) or p <= 0 or p >= 1:
        return np.nan
    return round(-100 * p / (1 - p)) if p >= 0.5 else round(100 * (1 - p) / p)


def no_vig(a, b):
    pa, pb = am_to_prob(a), am_to_prob(b)
    s = pa + pb
    return (pa / s, pb / s) if s and np.isfinite(s) else (np.nan, np.nan)


def ev(p_win, price, p_push=0.0):
    """Expected profit per $1 staked."""
    d = am_to_dec(price)
    return p_win * (d - 1) - (1 - p_win - p_push)


def p_over(arr, line):
    arr = np.asarray(arr, float)
    return float((arr > line).mean()), float((arr == line).mean())


# ================================================================== model fitting (cached per day)
def get_model(asof=None, calibrate=True, verbose=True, cache=True):
    asof = md.to_date(asof)
    path = os.path.join(md.DATA_DIR, f"model_{asof}_{'cal' if calibrate else 'raw'}.pkl")
    live = asof >= dt.date.today()
    if cache and os.path.exists(path) and (not live or (time.time() - os.path.getmtime(path)) < 6 * 3600):
        return pickle.load(open(path, "rb"))
    m = ms.fit(asof, verbose=verbose)
    if calibrate:
        ms.calibrate(m, verbose=verbose)
    if cache:
        pickle.dump(m, open(path, "wb"))
    return m


# ================================================================== markets
def _team_rows(res, dk, props_game, espn_team_ids):
    """Game-level markets: model prob, DraftKings price, no-vig prob, edge, EV."""
    A, H = res["away"], res["home"]
    tot, mar = res["_tot"], res["_margin"]
    away_r, home_r, f5 = res["_away"], res["_home"], res["_f5"]
    rows = []

    def add(market, side, line, p, price=np.nan, p_mkt=np.nan, p_push=0.0, price_open=np.nan, line_open=np.nan):
        rows.append({"game": res["game"], "game_pk": res["game_pk"], "market": market, "side": side, "line": line,
                     "model_prob": p, "fair_odds": prob_to_am(p / (1 - p_push)) if p_push < 1 else np.nan,
                     "dk_price": price, "dk_novig": p_mkt, "edge": p - p_mkt if np.isfinite(p_mkt) else np.nan,
                     "ev": ev(p, price, p_push) if np.isfinite(price) else np.nan,
                     "dk_line_open": line_open, "dk_price_open": price_open, "push": p_push})

    g = dk if dk is not None else {}
    w = "current"
    # moneyline
    pa_, ph_ = no_vig(g.get(f"{w}_away_ml", np.nan), g.get(f"{w}_home_ml", np.nan))
    add("moneyline", A, None, res["away_win"], g.get(f"{w}_away_ml", np.nan), pa_, price_open=g.get("open_away_ml", np.nan))
    add("moneyline", H, None, res["home_win"], g.get(f"{w}_home_ml", np.nan), ph_, price_open=g.get("open_home_ml", np.nan))
    # run line
    hl = g.get(f"{w}_home_rl", np.nan)
    if not np.isfinite(hl):
        hl = -1.5 if res["home_win"] >= 0.5 else 1.5
    p_home_cover = float((mar + hl > 0).mean())
    pa_rl, ph_rl = no_vig(g.get(f"{w}_away_rl_price", np.nan), g.get(f"{w}_home_rl_price", np.nan))
    add("run line", H, hl, p_home_cover, g.get(f"{w}_home_rl_price", np.nan), ph_rl)
    add("run line", A, -hl, 1 - p_home_cover, g.get(f"{w}_away_rl_price", np.nan), pa_rl)
    # total
    tl = g.get(f"{w}_total", np.nan)
    if not np.isfinite(tl):
        tl = round(res["total"] * 2) / 2
    po, pp = p_over(tot, tl)
    pov, pun = no_vig(g.get(f"{w}_over", np.nan), g.get(f"{w}_under", np.nan))
    add("total", "over", tl, po, g.get(f"{w}_over", np.nan), pov, pp, g.get("open_over", np.nan), g.get("open_total", np.nan))
    add("total", "under", tl, 1 - po - pp, g.get(f"{w}_under", np.nan), pun, pp, g.get("open_under", np.nan), g.get("open_total", np.nan))

    # DraftKings team-level props from ESPN (team totals, F5, first to score)
    pg = props_game if props_game is not None and len(props_game) else pd.DataFrame(columns=["market"])
    side_of = {espn_team_ids.get("away"): "away", espn_team_ids.get("home"): "home"}

    def pair(df):
        """over/under rows -> (line, over price, under price, open line, open over, open under)"""
        o = df[df.side == "over"]
        u = df[df.side == "under"]
        if not len(o) or not len(u):
            return None
        return (o.line.iloc[0], o.price.iloc[0], u.price.iloc[0], o.line_open.iloc[0], o.price_open.iloc[0], u.price_open.iloc[0])

    for s, arr, nm in (("away", away_r, A), ("home", home_r, H)):
        sub = pg[(pg.market == "team_total") & (pg.espn_team.map(side_of) == s)] if len(pg) else pg
        pr = pair(sub) if len(sub) else None
        line = pr[0] if pr else round(arr.mean() * 2) / 2 - 0.5 + 0.0
        po, pp = p_over(arr, line)
        nvo, nvu = no_vig(pr[1], pr[2]) if pr else (np.nan, np.nan)
        add(f"team total {_abbr(nm)}", "over", line, po, pr[1] if pr else np.nan, nvo, pp)
        add(f"team total {_abbr(nm)}", "under", line, 1 - po - pp, pr[2] if pr else np.nan, nvu, pp)

    # first 5 innings
    f5m = f5[:, 1] - f5[:, 0]
    tie = float((f5m == 0).mean())
    sub = pg[pg.market == "f5_ml"] if len(pg) else pg
    pr = {side_of.get(r.espn_team): r.price for r in sub.itertuples()} if len(sub) else {}
    nva, nvh = no_vig(pr.get("away", np.nan), pr.get("home", np.nan))
    add("F5 moneyline", A, None, res["f5_away_win"], pr.get("away", np.nan), nva, tie)
    add("F5 moneyline", H, None, res["f5_home_win"], pr.get("home", np.nan), nvh, tie)
    sub = pg[pg.market == "f5_total"] if len(pg) else pg
    p5 = pair(sub) if len(sub) else None
    l5 = p5[0] if p5 else round(res["f5_total"] * 2) / 2
    po, pp = p_over(f5.sum(1), l5)
    nvo, nvu = no_vig(p5[1], p5[2]) if p5 else (np.nan, np.nan)
    add("F5 total", "over", l5, po, p5[1] if p5 else np.nan, nvo, pp)
    add("F5 total", "under", l5, 1 - po - pp, p5[2] if p5 else np.nan, nvu, pp)
    sub = pg[pg.market == "f5_rl"] if len(pg) else pg
    if len(sub):
        rl = {side_of.get(r.espn_team): (r.line, r.price) for r in sub.itertuples()}
        if "home" in rl and "away" in rl:
            hl5 = rl["home"][0]
            ph5 = float((f5m + hl5 > 0).mean())
            push5 = float((f5m + hl5 == 0).mean())
            nva, nvh = no_vig(rl["away"][1], rl["home"][1])
            add("F5 run line", H, hl5, ph5, rl["home"][1], nvh, push5)
            add("F5 run line", A, -hl5, 1 - ph5 - push5, rl["away"][1], nva, push5)
    # first inning, first to score
    add("1st inning", "NRFI (no run)", None, res["nrfi"])
    add("1st inning", "YRFI (run)", None, res["yrfi"])
    sub = pg[pg.market == "first_score"] if len(pg) else pg
    pr = {side_of.get(r.espn_team): r.price for r in sub.itertuples()} if len(sub) else {}
    nva, nvh = no_vig(pr.get("away", np.nan), pr.get("home", np.nan))
    add("first to score", A, None, res["away_scores_first"], pr.get("away", np.nan), nva)
    add("first to score", H, None, res["home_scores_first"], pr.get("home", np.nan), nvh)
    return rows


def _prop_rows(res, props_game):
    """Player props: model distribution vs DraftKings line (open and latest)."""
    rows = []
    pg = props_game if props_game is not None and len(props_game) else None
    by_name = {}
    for p in res["props"]:
        by_name[md.norm_name(p["player"])] = p
    if pg is not None:
        for (player, market), df in pg[pg.role.isin(["batter", "pitcher"])].groupby(["player", "market"]):
            p = by_name.get(md.norm_name(player))
            if p is None:
                # last-name + first-initial fallback
                key = md.norm_name(player)
                cand = [v for k, v in by_name.items() if k[-6:] == key[-6:]]
                p = cand[0] if len(cand) == 1 else None
            if p is None or market not in p["_dist"]:
                continue
            dist = p["_dist"][market]
            for r in df.itertuples():
                if r.side == "yes":
                    prob = float((dist >= 1).mean())
                    other = np.nan
                    nv = am_to_prob(r.price) / 1.06 if np.isfinite(r.price) else np.nan   # one-sided: assume ~6% hold
                    nv_open = am_to_prob(r.price_open) / 1.06 if np.isfinite(r.price_open) else np.nan
                    rows.append(_prop_row(res, p, market, "yes", r.line, prob, 0.0, r.price, nv, r.price_open, r.line_open, nv_open, dist))
                    continue
                opp = df[(df.side != r.side) & (df.line == r.line)]
                po, pp = p_over(dist, r.line)
                prob = po if r.side == "over" else 1 - po - pp
                nv = np.nan
                if len(opp):
                    a, b = no_vig(r.price, opp.price.iloc[0])
                    nv = a
                nv_open = np.nan
                opp_o = df[(df.side != r.side) & (df.line_open == r.line_open)]
                if len(opp_o):
                    nv_open = no_vig(r.price_open, opp_o.price_open.iloc[0])[0]
                rows.append(_prop_row(res, p, market, r.side, r.line, prob, pp, r.price, nv, r.price_open, r.line_open, nv_open, dist))
    return rows


def _prop_row(res, p, market, side, line, prob, push, price, nv, price_open, line_open, nv_open, dist):
    if np.isfinite(line_open if line_open is not None else np.nan) and side in ("over", "under"):
        po_o, pp_o = p_over(dist, line_open)
        prob_open = po_o if side == "over" else 1 - po_o - pp_o
    else:
        prob_open, pp_o = prob, push
    return {"game": res["game"], "game_pk": res["game_pk"], "player": p["player"], "pid": p["pid"], "team": p["team"],
            "role": p["role"], "market": market, "side": side, "line": line, "model_mean": float(np.mean(dist)),
            "model_prob": prob, "fair_odds": prob_to_am(prob / max(1 - push, 1e-9)), "dk_price": price, "dk_novig": nv,
            "edge": prob - nv if np.isfinite(nv) else np.nan, "ev": ev(prob, price, push) if np.isfinite(price) else np.nan,
            "push": push, "dk_line_open": line_open, "dk_price_open": price_open, "dk_novig_open": nv_open,
            "model_prob_open": prob_open, "push_open": pp_o,
            "edge_open": prob_open - nv_open if np.isfinite(nv_open) else np.nan,
            "ev_open": ev(prob_open, price_open, pp_o) if np.isfinite(price_open) else np.nan}


# ================================================================== the slate
def _slate(date):
    date = md.to_date(date)
    sc = md.schedule(date, date)
    if not len(sc):
        return sc
    sc = sc[~sc.status.str.contains("Postpon|Cancel", case=False, na=False)]
    return sc


def daily_report(date=None, sims=6000, calibrate=True, log=True, export=True, odds=True, verbose=True):
    """Simulate every game on a date, compare with DraftKings, log predictions and export."""
    date = md.to_date(date)
    m = get_model(date, calibrate=calibrate, verbose=verbose)
    sc = _slate(date)
    if not len(sc):
        print(f"No MLB games on {date}.")
        return None
    dk = md.dk_lines(date) if odds else {"games": pd.DataFrame(), "props": pd.DataFrame()}
    sc = md.match_espn(sc, dk["games"]) if len(dk["games"]) else sc.assign(espn_id=None)
    games, board, props, results = [], [], [], {}
    for g in sc.itertuples():
        try:
            r = ms.run_game(m, g, n=sims, seed=int(g.game_pk) % 10000)
        except Exception as e:
            print(f"  skipped {g.away} @ {g.home}: {e}")
            continue
        results[g.game_pk] = r
        dg = None
        ids = {}
        if g.espn_id is not None and len(dk["games"]):
            row = dk["games"][dk["games"].espn_id == g.espn_id]
            if len(row):
                dg = row.iloc[0].to_dict()
                ids = {"away": str(dg.get("away_espn_team")), "home": str(dg.get("home_espn_team"))}
        pg = dk["props"][dk["props"].espn_id == g.espn_id] if g.espn_id is not None and len(dk["props"]) else None
        board += _team_rows(r, dg, pg, ids)
        props += _prop_rows(r, pg)
        G = r["G"]
        games.append({"game": r["game"], "game_pk": g.game_pk, "start_az": g.start.tz_convert("America/Phoenix").strftime("%-I:%M %p"),
                      "type": g.game_type, "away_sp": r["away_sp"], "home_sp": r["home_sp"],
                      "away_win": r["away_win"], "home_win": r["home_win"],
                      "away_runs": r["away_runs"], "home_runs": r["home_runs"], "total": r["total"],
                      "dk_away_ml": (dg or {}).get("current_away_ml"), "dk_home_ml": (dg or {}).get("current_home_ml"),
                      "dk_total": (dg or {}).get("current_total"),
                      "nrfi": r["nrfi"], "f5_total": r["f5_total"], "extras": r["extras"],
                      "temp": G["env"]["temp"], "wind_out": G["env"]["wind_out"], "roof_closed": G["env"]["roof_closed"],
                      "ump": G["env"]["ump"], "notes": " | ".join(G["notes"])})
    rep = {"date": date, "model": m, "games": pd.DataFrame(games), "board": pd.DataFrame(board),
           "props": pd.DataFrame(props), "results": results, "slate": sc, "dk": dk,
           "player_projections": _projections(results)}
    if verbose:
        _print_report(rep)
    if log:
        _log(rep)
    if export:
        export_report(rep)
    return rep


def _projections(results):
    rows = []
    for r in results.values():
        for p in r["props"]:
            rows.append({k: v for k, v in p.items() if k != "_dist"} | {"game": r["game"]})
    return pd.DataFrame(rows)


def best_bets(rep, min_edge=EDGE_MIN, props=True):
    b = rep["board"]
    b = b[b.edge >= min_edge].copy()
    b["type"] = "game"
    if props and len(rep["props"]):
        p = rep["props"][rep["props"].edge >= min_edge].copy()
        p["type"] = "prop"
        p["side"] = p.player + " " + p.side
        b = pd.concat([b, p[b.columns.intersection(p.columns).tolist()]], ignore_index=True)
    return b.sort_values("ev", ascending=False)


def _print_report(rep):
    g = rep["games"]
    print(f"\n=== MLB {rep['date']} ({len(g)} games) ===")
    if len(g):
        show = g[["game", "start_az", "away_sp", "home_sp", "away_win", "home_win", "total", "dk_total", "nrfi"]].copy()
        for c in ("away_win", "home_win", "nrfi"):
            show[c] = (show[c] * 100).round(1)
        print(show.round(2).to_string(index=False))
    bb = best_bets(rep)
    if len(bb):
        print(f"\nModel vs DraftKings, edge >= {EDGE_MIN:.0%} (sorted by EV per $1):")
        print(bb[["game", "market", "side", "line", "model_prob", "dk_novig", "dk_price", "edge", "ev"]].head(15).round(3).to_string(index=False))
    print("\nSept 2026 backtest: about even with DraftKings' closing moneyline, no proven profit at closing prices, and"
          "\nplayoff totals still run high. Treat edges as hypotheses to log and grade, not bets.")


# ================================================================== single game exploration
def explore_game(rep, team, lineups=None, starters=None, out=(), weather=None, sims=10000):
    """Rerun one game from a report with changes, e.g.
    explore_game(rep, "NYY", starters={"home": "Shane McClanahan"}, out=["Aaron Judge"],
                 weather={"temp": 55, "wind_out": -10})"""
    sc = rep["slate"]
    key = team.lower()
    hit = sc[sc.away.str.lower().str.contains(key) | sc.home.str.lower().str.contains(key) |
             sc.away.map(_abbr).str.lower().eq(key) | sc.home.map(_abbr).str.lower().eq(key)]
    if not len(hit):
        raise ValueError(f"No game for {team}")
    g = hit.iloc[0]
    r = ms.run_game(rep["model"], g, n=sims, seed=1, lineups=lineups, starters=starters, out=out, weather=weather)
    base = rep["results"].get(g.game_pk)
    print(f"{r['game']}:  {r['away_sp']} vs {r['home_sp']}")
    print(f"  home win {r['home_win']:.1%}" + (f" (was {base['home_win']:.1%})" if base else "") +
          f"   total {r['total']:.2f}" + (f" (was {base['total']:.2f})" if base else "") +
          f"   NRFI {r['nrfi']:.1%}   F5 total {r['f5_total']:.2f}")
    print("  " + "\n  ".join(r["G"]["notes"]))
    return r


def same_game(res, legs):
    """Joint probability of several legs from the same simulations.
    legs: (player or "game", market, "over"/"under"/"yes", line). Game markets: total, home_margin, away_runs, home_runs."""
    n = len(res["_tot"])
    mask = np.ones(n, bool)
    single = []
    for who, market, side, line in legs:
        if who == "game":
            arr = {"total": res["_tot"], "home_margin": res["_margin"], "away_runs": res["_away"],
                   "home_runs": res["_home"], "f5_total": res["_f5"].sum(1)}[market]
        else:
            p = [x for x in res["props"] if md.norm_name(x["player"]) == md.norm_name(who)]
            if not p:
                raise ValueError(f"{who} not in this game")
            arr = p[0]["_dist"][market]
        hit = arr > line if side == "over" else (arr < line if side == "under" else arr >= 1)
        single.append(hit.mean())
        mask &= hit
    joint = mask.mean()
    print(f"Joint {joint:.1%} (fair {prob_to_am(joint):+}) vs {np.prod(single):.1%} if the legs were independent")
    return joint


# ================================================================== logging, grading, CLV
def _log(rep):
    os.makedirs(LOG_DIR, exist_ok=True)
    now = pd.Timestamp.now(tz="UTC")
    b = rep["board"].assign(logged_at=now, model=MODEL_VERSION, date=rep["date"])
    b.to_csv(GAME_LOG, mode="a", header=not os.path.exists(GAME_LOG), index=False)
    if len(rep["props"]):
        p = rep["props"].assign(logged_at=now, model=MODEL_VERSION, date=rep["date"])
        p.to_csv(PROP_LOG, mode="a", header=not os.path.exists(PROP_LOG), index=False)


def _settle_game(row, sc):
    """Outcome of a logged game bet: 1 win, 0 loss, 0.5 push, nan unsettled."""
    g = sc.get(row.game_pk)
    if g is None:
        return np.nan
    a, h = g["away_score"], g["home_score"]
    if a is None or pd.isna(a):
        return np.nan
    away, home = row.game.split(" @ ")
    m = row.market
    if m == "moneyline":
        won_home = h > a
        return float(won_home if row.side == home else not won_home)
    if m == "run line":
        mar = (h - a) if row.side == home else (a - h)
        v = mar + row.line
        return 0.5 if v == 0 else float(v > 0)
    if m == "total":
        t = a + h
        return 0.5 if t == row.line else float(t > row.line if row.side == "over" else t < row.line)
    return np.nan


def grade_log():
    """Grade logged predictions against results and DraftKings closing lines."""
    if not os.path.exists(GAME_LOG):
        print("No predictions logged yet.")
        return None
    gl = pd.read_csv(GAME_LOG)
    gl["logged_at"] = pd.to_datetime(gl.logged_at, utc=True)
    gl = gl.sort_values("logged_at").groupby(["game_pk", "market", "side"]).tail(1)
    out = {}
    dates = sorted(set(pd.to_datetime(gl.date).dt.date))
    sc = pd.concat([md.schedule(d, d) for d in dates], ignore_index=True)
    sc = sc[sc.abstract == "Final"]
    scd = {r.game_pk: {"away_score": r.away_score, "home_score": r.home_score} for r in sc.itertuples()}
    gl["result"] = [_settle_game(r, scd) for r in gl.itertuples()]
    # closing lines
    close = []
    for d in dates:
        dk = md.dk_lines(d, include_props=False)["games"]
        if not len(dk):
            continue
        s = md.match_espn(sc[sc.date == d], dk)
        for r in s.itertuples():
            row = dk[dk.espn_id == r.espn_id]
            if len(row):
                close.append({"game_pk": r.game_pk, **row.iloc[0].to_dict()})
    cl = pd.DataFrame(close)
    done = gl[gl.result.notna() & gl.market.isin(["moneyline", "run line", "total"])].copy()
    if not len(done):
        print("Nothing graded yet (games not final).")
        return None
    done = done.merge(cl, on="game_pk", how="left")

    def close_nv(r):
        away, home = r.game.split(" @ ")
        if r.market == "moneyline":
            pa, ph = no_vig(r.close_away_ml, r.close_home_ml)
            return ph if r.side == home else pa
        if r.market == "run line":
            pa, ph = no_vig(r.close_away_rl_price, r.close_home_rl_price)
            return ph if r.side == home else pa
        if r.market == "total":
            if r.close_total != r.line:
                return np.nan
            po, pu = no_vig(r.close_over, r.close_under)
            return po if r.side == "over" else pu
        return np.nan
    done["close_novig"] = [close_nv(r) for r in done.itertuples()]
    bets = done[done.edge >= EDGE_MIN].copy()
    bets["profit"] = np.where(bets.result == 0.5, 0, np.where(bets.result == 1, bets.dk_price.map(am_to_dec) - 1, -1))
    bets["clv"] = bets.close_novig - bets.dk_novig
    summ = {"bets graded": len(bets), "record": f"{int((bets.result == 1).sum())}-{int((bets.result == 0).sum())}",
            "ROI": f"{bets.profit.mean():.1%}" if len(bets) else "n/a",
            "avg CLV (prob pts)": round(bets.clv.mean() * 100, 2) if len(bets) else np.nan,
            "beat close %": f"{(bets.clv > 0).mean():.1%}" if len(bets) else "n/a"}
    print(pd.Series(summ).to_string())
    out["bets"], out["all"], out["summary"] = bets, done, summ
    return out


# ================================================================== backtest
def _actual_props(box_game):
    """Actual stat lines keyed by normalized player name."""
    act = {}
    for side in ("away", "home"):
        t = box_game["teams"][side]
        for pid, b in t["batters"].items():
            act[pid] = {"hits": b["hits"], "total_bases": b["totalBases"], "home_runs": b["homeRuns"],
                        "runs": b["runs"], "rbi": b["rbi"], "hrr": b["hits"] + b["runs"] + b["rbi"],
                        "walks": b["baseOnBalls"], "singles": b["hits"] - b["doubles"] - b["triples"] - b["homeRuns"],
                        "doubles": b["doubles"], "strikeouts": b["strikeOuts"]}
        if t["pitchers"]:
            p = t["pitchers"][0]
            act[p["id"]] = {"strikeouts": p["k"], "outs": p["outs"], "hits_allowed": p["h"], "walks_allowed": p["bb"],
                            "earned_runs": p["er"], "win": int(p["win"])}
    return act


def backtest(start, end, refit_days=7, sims=3000, calibrate=True, props=True, verbose=True, save="backtest"):
    """As-of-date backtest. Every game is simulated with a model fitted only on games before
    the refit date, Statcast leaderboards from the prior season, the posted lineups and starters,
    and first-pitch weather. Predictions are graded against results and DraftKings' opening and
    closing lines (via ESPN)."""
    start, end = md.to_date(start), md.to_date(end)
    games, propr = [], []
    d = start
    m = None
    fit_date = None
    while d <= end:
        if m is None or (d - fit_date).days >= refit_days:
            fit_date = d
            m = get_model(d, calibrate=calibrate, verbose=verbose)
        sc = md.schedule(d, d)
        if len(sc):
            sc = sc[sc.abstract == "Final"]
            sc = sc[~sc.status.str.contains("Postpon|Cancel|Suspend", case=False, na=False)]
        if len(sc):
            dk = md.dk_lines(d, include_props=props)
            sc = md.match_espn(sc, dk["games"]) if len(dk["games"]) else sc.assign(espn_id=None)
            box = md.boxscores(sc.game_pk.tolist(), d.year)
            for g in sc.itertuples():
                b = box.get(int(g.game_pk))
                if b is None:
                    continue
                try:
                    r = ms.run_game(m, g, n=sims, seed=int(g.game_pk) % 10000)
                except Exception as e:
                    if verbose:
                        print("  skip", g.away, "@", g.home, e)
                    continue
                dg = dk["games"][dk["games"].espn_id == g.espn_id] if g.espn_id is not None and len(dk["games"]) else pd.DataFrame()
                dg = dg.iloc[0].to_dict() if len(dg) else {}
                games.append({"date": d, "game_pk": g.game_pk, "game": r["game"], "type": g.game_type,
                              "away": g.away, "home": g.home, "away_score": g.away_score, "home_score": g.home_score,
                              "p_home": r["home_win"], "model_total": r["total"], "model_home_runs": r["home_runs"],
                              "model_away_runs": r["away_runs"],
                              "p_home_rl": float((r["_margin"] + (dg.get("close_home_rl") if np.isfinite(dg.get("close_home_rl", np.nan)) else -1.5) > 0).mean()),
                              "p_over_open": p_over(r["_tot"], dg.get("open_total", np.nan))[0] if np.isfinite(dg.get("open_total", np.nan)) else np.nan,
                              "push_open": p_over(r["_tot"], dg.get("open_total", np.nan))[1] if np.isfinite(dg.get("open_total", np.nan)) else np.nan,
                              "p_over_close": p_over(r["_tot"], dg.get("close_total", np.nan))[0] if np.isfinite(dg.get("close_total", np.nan)) else np.nan,
                              "push_close": p_over(r["_tot"], dg.get("close_total", np.nan))[1] if np.isfinite(dg.get("close_total", np.nan)) else np.nan,
                              "nrfi": r["nrfi"], "f5_total": r["f5_total"],
                              "actual_nrfi": None, "fit_date": fit_date,
                              **{k: v for k, v in dg.items() if k.startswith(("open_", "close_"))}})
                if props and g.espn_id is not None and len(dk["props"]):
                    pg = dk["props"][dk["props"].espn_id == g.espn_id]
                    act = _actual_props(b)
                    for pr in _prop_rows(r, pg):
                        a = act.get(pr["pid"], {}).get(pr["market"])
                        propr.append({**pr, "date": d, "actual": a})
            if verbose:
                print(f"  {d}: {len(sc)} games simulated", flush=True)
            if save:
                os.makedirs(REPORT_DIR, exist_ok=True)
                pickle.dump({"games": games, "props": propr, "through": d},
                            open(os.path.join(REPORT_DIR, f"{save}_checkpoint.pkl"), "wb"))
        d += dt.timedelta(days=1)
    G = pd.DataFrame(games)
    P = pd.DataFrame(propr)
    # first-inning results from linescores
    if len(G):
        G["actual_nrfi"] = [_first_inning_zero(pk) for pk in G.game_pk]
    out = {"games": G, "props": P}
    out["summary"] = summarize_backtest(G, P, verbose=verbose)
    if save:
        os.makedirs(REPORT_DIR, exist_ok=True)
        with open(os.path.join(REPORT_DIR, f"{save}_{start}_{end}.pkl"), "wb") as f:
            pickle.dump(out, f)
    return out


def _first_inning_zero(pk):
    try:
        js = md.get_json(f"{md.API}/game/{pk}/linescore")
        inn = js["innings"][0]
        return int(inn["away"].get("runs", 0) + inn["home"].get("runs", 0) == 0)
    except Exception:
        return np.nan


def _logloss(p, y):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def summarize_backtest(G, P, edge=EDGE_MIN, verbose=True):
    S = {}
    if len(G):
        g = G.dropna(subset=["close_home_ml", "close_away_ml"]).copy()
        g["y"] = (g.home_score > g.away_score).astype(int)
        g["dk_close"] = [no_vig(a, h)[1] for a, h in zip(g.close_away_ml, g.close_home_ml)]
        g["dk_open"] = [no_vig(a, h)[1] for a, h in zip(g.open_away_ml, g.open_home_ml)]
        g = g.dropna(subset=["dk_close", "dk_open"])
        S["games"] = len(g)
        S["winner"] = {
            "model log loss": _logloss(g.p_home, g.y), "DK open log loss": _logloss(g.dk_open, g.y),
            "DK close log loss": _logloss(g.dk_close, g.y), "coin flip log loss": math.log(2),
            "model Brier": float(np.mean((g.p_home - g.y) ** 2)), "DK close Brier": float(np.mean((g.dk_close - g.y) ** 2)),
            "model picks winner": float(((g.p_home > .5) == g.y).mean()), "DK favorite wins": float(((g.dk_close > .5) == g.y).mean()),
            "50/50 blend log loss": _logloss(0.5 * g.p_home + 0.5 * g.dk_close, g.y)}
        # moneyline betting at the opening price when model beats open no-vig by `edge`
        bets = []
        for r in g.itertuples():
            for side, p, nv_o, nv_c, price in (("home", r.p_home, r.dk_open, r.dk_close, r.open_home_ml),
                                               ("away", 1 - r.p_home, 1 - r.dk_open, 1 - r.dk_close, r.open_away_ml)):
                if p - nv_o >= edge:
                    won = r.y == 1 if side == "home" else r.y == 0
                    bets.append({"won": won, "profit": (am_to_dec(price) - 1) if won else -1, "clv": nv_c - nv_o,
                                 "edge": p - nv_o})
        b = pd.DataFrame(bets)
        S["moneyline bets (open price)"] = _bet_summary(b)
        # the honest test: closing prices, which already contain lineups, weather and late news
        ll = lambda p, y: -(y * np.log(np.clip(p, 1e-4, 1 - 1e-4)) + (1 - y) * np.log(1 - np.clip(p, 1e-4, 1 - 1e-4)))
        diff = (ll(g.p_home.values, g.y.values) - ll(g.dk_close.values, g.y.values))
        rng = np.random.default_rng(0)
        bs = [rng.choice(diff, len(diff)).mean() for _ in range(2000)]
        S["winner"]["model minus DK close log loss"] = float(diff.mean())
        S["winner"]["  90% range"] = (float(np.percentile(bs, 5)), float(np.percentile(bs, 95)))
        cb = []
        for r in g.itertuples():
            for side, p, nv, price in (("home", r.p_home, r.dk_close, r.close_home_ml), ("away", 1 - r.p_home, 1 - r.dk_close, r.close_away_ml)):
                if p - nv >= edge:
                    won = r.y == 1 if side == "home" else r.y == 0
                    cb.append({"won": won, "profit": (am_to_dec(price) - 1) if won else -1, "edge": p - nv})
        S["moneyline bets (close price)"] = _bet_summary(pd.DataFrame(cb))
        # totals at the open
        t = G.dropna(subset=["open_total", "close_total", "open_over", "open_under"]).copy()
        t["act"] = t.away_score + t.home_score
        S["total MAE"] = {"model": float((t.model_total - t.act).abs().mean()), "DK close line": float((t.close_total - t.act).abs().mean()),
                          "model mean": float(t.model_total.mean()), "actual mean": float(t.act.mean()),
                          "DK close mean": float(t.close_total.mean())}
        tb = []
        for r in t.itertuples():
            nvo, nvu = no_vig(r.open_over, r.open_under)
            p_o, p_u = r.p_over_open, 1 - r.p_over_open - r.push_open
            for side, p, nv, price in (("over", p_o, nvo, r.open_over), ("under", p_u, nvu, r.open_under)):
                if p - nv >= edge:
                    res = 0.5 if r.act == r.open_total else float(r.act > r.open_total if side == "over" else r.act < r.open_total)
                    clv_pts = (r.close_total - r.open_total) * (1 if side == "over" else -1)
                    tb.append({"won": res == 1, "push": res == 0.5,
                               "profit": 0 if res == 0.5 else ((am_to_dec(price) - 1) if res == 1 else -1),
                               "clv": clv_pts, "edge": p - nv})
        S["total bets (open price)"] = _bet_summary(pd.DataFrame(tb), clv_label="line move toward model (runs)")
        tc = []
        for r in t.dropna(subset=["close_over", "close_under"]).itertuples():
            nvo, nvu = no_vig(r.close_over, r.close_under)
            p_o, p_u = r.p_over_close, 1 - r.p_over_close - r.push_close
            for side, p, nv, price in (("over", p_o, nvo, r.close_over), ("under", p_u, nvu, r.close_under)):
                if p - nv >= edge:
                    res = 0.5 if r.act == r.close_total else float(r.act > r.close_total if side == "over" else r.act < r.close_total)
                    tc.append({"won": res == 1, "push": res == 0.5, "over": side == "over",
                               "profit": 0 if res == 0.5 else ((am_to_dec(price) - 1) if res == 1 else -1), "edge": p - nv})
        tc = pd.DataFrame(tc)
        S["total bets (close price)"] = _bet_summary(tc)
        if len(tc):
            S["total bets (close price)"]["share that were overs"] = float(tc.over.mean())
        post = t[t.type != "R"]
        if len(post):
            S["postseason totals"] = {"games": len(post), "model mean": float(post.model_total.mean()),
                                      "actual mean": float(post.act.mean()), "DK close mean": float(post.close_total.mean())}
        n = G.dropna(subset=["actual_nrfi"])
        S["NRFI"] = {"model avg": float(n.nrfi.mean()), "actual rate": float(n.actual_nrfi.mean()),
                     "model log loss": _logloss(n.nrfi, n.actual_nrfi),
                     "constant-rate log loss": _logloss(np.full(len(n), n.actual_nrfi.mean()), n.actual_nrfi)}
    if len(P):
        p = P.dropna(subset=["actual"]).copy()
        p = p[p.side.isin(["over", "under", "yes"])]
        rows = []
        for (mk), d in p.groupby("market"):
            dd = d.dropna(subset=["dk_novig_open", "dk_price_open"]).copy()
            if not len(dd):
                continue
            line = dd.dk_line_open
            won = np.where(dd.side == "over", dd.actual > line, np.where(dd.side == "under", dd.actual < line, dd.actual >= 1))
            push = (dd.side != "yes") & (dd.actual == line)
            dd["won"], dd["push"] = won, push
            ov = dd[dd.side.isin(["over", "yes"])]
            bt = dd[dd.edge_open >= edge]
            prof = np.where(bt.push, 0, np.where(bt.won, bt.dk_price_open.map(am_to_dec) - 1, -1))
            mae_model = float((ov.model_mean - ov.actual).abs().mean()) if len(ov) else np.nan
            rows.append({"market": mk, "offers": len(ov), "model MAE": mae_model,
                         "model log loss": _logloss(ov.model_prob_open, ov.won) if len(ov) else np.nan,
                         "DK open log loss": _logloss(ov.dk_novig_open, ov.won) if len(ov) else np.nan,
                         "bets": len(bt), "win%": float(bt.won[~bt.push].mean()) if len(bt) else np.nan,
                         "ROI": float(prof.mean()) if len(bt) else np.nan})
        S["props"] = pd.DataFrame(rows)
    if verbose:
        print_backtest(S)
    return S


def _bet_summary(b, clv_label="CLV (prob pts)"):
    if not len(b):
        return {"bets": 0}
    out = {"bets": len(b), "record": f"{int(b.won.sum())}-{int((~b.won & ~b.get('push', pd.Series(False, index=b.index))).sum())}",
           "ROI": float(b.profit.mean()), "avg edge claimed": float(b.edge.mean())}
    if "clv" in b:
        c = b.clv.dropna()
        out[clv_label] = float(c.mean() * (100 if "prob" in clv_label else 1))
        out["market moved toward model %"] = float((c > 0).mean())
        out["market moved away %"] = float((c < 0).mean())
    # bootstrap interval on ROI
    rng = np.random.default_rng(0)
    boots = [rng.choice(b.profit.values, len(b)).mean() for _ in range(2000)]
    out["ROI 90% range"] = (float(np.percentile(boots, 5)), float(np.percentile(boots, 95)))
    return out


def print_backtest(S):
    print("\n================ BACKTEST vs DRAFTKINGS ================")
    print(f"Games with DraftKings lines: {S.get('games')}")
    for k in ("winner", "moneyline bets (close price)", "moneyline bets (open price)", "total MAE",
              "total bets (close price)", "total bets (open price)", "postseason totals", "NRFI"):
        if k in S:
            print(f"\n{k}:")
            for a, v in S[k].items():
                print(f"  {a:34s} {v:.4f}" if isinstance(v, float) else f"  {a:34s} {v}")
    if "props" in S and len(S["props"]):
        print("\nplayer props vs DraftKings opening lines:")
        print(S["props"].round(3).to_string(index=False))


# ================================================================== export
def export_report(rep):
    os.makedirs(REPORT_DIR, exist_ok=True)
    base = os.path.join(REPORT_DIR, f"mlb_{rep['date']}")
    with pd.ExcelWriter(base + ".xlsx") as xw:
        rep["games"].to_excel(xw, sheet_name="Games", index=False)
        best_bets(rep).to_excel(xw, sheet_name="Best edges", index=False)
        rep["board"].to_excel(xw, sheet_name="Game markets", index=False)
        if len(rep["props"]):
            rep["props"].to_excel(xw, sheet_name="Props vs DK", index=False)
        if len(rep["player_projections"]):
            rep["player_projections"].to_excel(xw, sheet_name="Player projections", index=False)
    with open(base + ".html", "w") as f:
        f.write(dashboard_html(rep))
    print(f"Saved {base}.xlsx and {base}.html")
    return base


def _pct(x):
    return "" if x is None or not np.isfinite(x) else f"{x * 100:.1f}%"


def _am(x):
    return "" if x is None or not np.isfinite(x) else f"{int(x):+d}"


def dashboard_html(rep):
    g = rep["games"]
    cards = []
    for r in g.itertuples():
        res = rep["results"][r.game_pk]
        b = rep["board"][rep["board"].game_pk == r.game_pk]
        rows = "".join(
            f"<tr class='{'edge' if (x.edge if np.isfinite(x.edge) else 0) >= EDGE_MIN else ''}'><td>{html.escape(x.market)}</td>"
            f"<td>{html.escape(str(x.side))}{'' if x.line is None or (isinstance(x.line, float) and np.isnan(x.line)) else f' {x.line:+g}' if 'line' in x.market else f' {x.line:g}'}</td>"
            f"<td>{_pct(x.model_prob)}</td><td>{_am(x.fair_odds)}</td><td>{_am(x.dk_price)}</td>"
            f"<td>{_pct(x.dk_novig)}</td><td>{'' if not np.isfinite(x.edge) else f'{x.edge*100:+.1f}'}</td></tr>"
            for x in b.itertuples())
        cards.append(f"""
<section class="card"><header><h2>{html.escape(r.game)}</h2><span>{r.start_az} AZ · {html.escape(str(r.away_sp))} vs {html.escape(str(r.home_sp))}</span></header>
<div class="kpis"><div><b>{_pct(r.away_win)}</b><small>{html.escape(_abbr(res['away']))} win</small></div>
<div><b>{_pct(r.home_win)}</b><small>{html.escape(_abbr(res['home']))} win</small></div>
<div><b>{r.away_runs:.1f}–{r.home_runs:.1f}</b><small>projected</small></div>
<div><b>{r.total:.2f}</b><small>total (DK {'' if r.dk_total is None or not np.isfinite(r.dk_total) else f'{r.dk_total:g}'})</small></div>
<div><b>{_pct(r.nrfi)}</b><small>NRFI</small></div></div>
<table><thead><tr><th>Market</th><th>Side</th><th>Model</th><th>Fair</th><th>DK</th><th>DK no-vig</th><th>Edge</th></tr></thead><tbody>{rows}</tbody></table>
<p class="notes">{html.escape(str(r.notes))}</p></section>""")
    props = rep["props"]
    ptab = ""
    if len(props):
        top = props[props.edge >= EDGE_MIN].sort_values("ev", ascending=False).head(25)
        ptab = "".join(f"<tr><td>{html.escape(x.player)}</td><td>{html.escape(x.market)}</td><td>{x.side} {x.line:g}</td>"
                       f"<td>{x.model_mean:.2f}</td><td>{_pct(x.model_prob)}</td><td>{_am(x.dk_price)}</td><td>{_pct(x.dk_novig)}</td>"
                       f"<td>{x.edge*100:+.1f}</td></tr>" for x in top.itertuples())
    return f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MLB Simulator {rep['date']}</title><style>
:root{{--bg:#f6f5f1;--fg:#1c1c1a;--mut:#6b6a64;--card:#fff;--line:#e3e1da;--acc:#0b6e4f;--hi:#e8f4ee}}
@media (prefers-color-scheme:dark){{:root{{--bg:#151513;--fg:#ecebe6;--mut:#9a988f;--card:#1f1f1c;--line:#33322d;--acc:#4cc38a;--hi:#173327}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 -apple-system,system-ui,sans-serif;padding:16px}}
h1{{font-size:22px;margin:0 0 4px}} .sub{{color:var(--mut);margin-bottom:16px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:14px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;overflow-x:auto}}
.card header h2{{font-size:16px;margin:0}} .card header span{{color:var(--mut);font-size:12px}}
.kpis{{display:flex;gap:14px;margin:10px 0;flex-wrap:wrap}} .kpis b{{display:block;font-size:17px}} .kpis small{{color:var(--mut)}}
table{{width:100%;border-collapse:collapse;font-size:12.5px}} th,td{{text-align:left;padding:4px 6px;border-bottom:1px solid var(--line);white-space:nowrap}}
th{{color:var(--mut);font-weight:600}} tr.edge td{{background:var(--hi)}} .notes{{color:var(--mut);font-size:11.5px}}
@media (max-width:480px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body><h1>MLB Game Simulator · {rep['date']}</h1>
<div class="sub">{MODEL_VERSION} · plate-appearance Monte Carlo · DraftKings lines via ESPN · highlighted rows: model beats DK no-vig by ≥{EDGE_MIN*100:.0f} pts. Sept 2026 backtest: even with DK's closing moneyline, no proven profit at closing prices, playoff totals run high. Log and grade before betting.</div>
<div class="grid">{''.join(cards)}</div>
<section class="card" style="margin-top:14px"><header><h2>Props: biggest model edges vs DraftKings</h2></header>
<table><thead><tr><th>Player</th><th>Market</th><th>Bet</th><th>Model mean</th><th>Model</th><th>DK</th><th>DK no-vig</th><th>Edge</th></tr></thead><tbody>{ptab}</tbody></table></section>
</body></html>"""
