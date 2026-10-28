"""
NFL Game Simulator v6: one-command weekly report, prediction log, and closing line value.

    from nfl_weekly_v6 import *
    report = weekly_report(week=5)          # everything for the week, logged and exported
    g = explore_game(report, "CHI", "PHI", home_qb="Case Keenum", out=["C.Williams"])
    grade_log()                             # how logged predictions did vs results and the market

Or from a terminal:  python run_week.py --week 5

What weekly_report does
  1. Refreshes this season's data files if they're older than `refresh_hours`.
  2. Fits the game model and props model on every game played before the week starts.
  3. Simulates every game and every player's stat line (props are consistent with the
     game: the QB's passing is the sum of his receivers in each simulated game).
  4. Pulls DraftKings' current lines if an Odds API key is set up (odds_api_key.txt).
  5. Logs every prediction with the market line at that moment (logs/).
  6. Writes reports/<season>_wk<week>.xlsx and .html.

Closing line value (CLV): after a game, compare the market's closing line with the line when
the prediction was logged. If the market moved toward the model's number, the model saw
something the market later agreed with. Averaged over many games, that's the fastest honest
signal of whether the model has an edge.
"""

import datetime as dt
import glob
import html
import os
import pickle
import time

import numpy as np
import pandas as pd

import nfl_game_sim_v6 as ns
import nfl_props_v6 as props

DATA_DIR = "nfl_data"
LOG_DIR = "logs"
REPORT_DIR = "reports"
MODEL_VERSION = "v6"
GAME_LOG = os.path.join(LOG_DIR, "game_predictions.csv")
PROP_LOG = os.path.join(LOG_DIR, "prop_predictions.csv")
DK_HISTORY = "dk_props_history.csv"


# ================================================================== data
def current_season(today=None):
    today = today or dt.date.today()
    return today.year if today.month >= 3 else today.year - 1


def _season_files(season):
    return [f"{DATA_DIR}/games.csv", f"{DATA_DIR}/pbp_{season}.parquet",
            f"{DATA_DIR}/injuries_{season}.parquet", f"{DATA_DIR}/roster_weekly_{season}.parquet",
            f"{DATA_DIR}/depth_charts_{season}.parquet", f"{DATA_DIR}/snap_counts_{season}.parquet",
            f"{DATA_DIR}/stats_player_week_{season}.parquet"]


def refresh_data(season, refresh_hours=12, verbose=True):
    """Delete this season's data files older than `refresh_hours` so they re-download.
    Past seasons never change and are left alone."""
    removed = []
    for f in _season_files(season):
        if os.path.exists(f) and (time.time() - os.path.getmtime(f)) / 3600 > refresh_hours:
            os.remove(f)
            removed.append(os.path.basename(f))
    if verbose:
        print(f"Refreshing: {', '.join(removed)}" if removed else "Data is fresh (no download needed).")
    return removed


def load_data(season, verbose=True):
    """Everything the models need, cached in nfl_data/prep_cache_<season>_v6.pkl.
    The cache rebuilds automatically when any of this season's files is newer."""
    os.makedirs(DATA_DIR, exist_ok=True)
    cache = f"{DATA_DIR}/prep_cache_{season}_v6.pkl"
    files = [f for f in _season_files(season) if os.path.exists(f)]
    stale = (not os.path.exists(cache) or len(files) < len(_season_files(season))
             or any(os.path.getmtime(f) > os.path.getmtime(cache) for f in files))
    if not stale:
        with open(cache, "rb") as fh:
            data = pickle.load(fh)
        if "rz_rush" in data["prep"]:
            return data
        stale = True   # cache from before red-zone carries were tracked: rebuild once
    t0 = time.time()
    if verbose:
        print("Building data (first run of the week takes ~1-2 minutes)...")
    seasons = list(range(season - 3, season + 1))
    games, pbp, injuries = ns.load_all(seasons, data_dir=DATA_DIR)
    prep = ns.prepare(games, pbp)
    del pbp
    ns.add_qb_history(prep, range(season - 7, season - 1), data_dir=DATA_DIR)
    ns.add_referee_history(prep, range(season - 7, season - 3), data_dir=DATA_DIR)
    rosters_all = pd.concat([ns.load_rosters(y, DATA_DIR)[0] for y in seasons])
    ns.add_line_data(prep, seasons, rosters_all, injuries, data_dir=DATA_DIR)
    stats = props.load_player_stats(seasons, games, data_dir=DATA_DIR)
    data = {"season": season, "games": games, "prep": prep, "injuries": injuries, "stats": stats}
    with open(cache, "wb") as fh:
        pickle.dump(data, fh)
    if verbose:
        print(f"Data ready in {time.time() - t0:.0f}s.")
    return data


def _kickoffs(games):
    t = games.gametime.fillna("13:00")
    k = pd.to_datetime(games.gameday.astype(str) + " " + t).dt.tz_localize("America/New_York")
    return k.dt.tz_convert("UTC")


# ================================================================== the weekly report
def weekly_report(week, season=None, n_sims=10000, refresh=True, refresh_hours=12, pull_dk=True,
                  log=True, export=True, weather=None, referees=None, live_injuries=True, verbose=True,
                  lines="auto"):
    """Simulate every game and prop for a week, compare with DraftKings, log, and export.
    lines: "auto" (Odds API if odds_api_key.txt exists, else FanDuel, else ESPN), "fanduel" (free,
           real prices), "espn" (free DraftKings lines, -110 assumed), "odds_api" (uses credits)
    weather:  {"PHI": {"temp": 38, "wind": 18}, ...} keyed by home team (optional)
    referees: {"PHI": "Shawn Hochuli", ...} keyed by home team (optional)"""
    season = season or current_season()
    if refresh:
        refresh_data(season, refresh_hours, verbose)
    D = load_data(season, verbose)
    games, prep, injuries, stats = D["games"], D["prep"], D["injuries"], D["stats"]
    slate = games[(games.season == season) & (games.week == week)].copy()
    if slate.empty:
        raise ValueError(f"No games found for {season} week {week}.")
    slate["kickoff_utc"] = _kickoffs(slate)
    as_of = pd.Timestamp(slate.gameday.min())
    model = ns.fit_model(prep, as_of)
    P = props.fit_props(model, stats)
    rosters, depth = ns.load_rosters(season, DATA_DIR)
    ctx = ns.build_context(season, week, rosters, injuries, depth)
    inj_week = injuries[(injuries.season == season)].week.max() if len(injuries) else None
    inj_note = (f"Week {week} injury report" if inj_week == week else
                f"Week {inj_week} injury report carried forward at reduced weight (week {week} not out yet)")
    upcoming = slate.kickoff_utc.max() > pd.Timestamp.now(tz="UTC")
    if live_injuries and upcoming:
        # official reports lag (Wed-Fri); ESPN's live list updates within hours of news
        live, stamp = ns.fetch_live_injuries(DATA_DIR, max_age_hours=0 if refresh else 3)
        if len(live):
            ctx = ns.apply_live_injuries(ctx, live, rosters, stamp)
            when = pd.Timestamp(stamp).tz_convert("America/Phoenix").strftime("%a %b %d, %I:%M %p") if stamp else "now"
            inj_note += f"; live ESPN injury list as of {when} Arizona time has the final word"

    game_rows, prop_frames, sims = [], [], {}
    for _, x in slate.iterrows():
        h, a = x.home_team, x.away_team
        g = ns.simulate_game(model, h, a, n_sims, neutral=x.location == "Neutral", seed=42, ctx=ctx,
                             weather=(weather or {}).get(h), referee=(referees or {}).get(h))
        sp = props.simulate_props(model, P, h, a, sim=g, ctx=ctx)
        sims[(h, a)] = (g, sp)
        gi = g["inputs"]
        game_rows.append({
            "game_id": x.game_id, "kickoff_utc": x.kickoff_utc, "matchup": f"{a} @ {h}", "home": h, "away": a,
            "away_qb": gi["away"]["notes"]["qb"], "home_qb": gi["home"]["notes"]["qb"],
            "home_win_pct": g["home_win_pct"], "model_line": g["proj_spread"],
            "vegas_line": -x.spread_line if pd.notna(x.spread_line) else np.nan,
            "model_total": g["proj_total"], "vegas_total": x.total_line,
            "proj_home": g["proj_home_pts"], "proj_away": g["proj_away_pts"],
            "played": pd.notna(x.result), "final": (f"{a} {x.away_score:.0f} - {h} {x.home_score:.0f}"
                                                    if pd.notna(x.result) else "")})
        pf = sp.medians.copy()
        mean = sp.means.set_index("player_id")
        for c in ("pass_yds", "rush_yds", "rec_yds", "receptions", "pass_att", "completions", "rush_att", "targets", "pass_tds"):
            if c in pf:
                pf[c + "_avg"] = pf.player_id.map(mean[c]) if c in mean else np.nan
        pf["game_id"], pf["kickoff_utc"] = x.game_id, x.kickoff_utc
        prop_frames.append(pf)
    games_df = pd.DataFrame(game_rows)
    games_df["line_diff"] = games_df.model_line - games_df.vegas_line
    games_df["total_diff"] = games_df.model_total - games_df.vegas_total
    props_df = pd.concat(prop_frames, ignore_index=True)

    # DraftKings
    dk_ou = dk_yes = None
    lines_note = "no DraftKings lines pulled"
    edges = pd.DataFrame()
    if pull_dk:
        try:
            dk_ou, dk_yes, lines_note = get_lines(season, week, lines, refresh=refresh, verbose=verbose)
        except Exception as e:  # never let the odds feed break the report
            print(f"DraftKings lines skipped: {e}")
    if dk_ou is not None:
        edges = _dk_edges(sims, dk_ou, dk_yes)
        if len(edges):
            book = "FanDuel" if lines_note.startswith("FanDuel") else "DraftKings"
            edges.insert(2, "book", book + (" (-110 assumed)" if "via ESPN" in lines_note else ""))

    report = {"season": season, "week": week, "as_of": as_of, "model": model, "P": P, "ctx": ctx,
              "games": games_df, "props": props_df, "sims": sims, "dk_ou": dk_ou, "dk_yes": dk_yes,
              "edges": edges, "injury_note": inj_note, "lines_note": lines_note, "scoring_level": model.get("scoring_level"),
              "injuries": injury_table(ctx, slate),
              "data": D}
    if log:
        n_g, n_p = log_predictions(report)
        report["logged"] = (n_g, n_p)
    report["track_record"] = grade_log(verbose=False)
    if export:
        report["files"] = export_report(report)
    if verbose:
        _print_summary(report)
    return report


ESPN_LINES = "dk_props_espn.csv"
FD_LINES = "fd_props_current.csv"


def _load_lines(path, season, week):
    ou, yes = props.load_dk_lines(path)
    ou = ou[(ou.season == season) & (ou.week == week)].reset_index(drop=True)
    yes = yes[(yes.season == season) & (yes.week == week)].reset_index(drop=True)
    snap = pd.to_datetime(pd.read_csv(path).snapshot, utc=True).max().tz_convert("America/Phoenix")
    return ou, yes, f"{snap:%a %b %d, %I:%M %p} Arizona time"


def get_lines(season, week, source="auto", refresh=True, verbose=True):
    """Sportsbook prop lines for the week. Returns (over/under lines, anytime-TD prices, note).
    source:
      "auto"      The Odds API if a key is set up, otherwise FanDuel, otherwise ESPN
      "odds_api"  DraftKings lines + prices (needs odds_api_key.txt, uses credits)
      "fanduel"   FanDuel lines + prices, free, no key (no attempts/completions/carries)
      "espn"      DraftKings lines via ESPN, free, no key (no prices: -110 assumed; no anytime TD)"""
    import fetch_dk_props as fdk
    import fetch_espn_props as fep
    import fetch_fanduel_props as ffd
    order = {"auto": ["odds_api", "fanduel", "espn"], "odds_api": ["odds_api", "fanduel", "espn"],
             "fanduel": ["fanduel", "espn"], "espn": ["espn"]}[source]
    for src in order:
        try:
            if src == "odds_api":
                fdk.api_key()   # raises if no key is set up
                ou, yes = props.auto_dk_lines(max_age_hours=6)
                if ou is not None and len(ou):
                    return ou, yes, "DraftKings lines and prices via The Odds API"
            elif src == "fanduel":
                if refresh or not os.path.exists(FD_LINES):
                    ffd.fetch_fanduel_props(week, season, FD_LINES, verbose=verbose)
                if os.path.exists(FD_LINES) and len(pd.read_csv(FD_LINES)):
                    ou, yes, when = _load_lines(FD_LINES, season, week)
                    if len(ou) or len(yes):
                        return ou, yes, (f"FanDuel lines and prices as of {when} (no attempts, completions "
                                         f"or carries)")
            else:
                if refresh or not os.path.exists(ESPN_LINES):
                    fep.fetch_espn_props(week, season, ESPN_LINES, DATA_DIR, verbose=verbose)
                if os.path.exists(ESPN_LINES) and len(pd.read_csv(ESPN_LINES)):
                    ou, yes, when = _load_lines(ESPN_LINES, season, week)
                    if len(ou):
                        return ou, yes, (f"DraftKings lines via ESPN as of {when} (lines only: prices "
                                         f"assumed -110, no anytime-TD prices)")
        except Exception as e:   # one feed failing shouldn't stop the next
            if src != "odds_api" or source == "odds_api":
                print(f"{src} lines unavailable: {e}")
        if verbose and src != order[-1]:
            print(f"No {src} lines; trying {order[order.index(src) + 1]}.")
    return None, None, "no sportsbook lines found"


def injury_table(ctx, slate):
    """Everyone the model is treating as out or limited for this week's teams, and why."""
    teams = set(slate.home_team) | set(slate.away_team)
    rows = []
    live = ctx.get("live_injuries")
    names = {}
    if live is not None and len(live):
        names = live.dropna(subset=["gsis_id"]).drop_duplicates("gsis_id").set_index("gsis_id")[["player", "pos", "comment"]]
    team_now = ctx.get("team_now")
    src = ctx.get("src", {})
    for g, w in ctx["miss"].items():
        team = team_now.get(g) if team_now is not None else None
        if team not in teams or w <= 0 or src.get(g) == "reserve":
            continue   # long-term roster reserve lists (preseason IR/PUP) aren't news; still applied
        nm = ctx.get("names", pd.Series(dtype=str)).get(g, g)
        pos, note = "", ""
        if len(names) and g in names.index:
            nm, pos, note = names.loc[g, "player"], names.loc[g, "pos"], names.loc[g, "comment"]
        rows.append({"Team": team, "Player": nm, "Pos": pos, "Status used": ctx["why"].get(g, ""),
                     "Chance out": f"{w:.0%}", "Latest news": note})
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    t["_w"] = t["Chance out"].str.rstrip("%").astype(float)
    return t.sort_values(["Team", "_w"], ascending=[True, False]).drop(columns="_w").reset_index(drop=True)


def _dk_edges(sims, dk_ou, dk_yes):
    """Every DraftKings prop next to the model: probabilities counted from the simulations."""
    rows = []
    for (h, a), (g, sp) in sims.items():
        ou = dk_ou[(dk_ou.home == h) & (dk_ou.away == a)]
        for _, r in ou.iterrows():
            name = sp._name(r.player)
            if (name, r.market) not in sp.draws or pd.isna(r.over_price) or pd.isna(r.under_price):
                continue
            p = sp.prob_over(name, r.market, r.line)
            pick_over = p >= 0.5
            prob = p if pick_over else 1 - p
            price = r.over_price if pick_over else r.under_price
            rows.append({"player": name, "game": f"{a} @ {h}", "market": r.market, "line": r.line,
                         "model_median": float(np.median(sp.draws[(name, r.market)])),
                         "pick": "over" if pick_over else "under", "model_prob": prob,
                         "dk_price": props._american(price), "fair_price": props._american_from_prob(prob),
                         "ev_per_unit": prob * price - 1})
        yes = dk_yes[(dk_yes.home == h) & (dk_yes.away == a)] if dk_yes is not None else pd.DataFrame()
        for _, r in yes.iterrows():
            name = sp._name(r.player)
            if (name, "anytime_td") not in sp.draws:
                continue
            p = sp.prob_over(name, "anytime_td", None)
            rows.append({"player": name, "game": f"{a} @ {h}", "market": "anytime_td", "line": np.nan,
                         "model_median": np.nan, "pick": "yes", "model_prob": p,
                         "dk_price": props._american(r.yes_price), "fair_price": props._american_from_prob(p),
                         "ev_per_unit": p * r.yes_price - 1})
    e = pd.DataFrame(rows)
    return e.sort_values("ev_per_unit", ascending=False).reset_index(drop=True) if len(e) else e


# ================================================================== explore one game
class GameView:
    """One game with your overrides. .game = game simulation, .props = SimProps."""

    def __init__(self, game, sp):
        self.game, self.props = game, sp

    def qbs(self):
        return position_table(self.props, "QB")

    def rbs(self):
        return position_table(self.props, "RB")

    def receivers(self):
        return position_table(self.props, "WR/TE")

    def combo(self, legs, dk_price=None):
        return self.props.joint_prob(legs, dk_price)


def explore_game(report, home, away, home_qb=None, away_qb=None, out=None, weather=None, referee=None,
                 neutral=False, n_sims=20000, show=True):
    """Re-simulate one game with your own settings, using the week's fitted models."""
    m, P, ctx = report["model"], report["P"], report["ctx"]
    g = ns.simulate_game(m, home, away, n_sims, neutral=neutral, seed=42, home_qb=home_qb, away_qb=away_qb,
                         out_players=out, ctx=ctx, weather=weather, referee=referee)
    sp = props.simulate_props(m, P, home, away, sim=g, ctx=ctx, out_players=out)
    if show:
        ns.print_game(g)
    return GameView(g, sp)


def position_table(sp, group, stat="median"):
    """QB / RB / WR-TE table from a SimProps, with anytime-TD fair odds.
    stat="median" (closest to how books set lines) or "mean"."""
    df = (sp.medians if stat == "median" else sp.means).copy()
    cols = {"QB": ["player", "team", "pass_att", "completions", "pass_yds", "pass_tds", "rush_att", "rush_yds", "anytime_td"],
            "RB": ["player", "team", "rush_att", "rush_yds", "targets", "receptions", "rec_yds", "anytime_td"],
            "WR/TE": ["player", "pos", "team", "targets", "receptions", "rec_yds", "anytime_td"]}[group]
    pos = {"QB": ["QB"], "RB": ["RB", "FB"], "WR/TE": ["WR", "TE"]}[group]
    t = df[df.pos.isin(pos)][cols].copy()
    t["td_odds"] = t.anytime_td.apply(props._american_from_prob)
    t["anytime_td"] = (t.anytime_td * 100).round(0).astype(int).astype(str) + "%"
    sort = {"QB": ["team"], "RB": ["team", "rush_yds"], "WR/TE": ["team", "rec_yds"]}[group]
    asc = [True] + [False] * (len(sort) - 1)
    return t.round(1).sort_values(sort, ascending=asc).reset_index(drop=True)


# ================================================================== prediction log
def _append(path, df):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def log_predictions(report):
    """Save predictions for games that haven't kicked off, with the market line right now."""
    now = pd.Timestamp.now(tz="UTC")
    stamp = now.isoformat(timespec="seconds")
    g = report["games"]
    g = g[g.kickoff_utc > now]
    if g.empty:
        return 0, 0
    gl = pd.DataFrame({
        "logged_at": stamp, "model_version": MODEL_VERSION, "season": report["season"], "week": report["week"],
        "game_id": g.game_id, "home": g.home, "away": g.away, "home_qb": g.home_qb, "away_qb": g.away_qb,
        "model_home_win": g.home_win_pct.round(4),
        "model_margin": (-g.model_line).round(2),            # home points minus away points
        "market_margin_at_log": (-g.vegas_line).round(2),
        "model_total": g.model_total.round(2), "market_total_at_log": g.vegas_total})
    _append(GAME_LOG, gl)
    n_p = 0
    e = report.get("edges")
    if e is not None and len(e):
        e = e.copy()
        gid = {f"{r.away} @ {r.home}": r.game_id for _, r in g.iterrows()}
        e["game_id"] = e.game.map(gid)
        e = e[e.game_id.notna()]
        if len(e):
            pl = pd.DataFrame({
                "logged_at": stamp, "model_version": MODEL_VERSION, "season": report["season"],
                "week": report["week"], "game_id": e.game_id, "player": e.player, "market": e.market,
                "dk_line_at_log": e.line, "dk_price_at_log": e.dk_price, "pick": e.pick,
                "model_prob": e.model_prob.round(4), "model_median": e.model_median.round(2),
                "ev_per_unit": e.ev_per_unit.round(4)})
            _append(PROP_LOG, pl)
            n_p = len(pl)
    return len(gl), n_p


def grade_log(season=None, verbose=True):
    """Grade every logged prediction whose game is final: results vs the logged line, and
    closing line value (did the market move toward the model after it was logged?)."""
    out = {}
    if not os.path.exists(GAME_LOG):
        if verbose:
            print("No predictions logged yet. weekly_report() logs them automatically.")
        return out
    L = pd.read_csv(GAME_LOG)
    if season:
        L = L[L.season == season]
    games = pd.read_csv(f"{DATA_DIR}/games.csv", low_memory=False)
    games["kick"] = _kickoffs(games)
    L["logged_at"] = pd.to_datetime(L.logged_at, utc=True)
    L = L.merge(games[["game_id", "kick", "result", "total", "spread_line", "total_line"]], on="game_id", how="left")
    L = L[L.logged_at < L.kick]
    last = L.sort_values("logged_at").groupby("game_id").tail(1)       # final prediction before kickoff
    done = last[last.result.notna()].copy()
    if done.empty:
        if verbose:
            print(f"{len(last)} games logged; none finished yet.")
        out["games"] = pd.DataFrame()
        return out
    d = done
    pick = np.sign(d.model_margin - d.market_margin_at_log)
    ats = d.result - d.market_margin_at_log
    d["ats"] = np.where((ats == 0) | (pick == 0), "push", np.where(np.sign(ats) == pick, "W", "L"))
    d["spread_clv"] = (d.spread_line - d.market_margin_at_log) * pick   # + = market moved toward model
    tp = np.sign(d.model_total - d.market_total_at_log)
    ou = d.total - d.market_total_at_log
    d["ou"] = np.where((ou == 0) | (tp == 0), "push", np.where(np.sign(ou) == tp, "W", "L"))
    d["total_clv"] = (d.total_line - d.market_total_at_log) * tp
    d["winner_right"] = (d.model_home_win > 0.5) == (d.result > 0)
    w, l = int((d.ats == "W").sum()), int((d.ats == "L").sum())
    ow, ol = int((d.ou == "W").sum()), int((d.ou == "L").sum())
    summ = {"games graded": len(d), "winner %": f"{d.winner_right.mean():.1%}",
            "spread miss (model)": round((d.model_margin - d.result).abs().mean(), 1),
            "spread miss (market at log)": round((d.market_margin_at_log - d.result).abs().mean(), 1),
            "ATS vs logged line": f"{w}-{l} ({w / max(w + l, 1):.1%})",
            "spread CLV (avg pts)": round(d.spread_clv.mean(), 2),
            "beat the closing spread": f"{(d.spread_clv > 0).mean():.1%} (moved away: {(d.spread_clv < 0).mean():.1%})",
            "O/U vs logged total": f"{ow}-{ol} ({ow / max(ow + ol, 1):.1%})",
            "total CLV (avg pts)": round(d.total_clv.mean(), 2)}
    out["games"], out["game_summary"] = d, summ

    if os.path.exists(PROP_LOG):
        pl = pd.read_csv(PROP_LOG)
        pl["logged_at"] = pd.to_datetime(pl.logged_at, utc=True)
        pl = pl.merge(games[["game_id", "kick"]], on="game_id", how="left")
        pl = pl[pl.logged_at < pl.kick].sort_values("logged_at").groupby(["game_id", "player", "market"]).tail(1)
        pl = _attach_prop_results(pl)
        pl = _attach_dk_close(pl)
        pd_ = pl[pl.actual.notna()].copy()
        if len(pd_):
            ou_m = pd_.market != "anytime_td"
            won = np.where(ou_m, np.where(pd_.pick == "over", pd_.actual > pd_.dk_line_at_log, pd_.actual < pd_.dk_line_at_log),
                           pd_.actual >= 1)
            push = ou_m & (pd_.actual == pd_.dk_line_at_log)
            dec = pd_.dk_price_at_log.map(props._decimal_from_american)
            pd_["profit"] = np.where(push, 0.0, np.where(won, dec - 1, -1.0))
            pd_["won"] = np.where(push, np.nan, won.astype(float))
            s = np.where(pd_.pick == "over", 1, -1)
            pd_["line_clv"] = np.where(ou_m, (pd_.dk_close_line - pd_.dk_line_at_log) * s, np.nan)
            out["props"] = pd_
            out["prop_summary"] = {
                "props graded": len(pd_), "record": f"{int(np.nansum(pd_.won))}-{int((pd_.won == 0).sum())}",
                "profit per $1": round(pd_.profit.mean(), 3),
                "line CLV (avg)": round(np.nanmean(pd_.line_clv), 2) if ou_m.any() else np.nan,
                "beat the closing line": f"{np.nanmean(pd_.line_clv > 0):.1%}" if ou_m.any() else "n/a"}
    if verbose:
        print("Games:", summ)
        if "prop_summary" in out:
            print("Props:", out["prop_summary"])
        if len(d) < 100:
            print(f"Only {len(d)} games graded: records this small are mostly luck. CLV becomes informative first.")
    return out


def _attach_prop_results(pl):
    stats_files = sorted(glob.glob(f"{DATA_DIR}/stats_player_week_*.parquet"))
    if not stats_files or pl.empty:
        pl["actual"] = np.nan
        return pl
    st = pd.concat([pd.read_parquet(f, columns=["game_id", "player_display_name", "passing_yards", "attempts",
                                                "completions", "passing_tds", "rushing_yards", "carries",
                                                "receiving_yards", "receptions", "rushing_tds", "receiving_tds"])
                    for f in stats_files[-2:]])
    st["anytime_td"] = st.rushing_tds + st.receiving_tds
    st["key"] = st.player_display_name.map(props._norm)
    col = dict(props.MARKETS, anytime_td="anytime_td")
    pl = pl.copy()
    pl["key"] = pl.player.map(props._norm)
    st = st.drop_duplicates(["game_id", "key"]).set_index(["game_id", "key"])
    pl["actual"] = [st[col[mk]].get((gid, k), np.nan) if mk in col else np.nan
                    for gid, k, mk in zip(pl.game_id, pl.key, pl.market)]
    return pl


def _attach_dk_close(pl):
    pl = pl.copy()
    pl["dk_close_line"] = np.nan
    if not os.path.exists(DK_HISTORY) or pl.empty:
        return pl
    h = pd.read_csv(DK_HISTORY)
    h["market"] = h.market.map(props.ODDS_API_MARKETS)
    h = h[h.market.notna() & (h.name.isin(["Over", "Under"]))]
    h["snapshot"] = pd.to_datetime(h.snapshot, utc=True)
    h["commence"] = pd.to_datetime(h.commence_time, utc=True)
    h = h[h.snapshot < h.commence]
    h["key"] = h.player.map(props._norm)
    over = h[h.name == "Over"].sort_values("snapshot")
    # main line = the one priced closest to even at the last pre-kickoff snapshot
    last = over.groupby(["key", "market"]).snapshot.transform("max")
    over = over[over.snapshot == last]
    close = over.assign(gap=(over.price - 1.91).abs()).sort_values("gap").groupby(["key", "market"]).point.first()
    pl["key"] = pl.player.map(props._norm)
    pl["dk_close_line"] = [close.get((k, mk), np.nan) for k, mk in zip(pl.key, pl.market)]
    return pl


# ================================================================== export
def export_report(report):
    os.makedirs(REPORT_DIR, exist_ok=True)
    base = os.path.join(REPORT_DIR, f"{report['season']}_wk{report['week']:02d}")
    tables = _report_tables(report)
    files = []
    try:
        import openpyxl  # noqa: F401
        with pd.ExcelWriter(base + ".xlsx") as xw:
            for name, df in tables.items():
                df.to_excel(xw, sheet_name=name[:31], index=False)
        files.append(base + ".xlsx")
    except ImportError:
        for name, df in tables.items():
            df.to_csv(f"{base}_{name.replace(' ', '_').replace('/', '-')}.csv", index=False)
            files.append(f"{base}_{name}.csv")
    with open(base + ".html", "w", encoding="utf-8") as f:
        f.write(_html(report, tables))
    files.append(base + ".html")
    return files


def games_table(report):
    """The week's games: model vs Vegas, the model's side and total lean."""
    return _report_tables(report)["Games"]


def _report_tables(report):
    g = report["games"].copy()
    side = np.where(g.model_line < g.vegas_line, g.home + " " + g.vegas_line.map(lambda v: f"{v:+g}"),
                    g.away + " " + (-g.vegas_line).map(lambda v: f"{v:+g}"))
    lean = np.where(g.model_total > g.vegas_total, "Over ", "Under ") + g.vegas_total.map(lambda v: f"{v:g}")
    games = pd.DataFrame({
        "Matchup": g.matchup, "QBs (away / home)": g.away_qb + " / " + g.home_qb,
        "Home win %": (g.home_win_pct * 100).round(1), "Model line": g.model_line.round(1),
        "Vegas line": g.vegas_line, "Model's side": np.where(g.vegas_line.isna(), "", side),
        "Spread gap": g.line_diff.abs().round(1), "Model total": g.model_total.round(1),
        "Vegas total": g.vegas_total, "Model's total lean": np.where(g.vegas_total.isna(), "", lean),
        "Total gap": g.total_diff.abs().round(1), "Final": g.final})
    qb, rb, wr = [], [], []
    for (h, a), (_, sp) in report["sims"].items():
        qb.append(position_table(sp, "QB"))
        rb.append(position_table(sp, "RB"))
        wr.append(position_table(sp, "WR/TE"))
    nice = {"player": "Player", "pos": "Pos", "team": "Team", "pass_att": "Att", "completions": "Cmp",
            "pass_yds": "Pass yds", "pass_tds": "Pass TD", "rush_att": "Rush att", "rush_yds": "Rush yds",
            "targets": "Targets", "receptions": "Rec", "rec_yds": "Rec yds", "anytime_td": "Anytime TD",
            "td_odds": "TD fair odds"}
    tables = {"Games": games, "QBs": pd.concat(qb, ignore_index=True).rename(columns=nice),
              "RBs": pd.concat(rb, ignore_index=True).rename(columns=nice),
              "WR-TE": pd.concat(wr, ignore_index=True).rename(columns=nice)}
    if report.get("injuries") is not None and len(report["injuries"]):
        tables["Injuries"] = report["injuries"]
    e = report.get("edges")
    if e is not None and len(e):
        tables["Sportsbook"] = e.assign(model_prob=(e.model_prob * 100).round(1), ev_per_unit=(e.ev_per_unit * 100).round(1),
                                        model_median=e.model_median.round(1)).rename(
            columns={"model_prob": "model %", "ev_per_unit": "EV per $100"})
    tr = report.get("track_record") or {}
    if "game_summary" in tr:
        rows = [{"Metric": k, "Value": v} for k, v in tr["game_summary"].items()]
        rows += [{"Metric": "props: " + k, "Value": v} for k, v in tr.get("prop_summary", {}).items()]
        tables["Track record"] = pd.DataFrame(rows)
    return tables


def _html(report, tables):
    def tab(df, tid):
        head = "".join(f"<th onclick=\"sortT('{tid}',{i})\">{html.escape(str(c))}</th>" for i, c in enumerate(df.columns))
        body = []
        for _, r in df.iterrows():
            cells = []
            for c in df.columns:
                v = r[c]
                if isinstance(v, float) and np.isnan(v):
                    v = ""
                cls = ""
                if c == "EV per $100" and v != "":
                    cls = " class='pos'" if float(v) > 0 else " class='neg'"
                cells.append(f"<td{cls}>{html.escape(str(v))}</td>")
            body.append("<tr>" + "".join(cells) + "</tr>")
        return f"<div class='scroll'><table id='{tid}'><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"
    names = list(tables)
    tabs = "".join(f"<button class='tab{' on' if i == 0 else ''}' onclick=\"show({i})\">{html.escape(n)}</button>" for i, n in enumerate(names))
    panes = "".join(f"<section class='pane{' on' if i == 0 else ''}'>{tab(tables[n], f't{i}')}</section>" for i, n in enumerate(names))
    lvl = report.get("scoring_level") or {}
    notes = [f"Ratings use games before {report['as_of']:%b %d, %Y}.", report["injury_note"] + "."]
    if report.get("lines_note"):
        notes.append(report["lines_note"] + ".")
    if lvl.get("season_so_far"):
        notes.append(f"League scoring this season: {lvl['season_so_far']:.1f} points per game "
                     f"({lvl['games_so_far']} games) vs {lvl['prior_total']:.1f} the last two seasons.")
    notes.append("Lines use the Vegas convention: negative = home team favored. Prop numbers are medians "
                 "(the middle outcome, where sportsbooks set lines). Odds are fair odds with no sportsbook margin.")
    stamp = pd.Timestamp.now(tz="America/Phoenix").strftime("%a %b %d, %I:%M %p")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NFL Week {report['week']} Model</title>
<style>
:root{{--bg:#f7f7f5;--card:#fff;--ink:#1d1d1b;--muted:#6b6b66;--line:#e4e3de;--acc:#1f4e8c;--pos:#1b7a3d;--neg:#b3261e}}
@media (prefers-color-scheme:dark){{:root{{--bg:#151514;--card:#1f1f1d;--ink:#ecebe6;--muted:#a3a29c;--line:#34332f;--acc:#8fb4ea;--pos:#6fcf8f;--neg:#f08a80}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 -apple-system,Segoe UI,Roboto,sans-serif}}
main{{max-width:1180px;margin:0 auto;padding:24px 16px 48px}}h1{{font-size:22px;margin:0 0 4px}}.sub{{color:var(--muted);margin:0 0 16px}}
.tabs{{display:flex;flex-wrap:wrap;gap:6px;margin:16px 0}}.tab{{border:1px solid var(--line);background:var(--card);color:var(--ink);padding:6px 12px;border-radius:999px;cursor:pointer;font:inherit}}
.tab.on{{background:var(--acc);color:var(--bg);border-color:var(--acc)}}.pane{{display:none}}.pane.on{{display:block}}
.scroll{{overflow-x:auto;background:var(--card);border:1px solid var(--line);border-radius:10px}}
table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}th,td{{padding:7px 10px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}}
th{{position:sticky;top:0;background:var(--card);cursor:pointer;font-weight:600;color:var(--muted)}}tr:last-child td{{border-bottom:0}}
td.pos{{color:var(--pos)}}td.neg{{color:var(--neg)}}ul{{color:var(--muted);padding-left:18px;margin:8px 0 0}}
</style></head><body><main>
<h1>NFL {report['season']} Week {report['week']}: model vs market</h1>
<p class="sub">Model {MODEL_VERSION} &middot; generated {stamp} Arizona time</p>
<ul>{''.join(f'<li>{html.escape(n)}</li>' for n in notes)}</ul>
<div class="tabs">{tabs}</div>{panes}
</main><script>
function show(i){{document.querySelectorAll('.tab').forEach((b,j)=>b.classList.toggle('on',i===j));document.querySelectorAll('.pane').forEach((p,j)=>p.classList.toggle('on',i===j));}}
function sortT(id,c){{const t=document.getElementById(id),b=t.tBodies[0],r=[...b.rows],d=t.dataset.c==c&&t.dataset.d!='d'?'d':'a';
t.dataset.c=c;t.dataset.d=d;const v=x=>{{const s=x.cells[c].innerText.replace('%','').replace('+','');const n=parseFloat(s);return isNaN(n)?s:n;}};
r.sort((x,y)=>{{const a=v(x),bb=v(y);return (a>bb?1:a<bb?-1:0)*(d=='a'?1:-1);}});r.forEach(x=>b.appendChild(x));}}
</script></body></html>"""


def _print_summary(report):
    g = report["games"]
    print(f"\n{report['season']} week {report['week']} | {report['injury_note']}")
    show = g[["matchup", "home_win_pct", "model_line", "vegas_line", "model_total", "vegas_total"]].copy()
    show["home_win_pct"] = (show.home_win_pct * 100).round(1)
    print(show.round(1).to_string(index=False))
    if report.get("edges") is not None and len(report["edges"]):
        print(f"\nSportsbook props compared: {len(report['edges'])} (see the Sportsbook tab)")
    if report.get("lines_note"):
        print("Lines:", report["lines_note"])
    if report.get("logged"):
        print(f"Logged {report['logged'][0]} game predictions and {report['logged'][1]} prop predictions.")
    for f in report.get("files", []):
        print("Saved:", f)


# ================================================================== TD favorites
def td_favorites(report, top=1):
    """Most likely touchdown scorer(s) in each game = the shortest fair anytime-TD odds.
    Probabilities come straight from the simulations. If DraftKings lines were pulled,
    their price and the model's expected value per $1 are shown alongside."""
    dk = report.get("dk_yes")
    rows = []
    for (h, a), (g, sp) in report["sims"].items():
        kick = report["games"].set_index("matchup").kickoff_utc.get(f"{a} @ {h}")
        m = sp.means.sort_values("anytime_td", ascending=False).head(top)
        for rank, (_, r) in enumerate(m.iterrows(), 1):
            row = {"Game": f"{a} @ {h}",
                   "Kickoff (AZ)": kick.tz_convert("America/Phoenix").strftime("%a %I:%M %p") if kick is not None else "",
                   "Rank": rank, "Player": r.player, "Pos": r.pos, "Team": r.team,
                   "Model TD %": round(r.anytime_td * 100, 1),
                   "Fair odds": props._american_from_prob(r.anytime_td),
                   "Expected TDs": round(r.exp_tds, 2)}
            if dk is not None and len(dk):
                d = dk[(dk.home == h) & (dk.away == a)]
                d = d[d.player.map(sp._name) == r.player]
                if len(d):
                    price = float(d.yes_price.iloc[0])
                    row["Book price"] = props._american(price)
                    row["EV per $1"] = round(r.anytime_td * price - 1, 3)
            rows.append(row)
    out = pd.DataFrame(rows)
    if top == 1 and len(out):
        out = out.drop(columns="Rank")
    return out


# ================================================================== IR / injury moves
def ir_moves(report, since=None, statuses=("Injured Reserve", "Physically Unable to Perform", "Out", "Suspension"),
             teams=None):
    """Players ESPN lists as IR / PUP / Out / suspended, newest first, with whether the model
    has them ruled out for this run. since: "2026-09-28" (default: 4 days before the run).
    in_model = False means the player didn't match a roster ID; rule him out with out=[...]."""
    ctx = report["ctx"]
    live = ctx.get("live_injuries")
    if live is None or not len(live):
        print("No live injury list in this report (run weekly_report with live_injuries=True).")
        return pd.DataFrame()
    stamp = pd.Timestamp(ctx.get("live_stamp") or pd.Timestamp.now(tz="UTC"))
    stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp
    since = pd.Timestamp(since, tz="UTC") if since else stamp - pd.Timedelta(days=4)
    x = live[live.status.isin(statuses) & (live.date >= since)]
    if teams:
        x = x[x.team.isin(teams)]
    x = x.sort_values("date", ascending=False)
    out = pd.DataFrame({
        "Player": x.player, "Team": x.team, "Pos": x.pos, "Status": x.status,
        "Listed (AZ)": x.date.dt.tz_convert("America/Phoenix").dt.strftime("%a %b %d %I:%M %p"),
        "Injury": x.injury,
        "In model": [ctx["miss"].get(g, 0) >= 1.0 if isinstance(g, str) else False for g in x.gsis_id]})
    print(f"ESPN injury list as of {stamp.tz_convert('America/Phoenix'):%a %b %d %I:%M %p} Arizona time; "
          f"{len(out)} players listed since {since.tz_convert('America/Phoenix'):%b %d}.")
    return out.reset_index(drop=True)
