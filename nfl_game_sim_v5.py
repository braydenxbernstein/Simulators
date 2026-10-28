"""
NFL game simulator v5: play-by-play powered, possession-based.
Data: nflverse (github.com/nflverse). Usage: see NFL_Game_Simulator_v5.ipynb.

Game engine
  A real game clock (two 30-minute halves + overtime). Every possession starts
  at a field position, plays out as a drive whose result (TD, FG, missed FG,
  punt, turnover, turnover on downs) and duration are drawn from nflverse
  drive data, adjusted for the two teams. The next possession starts where the
  last one left the ball: short fields after turnovers, pick-sixes, kickoffs
  after scores. Possession counts vary game to game, like real football.

What moves the odds
  Team efficiency   opponent-adjusted EPA/play (pass and run separately) + points
  Starting QB       QB EPA per dropback; swapping QBs moves the offense
  QB mistakes       QB turnover rate (INT + lost fumbles) drives the turnover odds
  Rosters           current weekly rosters + depth chart; IR / PUP / NFI / suspended = out,
                    injury report weighted by status, players who changed teams carry
                    their value to the new team
  Skill players     WR/TE/RB value above replacement, removed when unavailable
  Game plan         pass rate vs defenses similar to this opponent
  Style matchup     how the offense has done vs defenses similar to this one
  4th-down coaching go-rate over expected (punt zone and FG zone)
  Halftime          coach 2nd-half vs 1st-half efficiency swing
  Head-to-head      past meetings within 2 years (heavily shrunk)
  Weather / roof    wind, cold, indoor (forecast input or stadium monthly average)
  Rest / travel     rest days, short weeks, West Coast teams in early Eastern kickoffs
  Pass rush         pressure allowed vs pressure generated, log5 matchup
  Kicker            depth-chart kicker's accuracy vs league by distance
  Red zone          TD vs FG finishing inside the 20 (heavily shrunk)
  Referee           crew chief flag rate and home/away flag split
  OL / defense      starters missing (snap counts + injury report) vs the team's usual level,
                    by position group, defenders weighted by pass-rush / coverage production

Player props live in nfl_props_v5.py and only read this module's output.

Every added effect is estimated from data and shrunk by its reliability, so factors
without a measurable effect show up in the notes but barely move the line.
"""

from pathlib import Path
import numpy as np
import pandas as pd

GAMES_URL = "https://github.com/nflverse/nfldata/raw/master/data/games.csv"
REL = "https://github.com/nflverse/nflverse-data/releases/download"
TEAM_FIX = {"OAK": "LV", "SD": "LAC", "STL": "LA", "LAR": "LA"}
INJ_WEIGHT = {"Out": 1.0, "Doubtful": 0.85, "Questionable": 0.25}

ALL_FEATURES = ("epa", "qb", "skill", "gameplan", "style", "turnovers",
                "fourth", "halftime", "h2h", "weather", "rest_travel", "pass_rush",
                "kicker", "redzone", "referee", "lines")

# Drive results used by the engine
TD, FG, MISS, PUNT, TO, DOWNS = range(6)
RESULT_MAP = {"Touchdown": TD, "Field goal": FG, "Missed field goal": MISS,
              "Punt": PUNT, "Safety": PUNT, "Turnover": TO,
              "Opp touchdown": TO, "Turnover on downs": DOWNS}
N_BINS = 10  # start field position bins, 10 yards each (yardline_100)


# =============================================================== data loading
def _cached(url, path):
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        import urllib.request
        urllib.request.urlretrieve(url, path)
    return path


def load_all(seasons, data_dir="nfl_data", games_path=None):
    """Download (once) and load games, play-by-play and injury reports."""
    games = pd.read_csv(games_path or _cached(GAMES_URL, f"{data_dir}/games.csv"),
                        low_memory=False)
    for c in ("home_team", "away_team"):
        games[c] = games[c].replace(TEAM_FIX)
    games["gameday"] = pd.to_datetime(games["gameday"])

    pbp, inj = [], []
    for s in seasons:
        pbp.append(pd.read_parquet(_cached(f"{REL}/pbp/play_by_play_{s}.parquet",
                                           f"{data_dir}/pbp_{s}.parquet")))
        try:
            inj.append(pd.read_parquet(_cached(f"{REL}/injuries/injuries_{s}.parquet",
                                               f"{data_dir}/injuries_{s}.parquet")))
        except Exception:
            pass
    pbp = pd.concat(pbp, ignore_index=True)
    inj = pd.concat(inj, ignore_index=True) if inj else pd.DataFrame(
        columns=["season", "week", "team", "gsis_id", "full_name", "position", "report_status"])
    for c in ("posteam", "defteam", "home_team", "away_team"):
        pbp[c] = pbp[c].replace(TEAM_FIX)
    inj["team"] = inj["team"].replace(TEAM_FIX)
    return games, pbp, inj


def _defense_production(pbp):
    """Per player per game: pass-rush events (sacks, half sacks, QB hits) and
    coverage plays (interceptions, passes defended)."""
    parts = []
    for col, kind, val in (("sack_player_id", "press", 1.0), ("half_sack_1_player_id", "press", 0.5),
                           ("half_sack_2_player_id", "press", 0.5), ("qb_hit_1_player_id", "press", 1.0),
                           ("interception_player_id", "cov", 1.0), ("pass_defense_1_player_id", "cov", 1.0),
                           ("pass_defense_2_player_id", "cov", 1.0)):
        if col in pbp:
            x = pbp[pbp[col].notna()][["game_id", col]].rename(columns={col: "gsis"})
            parts.append(x.assign(kind=kind, val=val))
    d = pd.concat(parts)
    return d.pivot_table(index=["game_id", "gsis"], columns="kind", values="val",
                         aggfunc="sum", fill_value=0).reset_index()


def _penalty_games(pbp):
    """Accepted flags per game, split by whether the home or away team was flagged."""
    p = pbp[(pbp.penalty == 1) & pbp.penalty_team.notna()]
    home = p.penalty_team.replace(TEAM_FIX) == p.home_team.replace(TEAM_FIX)
    return pd.DataFrame({"game_id": p.game_id, "home_flag": home.astype(int),
                         "away_flag": (~home).astype(int),
                         "yds": p.penalty_yards.fillna(0)}).groupby("game_id").sum()


def add_referee_history(prep, seasons, data_dir="nfl_data"):
    """Add older seasons of penalty data so crew-chief tendencies rest on more games."""
    cols = ["game_id", "penalty", "penalty_team", "penalty_yards", "home_team"]
    old = [_penalty_games(pd.read_parquet(_cached(f"{REL}/pbp/play_by_play_{s}.parquet",
                                                  f"{data_dir}/pbp_{s}.parquet"), columns=cols))
           for s in seasons]
    prep["pen"] = pd.concat(old + [prep["pen"]])
    prep["pen"] = prep["pen"][~prep["pen"].index.duplicated(keep="last")]
    return prep["pen"]


HIST_WEIGHT = 0.35  # weight of a dropback from before the rating window


def add_qb_history(prep, seasons, data_dir="nfl_data"):
    """Add older seasons of QB play (e.g. range(2019, 2023)) so veteran backups who
    haven't played recently get rated on their own history, not a generic backup prior.
    Only QB ratings use this; team ratings still use the normal window."""
    cols = ["game_id", "season", "posteam", "play_type", "epa", "qb_epa", "qb_dropback",
            "qb_kneel", "qb_spike", "passer_player_id", "passer_player_name",
            "rusher_player_id", "rusher_player_name", "interception", "fumble_lost",
            "fumbled_1_player_id"]
    rows = []
    for s in seasons:
        p = pd.read_parquet(_cached(f"{REL}/pbp/play_by_play_{s}.parquet",
                                    f"{data_dir}/pbp_{s}.parquet"), columns=cols)
        p = p[p.play_type.isin(["pass", "run"]) & p.epa.notna() & (p.qb_dropback == 1)
              & (p.qb_kneel != 1) & (p.qb_spike != 1)].copy()
        p["qb_id"] = p.passer_player_id.fillna(p.rusher_player_id)
        p["name"] = p.passer_player_name.fillna(p.rusher_player_name)
        p["to"] = p.interception.fillna(0) + ((p.fumble_lost == 1)
                                              & (p.fumbled_1_player_id == p.qb_id)).astype(int)
        rows.append(p.dropna(subset=["qb_id"]).groupby(["season", "qb_id", "posteam"]).agg(
            name=("name", "last"), db=("qb_epa", "size"), epa=("qb_epa", "sum"),
            to=("to", "sum")).reset_index())
    prep["qb_hist"] = pd.concat(rows, ignore_index=True)
    prep["qb_hist"]["posteam"] = prep["qb_hist"]["posteam"].replace(TEAM_FIX)
    return prep["qb_hist"]


# ============================================================ pre-aggregation
def prepare(games, pbp):
    """Collapse play-by-play into game-level tables so model fits are fast."""
    gd = games.set_index("game_id")
    pbp = pbp[pbp["game_id"].isin(gd.index)].copy()
    pbp["gameday"] = pbp["game_id"].map(gd["gameday"])
    pbp["season"] = pbp["game_id"].map(gd["season"])

    # --- offensive plays
    off = pbp[pbp.play_type.isin(["pass", "run"]) & pbp.epa.notna() & pbp.posteam.notna()
              & (pbp.qb_kneel != 1) & (pbp.qb_spike != 1)].copy()
    off["is_pass"] = (off["pass"] == 1).astype(int)
    off["pass_epa"] = off.epa * off.is_pass
    off["rush_epa"] = off.epa * (1 - off.is_pass)
    off["to"] = off.interception.fillna(0) + off.fumble_lost.fillna(0)
    off["h1"] = (off.game_half == "Half1").astype(int)
    off["h2"] = (off.game_half == "Half2").astype(int)
    off["h1_epa"] = off.epa * off.h1
    off["h2_epa"] = off.epa * off.h2
    off["dropback"] = off.qb_dropback.fillna(0)
    off["pressure"] = ((off.sack.fillna(0) == 1) | (off.qb_hit.fillna(0) == 1)).astype(int) * off.dropback

    tg = off.groupby(["game_id", "posteam", "defteam"]).agg(
        plays=("epa", "size"), epa=("epa", "sum"),
        pass_n=("is_pass", "sum"), pass_epa=("pass_epa", "sum"),
        rush_epa=("rush_epa", "sum"), to=("to", "sum"),
        h1=("h1", "sum"), h1_epa=("h1_epa", "sum"),
        h2=("h2", "sum"), h2_epa=("h2_epa", "sum"),
        dropbacks=("dropback", "sum"), pressures=("pressure", "sum"),
    ).reset_index()
    tg["rush_n"] = tg.plays - tg.pass_n
    g = gd.loc[tg.game_id]
    tg["gameday"] = g.gameday.values
    tg["season"] = g.season.values
    tg["home"] = (tg.posteam.values == g.home_team.values)
    tg["neutral"] = (g.location.values == "Neutral")
    tg["points"] = np.where(tg.home, g.home_score.values, g.away_score.values)
    tg["coach"] = np.where(tg.home, g.home_coach.values, g.away_coach.values)
    tg["def_coach"] = np.where(tg.home, g.away_coach.values, g.home_coach.values)

    # --- quarterbacks (dropbacks incl. scrambles and sacks)
    db = off[off.qb_dropback == 1].copy()
    db["qb_id"] = db.passer_player_id.fillna(db.rusher_player_id)
    db["qb_name"] = db.passer_player_name.fillna(db.rusher_player_name)
    db["qb_to"] = db.interception.fillna(0) + (
        (db.fumble_lost == 1) & (db.fumbled_1_player_id == db.qb_id)).astype(int)
    qb = db.dropna(subset=["qb_id"]).groupby(["game_id", "posteam", "qb_id"]).agg(
        name=("qb_name", "first"), db=("qb_epa", "size"),
        epa=("qb_epa", "sum"), to=("qb_to", "sum")).reset_index()
    qb["gameday"] = qb.game_id.map(gd.gameday)

    # --- skill players: targets and designed runs
    tgt = off[(off.is_pass == 1) & off.receiver_player_id.notna()]
    rec = tgt.groupby(["game_id", "posteam", "receiver_player_id"]).agg(
        name=("receiver_player_name", "first"), n=("epa", "size"),
        epa=("epa", "sum")).reset_index().rename(columns={"receiver_player_id": "pid"})
    rec["kind"] = "rec"
    runs = off[(off.is_pass == 0) & (off.qb_scramble != 1) & off.rusher_player_id.notna()]
    rus = runs.groupby(["game_id", "posteam", "rusher_player_id"]).agg(
        name=("rusher_player_name", "first"), n=("epa", "size"),
        epa=("epa", "sum")).reset_index().rename(columns={"rusher_player_id": "pid"})
    rus["kind"] = "rush"
    skill = pd.concat([rec, rus], ignore_index=True)
    skill["gameday"] = skill.game_id.map(gd.gameday)

    # --- 4th downs: go-for-it decisions vs league expectation
    fd = pbp[(pbp.down == 4) & pbp.play_type.isin(["pass", "run", "punt", "field_goal"])
             & pbp.posteam.notna()].copy()
    fd["go"] = fd.play_type.isin(["pass", "run"]).astype(int)
    fd["conv"] = fd.fourth_down_converted.fillna(0)
    fd = fd[(fd.wp.between(0.10, 0.90)) & (fd.game_seconds_remaining > 300)]  # skip desperation spots
    fd["zone"] = np.select([(fd.yardline_100 <= 35) & (fd.ydstogo <= 4),
                            (fd.yardline_100 > 35) & (fd.yardline_100 <= 60) & (fd.ydstogo <= 3)],
                           ["fg", "punt"], "other")
    fd["ytg_bin"] = pd.cut(fd.ydstogo, [0, 1, 2, 3, 5, 10, 99]).astype(str)
    fd["yl_bin"] = (fd.yardline_100 // 10).clip(0, 9)
    fd["trail"] = (fd.score_differential < 0).astype(int)
    fd["coach"] = np.where(fd.posteam == fd.home_team, fd.home_coach, fd.away_coach)
    fd = fd[["game_id", "gameday", "posteam", "coach", "go", "conv", "zone",
             "ytg_bin", "yl_bin", "trail"]]

    # --- drives: start spot, result, duration, next start for the opponent
    dp = pbp[pbp.fixed_drive.notna() & pbp.posteam.notna() & pbp.play_type.isin(
        ["pass", "run", "punt", "field_goal", "qb_kneel", "qb_spike", "no_play"])]
    dr = dp.groupby(["game_id", "fixed_drive"]).agg(
        team=("posteam", "first"), start=("yardline_100", "first"),
        result=("fixed_drive_result", "first"), top=("drive_time_of_possession", "first"),
        half=("game_half", "first"), gameday=("gameday", "first"),
        season=("season", "first"), defteam=("defteam", "first"),
        inside20=("drive_inside20", "max")).reset_index().sort_values(["game_id", "fixed_drive"])
    mm = dr.top.fillna("0:00").str.split(":", expand=True).astype(float)
    dr["secs"] = (mm[0] * 60 + mm[1]).clip(5, 600)
    dr["code"] = dr.result.map(RESULT_MAP)
    nxt = dr.groupby("game_id").shift(-1)
    dr["next_start"] = np.where((nxt.team != dr.team) & (nxt.half == dr.half), nxt.start, np.nan)
    dr["bin"] = ((dr.start - 1) // 10).clip(0, N_BINS - 1)
    # pick-six / fumble-return TDs
    ret = pbp[(pbp.interception.eq(1) | pbp.fumble_lost.eq(1)) & pbp.return_touchdown.eq(1)]
    dr["def_td"] = dr.set_index(["game_id", "fixed_drive"]).index.isin(
        ret.set_index(["game_id", "fixed_drive"]).index)

    # --- field goal attempts (blocked = miss) and the kick distance of FG drives
    fgp = pbp[(pbp.play_type == "field_goal") & pbp.kick_distance.notna()].copy()
    fgp["made"] = (fgp.field_goal_result == "made").astype(int)
    kicks = fgp[["game_id", "gameday", "season", "posteam", "kicker_player_id",
                 "kicker_player_name", "kick_distance", "made"]].rename(
        columns={"kicker_player_id": "kid", "kicker_player_name": "kname", "kick_distance": "dist"})
    kd = fgp.groupby(["game_id", "fixed_drive"]).kick_distance.last()
    dr["kick_dist"] = kd.reindex(pd.MultiIndex.from_frame(dr[["game_id", "fixed_drive"]])).to_numpy()

    return {"games": games, "tg": tg, "qb": qb, "skill": skill, "fd": fd, "drives": dr,
            "kicks": kicks, "pen": _penalty_games(pbp), "def_prod": _defense_production(pbp),
            "pressure_cost": off[(off.dropback == 1) & (off.pressure == 1)].epa.mean()
            - off[(off.dropback == 1) & (off.pressure == 0)].epa.mean()}


# ================================================================== helpers
def _recency(dates, seasons, half_life_weeks=16, prior_season_weight=0.6):
    order = pd.Series(dates).rank(method="dense").to_numpy()
    age = order.max() - order
    w = 0.5 ** (age / half_life_weeks)
    return w * np.where(np.asarray(seasons) < np.max(seasons), prior_season_weight, 1.0)


def _ridge_off_def(df, value, weight, ridge):
    """Opponent-adjusted offense/defense ratings for a per-play (or per-game) stat."""
    teams = sorted(set(df.posteam) | set(df.defteam))
    ix = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    X = np.zeros((len(df), 2 * n + 1))
    r = np.arange(len(df))
    X[r, df.posteam.map(ix).to_numpy()] = 1
    X[r, n + df.defteam.map(ix).to_numpy()] = -1
    X[:, -1] = np.where(df.neutral, 0, np.where(df.home, 0.5, -0.5))
    y, W = df[value].to_numpy(float), df[weight].to_numpy(float)
    mu = np.average(y, weights=W)
    P = np.eye(2 * n + 1) * ridge
    P[-1, -1] = 0
    b = np.linalg.solve(X.T @ (W[:, None] * X) + P, X.T @ (W * (y - mu)))
    out = pd.DataFrame({"off": b[:n], "def": b[n:2 * n]}, index=teams)
    return out, mu, b[-1]


def _shrink(total, n, prior, k):
    return (total + prior * k) / (n + k)


# ================================================================ model fit
def fit_model(prep, as_of, seasons_back=2, K=None, w_epa=None):
    """Build every rating the simulator needs from games played before `as_of`."""
    as_of = pd.Timestamp(as_of)
    tg = prep["tg"][prep["tg"].gameday < as_of]
    last = tg.season.max()
    tg = tg[tg.season > last - seasons_back].copy()
    tg["w"] = _recency(tg.gameday, tg.season)
    m = {"as_of": as_of, "K": K if K is not None else 1.2,
         "w_epa": w_epa if w_epa is not None else 0.35}

    # --- team ratings (per-play stats weighted by plays x recency)
    for col, num, den, ridge in (("epa", "epa", "plays", 4000),
                                 ("pass", "pass_epa", "pass_n", 2500),
                                 ("rush", "rush_epa", "rush_n", 2500),
                                 ("to", "to", "plays", 20000)):
        t = tg.assign(val=tg[num] / tg[den].clip(1), ww=tg[den] * tg.w)
        m[col], m[col + "_mu"], m[col + "_hfa"] = _ridge_off_def(t, "val", "ww", ridge)
    t = tg.assign(ww=tg.w)
    m["pts"], m["pts_mu"], m["pts_hfa"] = _ridge_off_def(t, "points", "ww", 2.0)

    wsum = tg.w.sum()
    m["plays_pg"] = (tg.plays * tg.w).sum() / wsum
    m["pass_rate"] = (tg.groupby("posteam").apply(
        lambda d: (d.pass_n * d.w).sum() / (d.plays * d.w).sum(), include_groups=False))
    m["lg_pass_rate"] = (tg.pass_n * tg.w).sum() / (tg.plays * tg.w).sum()
    m["pass_pg"] = (tg.pass_n * tg.w).sum() / wsum
    m["rush_pg"] = m["plays_pg"] - m["pass_pg"]

    _fit_players(m, prep, tg, as_of, last, seasons_back)
    _fit_coaches(m, prep, tg, as_of, last, seasons_back)
    _fit_style(m, tg)
    _fit_engine(m, prep, as_of, last, seasons_back)
    _fit_pressure(m, tg, prep)
    _fit_kickers(m, prep, as_of, last, seasons_back)
    _fit_redzone(m, prep, as_of, last, seasons_back)
    _fit_context(m, prep["games"], as_of)
    _fit_referees(m, prep, as_of)
    _fit_lines(m, prep, tg, as_of)
    m["prep"] = prep
    _calibrate(m)
    m["games"] = prep["games"]
    return m


def _calibrate(m, n=5000):
    """Game flow (possessions, field position, the opponent's pace) pulls scoring toward
    the mean. Fit realized = a + b*own_target + c*opp_target over a grid of matchups,
    then invert it per game so simulated points match projected points."""
    m["cal"] = None
    rng = np.random.default_rng(123)
    mu = m["pts_mu"]
    avg = {"pts": mu, "to_ratio": 1.0, "goe": {"fg": 0, "punt": 0}, "conv": m["lg_conv"],
           "half_pts": 0, "pre_cal": True}
    X, y = [], []
    for th_, ta_ in ((mu - 8, mu - 8), (mu, mu), (mu + 8, mu + 8), (mu - 8, mu + 8), (mu + 8, mu - 8)):
        t1, t2 = dict(avg, pts=th_), dict(avg, pts=ta_)
        tabs = [_team_engine(m, t1), _team_engine(m, t2)]
        sc, hs, pc, tc = (np.zeros((2, n)) for _ in range(4))
        _play_halves(m, rng, tabs, [t1, t2], sc, hs, pc, tc, rng.integers(0, 2, n))
        X += [[1, th_, ta_], [1, ta_, th_]]
        y += [sc[0].mean(), sc[1].mean()]
    m["cal"] = tuple(np.linalg.lstsq(np.array(X), np.array(y), rcond=None)[0])


def _apply_cal(m, th, ta):
    """Solve for engine targets that reproduce the projected points for both teams."""
    if not m.get("cal"):
        return
    a, b, c = m["cal"]
    M = np.array([[b, c], [c, b]])
    xh, xa = np.linalg.solve(M, [th["pts"] - a, ta["pts"] - a])
    for t, x in ((th, xh), (ta, xa)):
        t["pts"], t["half_pts"], t["pre_cal"] = x, t["half_pts"] / b, True


def _fit_players(m, prep, tg, as_of, last, seasons_back):
    gw = tg.set_index(["game_id", "posteam"]).w  # recency weight of each team-game
    m["team_w"] = tg.groupby("posteam").w.sum()

    qb = prep["qb"][prep["qb"].gameday < as_of].copy()
    qb["w"] = gw.reindex(pd.MultiIndex.from_frame(qb[["game_id", "posteam"]])).to_numpy()
    qb = qb.dropna(subset=["w"])
    lg_epa = qb.epa.sum() / qb.db.sum()
    lg_to = qb.to.sum() / qb.db.sum()
    g = qb.assign(we=qb.epa * qb.w, wd=qb.db * qb.w, wt=qb.to * qb.w).groupby("qb_id").agg(
        name=("name", "last"), team=("posteam", "last"), db=("db", "sum"),
        wd=("wd", "sum"), we=("we", "sum"), wt=("wt", "sum"), last=("gameday", "max"))
    repl = lg_epa - 0.10  # backup-caliber prior
    # window-only ratings: the team baseline must match the data team ratings came from
    g0_epa = _shrink(g.we, g.wd, repl, 150)
    g0_to = _shrink(g.wt, g.wd, lg_to * 1.1, 250)
    hist = prep.get("qb_hist")
    if hist is not None and len(hist):
        # older seasons (before the rating window) count at a flat discount
        h = hist[hist.season <= last - seasons_back].copy()
        h["w"] = HIST_WEIGHT
        hg = h.assign(we=h.epa * h.w, wd=h.db * h.w, wt=h.to * h.w).groupby("qb_id").agg(
            name=("name", "last"), team=("posteam", "last"), db=("db", "sum"),
            wd=("wd", "sum"), we=("we", "sum"), wt=("wt", "sum"))
        hg["last"] = pd.NaT
        g = pd.concat([g, hg]).groupby(level=0).agg(
            name=("name", "first"), team=("team", "first"), db=("db", "sum"),
            wd=("wd", "sum"), we=("we", "sum"), wt=("wt", "sum"), last=("last", "max"))
    g["epa_db"] = _shrink(g.we, g.wd, repl, 150)
    g["to_db"] = _shrink(g.wt, g.wd, lg_to * 1.1, 250)
    m["qbs"], m["qb_repl"], m["qb_lg_to"] = g, repl, lg_to

    # team's recency-weighted QB baseline (what its ratings were built with)
    qb["epa_db"] = qb.qb_id.map(g0_epa)
    qb["to_db"] = qb.qb_id.map(g0_to)
    m["qb_base"] = qb.groupby("posteam").apply(lambda d: pd.Series({
        "epa_db": np.average(d.epa_db, weights=d.db * d.w),
        "to_db": np.average(d.to_db, weights=d.db * d.w)}), include_groups=False)
    latest = qb.sort_values("gameday").groupby("posteam").game_id.last()
    m["starters"] = qb[qb.game_id.isin(latest)].sort_values("db").groupby("posteam").qb_id.last()

    # skill players: value per game above replacement, tracked per (player, team)
    sk = prep["skill"][prep["skill"].gameday < as_of].copy()
    sk["w"] = gw.reindex(pd.MultiIndex.from_frame(sk[["game_id", "posteam"]])).to_numpy()
    sk = sk.dropna(subset=["w"])
    out = []
    for kind, d in sk.groupby("kind"):
        p = d.assign(wn=d.n * d.w, we=d.epa * d.w).groupby("pid").agg(
            name=("name", "last"), wn=("wn", "sum"), we=("we", "sum"), n=("n", "sum"),
            gw=("w", "sum"))
        big = p[p.n >= 25]
        repl_k = (big.we.sum() / big.wn.sum()) - (0.12 if kind == "rec" else 0.06)
        p["value_pg"] = (_shrink(p.we, p.wn, repl_k, 60) - repl_k) * p.wn / p.gw.clip(lower=1e-6)
        by_team = d.groupby(["pid", "posteam"]).w.sum().rename("tw").reset_index()
        by_team = by_team.join(p[["name", "value_pg"]], on="pid")
        by_team["share"] = by_team.tw / by_team.posteam.map(m["team_w"])  # share of team's games
        out.append(by_team.assign(kind=kind))
    m["skill"] = pd.concat(out, ignore_index=True) if out else pd.DataFrame(
        columns=["pid", "posteam", "tw", "name", "value_pg", "share", "kind"])


def _fit_coaches(m, prep, tg, as_of, last, seasons_back):
    fd = prep["fd"][(prep["fd"].gameday < as_of)
                    & (prep["fd"].gameday >= pd.Timestamp(f"{last - seasons_back}-03-01"))].copy()
    exp = fd.groupby(["ytg_bin", "yl_bin", "trail"]).go.transform("mean")
    fd["goe"] = fd.go - exp
    z = fd[fd.zone != "other"]
    lg = z.groupby("zone").agg(go=("go", "mean"), n=("go", "size"))
    conv = z[z.go == 1].conv.mean()
    c = z.groupby(["coach", "zone"]).agg(goe=("goe", "sum"), n=("goe", "size")).reset_index()
    c["goe_shrunk"] = c.goe / (c.n + 25)
    m["coach_goe"] = c.set_index(["coach", "zone"]).goe_shrunk
    m["lg_go"] = lg.go
    m["lg_conv"] = conv
    t4 = z[z.go == 1].groupby("posteam").agg(c=("conv", "sum"), n=("conv", "size"))
    m["team_conv"] = _shrink(t4.c, t4.n, conv, 30)
    drives = prep["drives"]
    nd = drives[(drives.gameday < as_of) & drives.gameday.isin(z.gameday)].shape[0]
    m["zone_rate"] = (lg.n / max(nd, 1)).to_dict()  # zone 4th-down decisions per drive
    m["coach_goe_raw"] = c

    # halftime swing: (2H - 1H) EPA/play vs league, offense and defense, by coach
    h = tg[(tg.h1 > 5) & (tg.h2 > 5)].copy()
    h["swing"] = h.h2_epa / h.h2 - h.h1_epa / h.h1
    lg_sw = h.swing.mean()
    o = h.groupby("coach").swing.agg(["sum", "size"])
    d = h.groupby("def_coach").swing.agg(["sum", "size"])
    m["half_off"] = (o["sum"] - lg_sw * o["size"]) / (o["size"] + 30)
    m["half_def"] = (d["sum"] - lg_sw * d["size"]) / (d["size"] + 30)
    m["team_coach"] = tg.sort_values("gameday").groupby("posteam").coach.last()


def _fit_style(m, tg):
    """Defensive profiles + each game's offensive residual for similarity matching."""
    prof = pd.DataFrame({"pass": m["pass"]["def"], "rush": m["rush"]["def"],
                         "to": m["to"]["def"]})
    m["def_profile"] = (prof - prof.mean()) / prof.std()
    e = m["epa"]
    hfa = np.where(tg.neutral, 0, np.where(tg.home, 0.5, -0.5)) * m["epa_hfa"]
    exp = m["epa_mu"] + tg.posteam.map(e["off"]) - tg.defteam.map(e["def"]) + hfa
    s = tg[["posteam", "defteam", "plays", "pass_n", "w", "gameday", "coach"]].copy()
    s["resid"] = tg.epa / tg.plays - exp
    s["prate"] = tg.pass_n / tg.plays
    m["style_games"] = s


def _fit_engine(m, prep, as_of, last, seasons_back):
    """Empirical drive tables: result odds by start spot, durations, next spots."""
    d = prep["drives"]
    d = d[(d.gameday < as_of) & (d.season > last - seasons_back - 1)
          & d.code.notna() & d.start.notna()]
    probs = np.zeros((N_BINS, 6))
    for b in range(N_BINS):
        cnt = np.bincount(d[d.bin == b].code.astype(int), minlength=6).astype(float) + 1
        probs[b] = cnt / cnt.sum()
    m["base_probs"] = probs
    m["bin_weights"] = np.bincount(d.bin.astype(int), minlength=N_BINS) / len(d)
    tos = d[d.code == TO]
    m["def_td_rate"] = tos.def_td.mean()
    # pools to sample from: (result, bin) -> next opponent start, drive seconds
    pools = {}
    for (code, b), g in d.groupby(["code", "bin"]):
        ns = g.next_start.dropna().to_numpy()
        pools[(int(code), int(b))] = (ns if len(ns) else np.array([70.0]), g.secs.to_numpy())
    for code in range(6):  # coarse fallback by result only
        g = d[d.code == code]
        pools[(code, -1)] = (g.next_start.dropna().to_numpy(), g.secs.to_numpy())
    m["pools"] = pools
    m["drives_pg"] = len(d) / d.game_id.nunique() / 2
    ko = d.groupby("game_id").first().start  # opening kickoff starts
    m["kick_starts"] = ko.to_numpy()


# ======================================================== added factors
TZ = {t: 0 for t in ("ATL", "BAL", "BUF", "CAR", "CIN", "CLE", "DET", "IND", "JAX", "MIA",
                     "NE", "NYG", "NYJ", "PHI", "PIT", "TB", "WAS")}
TZ.update({t: -1 for t in ("CHI", "DAL", "GB", "HOU", "KC", "MIN", "NO", "TEN")})
TZ.update({"DEN": -2, "LA": -3, "LAC": -3, "LV": -3, "SEA": -3, "SF": -3})


def team_tz(team, month):
    """Hours behind Eastern. Arizona skips daylight saving: Pacific-like until November."""
    if team == "ARI":
        return -3 if month < 11 else -2
    return TZ.get(team, 0)


def _context_features(df):
    """Rest, travel and weather features for rows of games.csv."""
    month = df.gameday.dt.month
    neutral = (df.location == "Neutral").to_numpy()
    tzh = np.array([team_tz(t, mo) for t, mo in zip(df.home_team, month)])
    tza = np.array([team_tz(t, mo) for t, mo in zip(df.away_team, month)])
    east = np.where(neutral, 0, np.clip(tzh - tza, 0, None))
    west = np.where(neutral, 0, np.clip(tza - tzh, 0, None))
    early = (~neutral) & (tza <= -2) & (tzh == 0) & (df.gametime.fillna("13:00").to_numpy() < "14:00")
    hr, ar = df.home_rest.fillna(7).to_numpy(), df.away_rest.fillna(7).to_numpy()
    indoor = df.roof.isin(["dome", "closed"]).to_numpy()
    wind = np.where(indoor, 0, np.clip(df.wind.to_numpy(float) - 10, 0, None))
    cold = np.where(indoor, 0, np.clip(40 - df.temp.to_numpy(float), 0, None))
    return pd.DataFrame({
        "rest_diff": np.clip(hr - ar, -7, 7), "home_short": (hr <= 5).astype(float),
        "away_short": (ar <= 5).astype(float), "east_hours": east, "west_hours": west,
        "west_early": early.astype(float), "indoor": indoor.astype(float),
        "wind_over_10": wind, "cold_below_40": cold}, index=df.index)


REST_COLS = ["rest_diff", "home_short", "away_short", "east_hours", "west_hours", "west_early"]
WX_COLS = ["indoor", "wind_over_10", "cold_below_40"]


def _ols_shrunk(A, y):
    """OLS, then shrink each coefficient toward 0 by its reliability (1 - 1/t^2).
    Noisy effects fade out instead of pushing the line around."""
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    sigma2 = resid @ resid / max(len(y) - A.shape[1], 1)
    se = np.sqrt(np.diag(sigma2 * np.linalg.pinv(A.T @ A)))
    t2 = (coef / se) ** 2
    return coef * np.clip(1 - 1 / np.maximum(t2, 1e-9), 0, 1), coef / se


def _fit_context(m, games, as_of, years=12):
    """Estimate rest/travel effects on margin and weather/roof effects on total points
    from 12 seasons of results, controlling for each team's season strength."""
    g = games[(games.gameday < as_of) & games.result.notna() & (games.game_type == "REG")
              & (games.season >= as_of.year - years)].copy()
    long = pd.concat([
        pd.DataFrame({"i": g.index, "season": g.season, "team": g.home_team, "pf": g.home_score, "pa": g.away_score}),
        pd.DataFrame({"i": g.index, "season": g.season, "team": g.away_team, "pf": g.away_score, "pa": g.home_score})])
    grp = long.groupby(["season", "team"])
    n = grp.pf.transform("size")
    long["pf_loo"] = (grp.pf.transform("sum") - long.pf) / (n - 1)
    long["pa_loo"] = (grp.pa.transform("sum") - long.pa) / (n - 1)
    h = long.iloc[:len(g)].set_index("i")
    a = long.iloc[len(g):].set_index("i")
    X = _context_features(g)
    strength = (h.pf_loo - h.pa_loo) - (a.pf_loo - a.pa_loo)
    exp_total = (h.pf_loo + a.pa_loo + a.pf_loo + h.pa_loo) / 2
    neutral = (g.location == "Neutral").astype(float)

    A = np.column_stack([np.ones(len(g)), strength, 1 - neutral, X[REST_COLS]])
    ok = np.isfinite(A).all(1)
    coef, t = _ols_shrunk(A[ok], g.result.to_numpy(float)[ok])
    m["rest_coef"] = pd.Series(coef[3:], index=REST_COLS)
    m["rest_t"] = pd.Series(t[3:], index=REST_COLS)
    # Raw time-zone counts say eastbound road teams do *better*, the opposite of every
    # travel study; almost certainly confounded (who travels far, when). Kept as
    # controls, not applied. The early-kickoff body-clock effect is applied.
    m["rest_coef"][["east_hours", "west_hours"]] = 0.0
    m["rest_mean"] = X[REST_COLS][ok].mean()

    wx = X[WX_COLS].copy()
    has = g.roof.isin(["dome", "closed"]) | (g.temp.notna() & g.wind.notna())
    B = np.column_stack([np.ones(len(g)), exp_total, wx])
    ok = has.to_numpy() & np.isfinite(B).all(1)
    coef, t = _ols_shrunk(B[ok], g.total.to_numpy(float)[ok])
    m["wx_coef"] = pd.Series(coef[2:], index=WX_COLS)
    m["wx_t"] = pd.Series(t[2:], index=WX_COLS)
    m["wx_mean"] = wx[ok].mean()
    # climate defaults for games with no forecast yet: stadium x month averages
    out = games[games.temp.notna() & games.wind.notna() & ~games.roof.isin(["dome", "closed"])]
    out = out.assign(month=out.gameday.dt.month)
    m["climate"] = out.groupby(["stadium_id", "month"])[["temp", "wind"]].mean()
    m["climate_month"] = out.groupby("month")[["temp", "wind"]].mean()


REF_ALIASES = {"Bradley Rogers": "Brad Rogers", "Ronald Torbert": "Ron Torbert",
               "Adrian Hall": "Adrian Hill"}


def normalize_referee(m, name):
    """Accept full name, alias, or unique last name, any capitalization."""
    if not name:
        return None
    name = REF_ALIASES.get(str(name).strip(), str(name).strip())
    crews = m["crews"].index
    for c in crews:
        if c.lower() == name.lower():
            return c
    hits = [c for c in crews if c.split()[-1].lower() == name.split()[-1].lower()]
    return hits[0] if len(hits) == 1 else None


def _fit_referees(m, prep, as_of, k=24):
    """Crew-chief tendencies: flags per game and home/away flag split, shrunk toward
    league average (k = games of league-average data added to each crew). Their effect
    on totals and margins is estimated from results, controlling for team strength,
    and shrunk by reliability. There is no public data grading call *accuracy*."""
    g = prep["games"]
    g = g[(g.gameday < as_of) & g.result.notna() & g.referee.notna()
          & g.game_id.isin(prep["pen"].index)].copy()
    g["referee"] = g.referee.replace(REF_ALIASES)
    pen = prep["pen"].reindex(g.game_id)
    g["n_flags"] = (pen.home_flag + pen.away_flag).to_numpy()
    g["fbias"] = (pen.away_flag - pen.home_flag).to_numpy()  # + = away flagged more
    lg_f, lg_b = g["n_flags"].mean(), g["fbias"].mean()
    grp = g.groupby("referee")
    n, sf, sb = grp["n_flags"].transform("size"), grp["n_flags"].transform("sum"), grp["fbias"].transform("sum")
    # leave-one-out crew tendencies so a game's own flags don't explain its own result
    g["crew_flags"] = (sf - g["n_flags"] + lg_f * k) / (n - 1 + k) - lg_f
    g["crew_bias"] = (sb - g["fbias"] + lg_b * k) / (n - 1 + k) - lg_b
    # team strength controls (same season, leave-one-out)
    long = pd.concat([
        pd.DataFrame({"i": g.index, "season": g.season, "team": g.home_team, "pf": g.home_score, "pa": g.away_score}),
        pd.DataFrame({"i": g.index, "season": g.season, "team": g.away_team, "pf": g.away_score, "pa": g.home_score})])
    gr = long.groupby(["season", "team"])
    c = gr.pf.transform("size")
    long["pf"] = (gr.pf.transform("sum") - long.pf) / (c - 1)
    long["pa"] = (gr.pa.transform("sum") - long.pa) / (c - 1)
    h, a = long.iloc[:len(g)].set_index("i"), long.iloc[len(g):].set_index("i")
    strength = (h.pf - h.pa) - (a.pf - a.pa)
    exp_total = (h.pf + a.pa + a.pf + h.pa) / 2
    A = np.column_stack([np.ones(len(g)), exp_total, g.crew_flags])
    ok = np.isfinite(A).all(1)
    ct, tt = _ols_shrunk(A[ok], g.total.to_numpy(float)[ok])
    B = np.column_stack([np.ones(len(g)), strength, g.crew_bias])
    ok = np.isfinite(B).all(1)
    cm, tm = _ols_shrunk(B[ok], g.result.to_numpy(float)[ok])
    m["ref_total_coef"], m["ref_total_t"] = ct[2], tt[2]
    m["ref_margin_coef"], m["ref_margin_t"] = cm[2], tm[2]
    crews = grp.agg(games=("n_flags", "size"), nf=("n_flags", "sum"), nb=("fbias", "sum"))
    crews["flags_pg"] = (crews.nf + lg_f * k) / (crews.games + k)
    crews["away_minus_home_flags"] = (crews.nb + lg_b * k) / (crews.games + k)
    m["crews"], m["lg_flags"], m["lg_bias"] = crews[["games", "flags_pg", "away_minus_home_flags"]], lg_f, lg_b


def referee_effect(m, referee):
    """(margin shift for home, total shift) from this crew chief's tendencies."""
    referee = normalize_referee(m, referee)
    if not referee:
        return 0.0, 0.0, None
    c = m["crews"].loc[referee]
    margin = m["ref_margin_coef"] * (c.away_minus_home_flags - m["lg_bias"])
    total = m["ref_total_coef"] * (c.flags_pg - m["lg_flags"])
    return float(margin), float(total), f"{referee}: {c.flags_pg:.1f} flags/game (league {m['lg_flags']:.1f}), " \
        f"away-minus-home {c.away_minus_home_flags:+.2f} ({int(c.games)} games)"


def active_referees(m, since_season=2025):
    """Crew chiefs who worked games since `since_season`: the valid REFEREE values."""
    g = m["games"]
    g = g[g.referee.notna() & (g.season >= since_season)].assign(ref=lambda d: d.referee.replace(REF_ALIASES))
    act = sorted(g.ref.unique())
    return m["crews"].loc[[r for r in act if r in m["crews"].index]].round(2)


def referee_table(m, min_games=15):
    c = m["crews"][m["crews"].games >= min_games].copy()
    c["total_pts_effect"] = m["ref_total_coef"] * (c.flags_pg - m["lg_flags"])
    c["home_margin_effect"] = m["ref_margin_coef"] * (c.away_minus_home_flags - m["lg_bias"])
    return c.round(2).sort_values("flags_pg", ascending=False)


def game_context(m, home, away, neutral=False, weather=None):
    """Margin and total adjustments for this specific game (home perspective)."""
    g = m["games"]
    row = g[(g.home_team == home) & (g.away_team == away)
            & (g.gameday >= m["as_of"] - pd.Timedelta(days=3))].sort_values("gameday").head(1)
    if row.empty:  # hypothetical matchup: normal week, home stadium
        last = g[g.home_team == home].sort_values("gameday").tail(1)
        row = pd.DataFrame({"gameday": [m["as_of"]], "home_team": [home], "away_team": [away],
                            "location": ["Neutral" if neutral else "Home"], "gametime": ["13:00"],
                            "home_rest": [7], "away_rest": [7],
                            "roof": last.roof.to_numpy()[:1] if len(last) else ["outdoors"],
                            "temp": [np.nan], "wind": [np.nan],
                            "stadium_id": last.stadium_id.to_numpy()[:1] if len(last) else [None]})
    row = row.copy()
    note = {}
    indoor = row.roof.isin(["dome", "closed"]).iloc[0]
    if weather:
        row["temp"], row["wind"] = weather.get("temp", 60), weather.get("wind", 5)
        note["weather"] = f"{row.temp.iloc[0]:.0f}F, wind {row.wind.iloc[0]:.0f} mph (your input)"
    elif not indoor and (pd.isna(row.temp.iloc[0]) or pd.isna(row.wind.iloc[0])):
        mo = pd.Timestamp(row.gameday.iloc[0]).month
        key = (row.stadium_id.iloc[0], mo)
        c = m["climate"].loc[key] if key in m["climate"].index else m["climate_month"].loc[mo]
        row["temp"], row["wind"] = c.temp, c.wind
        note["weather"] = f"~{c.temp:.0f}F, wind ~{c.wind:.0f} mph (stadium average, no forecast)"
    elif indoor:
        note["weather"] = f"indoors ({row.roof.iloc[0]})"
    else:
        note["weather"] = f"{row.temp.iloc[0]:.0f}F, wind {row.wind.iloc[0]:.0f} mph"
    X = _context_features(row).iloc[0]
    margin = float(((X[REST_COLS] - m["rest_mean"]) * m["rest_coef"]).sum())
    total = float(((X[WX_COLS] - m["wx_mean"]) * m["wx_coef"]).sum())
    note["rest"] = f"{home} {row.home_rest.iloc[0]:.0f} days, {away} {row.away_rest.iloc[0]:.0f} days"
    if "referee" in row and pd.notna(row.referee.iloc[0]):
        note["referee_name"] = row.referee.iloc[0]
    if X.east_hours or X.west_hours:
        note["travel"] = f"{away} crosses {int(X.east_hours or X.west_hours)} time zone(s)" + (
            ", early body-clock kickoff" if X.west_early else "")
    return margin, total, note


def _fit_pressure(m, tg, prep):
    t = tg[tg.dropbacks > 0].assign(val=lambda d: d.pressures / d.dropbacks,
                                    ww=lambda d: d.dropbacks * d.w)
    m["pr"], m["pr_mu"], _ = _ridge_off_def(t, "val", "ww", 300)
    m["pressure_cost"] = prep["pressure_cost"]


def pressure_matchup(m, off, dfn):
    """Pressure rate for this OL vs this pass rush. Additive ratings assume effects
    simply add; log5 lets a bad line vs a great rush compound. Returns both."""
    mu = m["pr_mu"]
    po = np.clip(mu + m["pr"]["off"].get(off, 0), 0.05, 0.6)   # line allows
    pd_ = np.clip(mu - m["pr"]["def"].get(dfn, 0), 0.05, 0.6)  # rush generates
    add = po + pd_ - mu
    num = po * pd_ / mu
    log5 = num / (num + (1 - po) * (1 - pd_) / (1 - mu))
    return add, log5


def _logit_fit(x, y, iters=25):
    X = np.column_stack([np.ones_like(x), x])
    b = np.zeros(2)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ b))
        W = p * (1 - p)
        b += np.linalg.solve(X.T @ (W[:, None] * X) + 1e-6 * np.eye(2), X.T @ (y - p))
    return b


def _fit_kickers(m, prep, as_of, last, seasons_back):
    k = prep["kicks"][(prep["kicks"].gameday < as_of)]
    k = k[k.season > last - seasons_back - 1]  # three seasons of kicks
    b = _logit_fit(k.dist.to_numpy(float), k.made.to_numpy(float))
    p = 1 / (1 + np.exp(-(b[0] + b[1] * k.dist)))
    k = k.assign(r=k.made - p, v=p * (1 - p))
    kk = k.groupby("kid").agg(name=("kname", "last"), team=("posteam", "last"),
                              att=("made", "size"), made=("made", "sum"), r=("r", "sum"), v=("v", "sum"))
    kk["offset"] = kk.r / (kk.v + 15)  # shrunk logit offset vs league at same distance
    m["fg_logit"], m["kickers"] = b, kk
    m["team_kicker"] = k.sort_values("gameday").groupby("posteam").kid.last()
    d = prep["drives"]
    d = d[(d.gameday < as_of) & d.code.isin([FG, MISS]) & d.kick_dist.notna()]
    m["kick_pools"] = {int(b_): g.kick_dist.to_numpy() for b_, g in d.groupby("bin")}
    m["kick_pool_all"] = d.kick_dist.to_numpy()


def _fg_prob(m, dist, offset):
    b = m["fg_logit"]
    return 1 / (1 + np.exp(-(b[0] + b[1] * dist + offset)))


def _sample_kick_dist(m, rng, bins):
    out = np.empty(len(bins))
    for b in np.unique(bins):
        i = np.where(bins == b)[0]
        pool = m["kick_pools"].get(int(b), m["kick_pool_all"])
        out[i] = pool[rng.integers(0, len(pool), len(i))]
    return out


def _fit_redzone(m, prep, as_of, last, seasons_back):
    d = prep["drives"]
    d = d[(d.gameday < as_of) & (d.season > last - seasons_back) & (d.inside20 == 1)].copy()
    d["td"] = (d.code == TD).astype(float)
    d["w"] = _recency(d.gameday, d.season)
    lg = np.average(d.td, weights=d.w)
    k = 120  # heavy shrinkage: red-zone TD rate barely repeats
    o = d.groupby("team").apply(lambda x: ((x.td - lg) * x.w).sum() / (x.w.sum() + k), include_groups=False)
    df = d.groupby("defteam").apply(lambda x: ((x.td - lg) * x.w).sum() / (x.w.sum() + k), include_groups=False)
    m["rz_off"], m["rz_def"], m["rz_lg"] = o, df, lg
    m["rz_trips_pg"] = len(d) / max(prep["drives"][(prep["drives"].gameday < as_of)
                                                   & (prep["drives"].season > last - seasons_back)]
                                    .groupby(["game_id", "team"]).ngroups, 1)


# ======================================================= OL and defense starters
POS_GROUP = {"T": "OL", "G": "OL", "C": "OL", "OL": "OL", "OT": "OL", "OG": "OL",
             "DE": "DL", "DT": "DL", "NT": "DL", "DL": "DL",
             "LB": "LB", "ILB": "LB", "MLB": "LB", "OLB": "LB",
             "CB": "CB", "DB": "CB", "S": "S", "FS": "S", "SS": "S", "SAF": "S"}
DEF_GROUPS = ("DL", "LB", "CB", "S")
LINE_GROUPS = ("OL",) + DEF_GROUPS


def _regulars(pivot, j, lookback=8):
    """Players who started in the team's recent games: played 2+ of the last
    `lookback` games at 60%+ of snaps when they played."""
    prior = pivot.iloc[:, max(0, j - lookback):j]
    if prior.shape[1] == 0:
        return pd.Series(dtype=float)
    played = (prior > 0.05).sum(1)
    avg = prior.where(prior > 0.05).mean(1)
    return avg[(played >= 2) & (avg >= 0.6)]


def add_line_data(prep, seasons, rosters, injuries=None, data_dir="nfl_data"):
    """Load snap counts and work out, for every team-game, which regular OL and
    defensive starters did not play. `rosters` = weekly rosters (IDs, reserve lists),
    `injuries` = injury reports: used to tag which absences were announced beforehand."""
    snaps = pd.concat([pd.read_parquet(_cached(f"{REL}/snap_counts/snap_counts_{y}.parquet",
                                               f"{data_dir}/snap_counts_{y}.parquet")) for y in seasons])
    snaps["team"] = snaps.team.replace(TEAM_FIX)
    snaps["group"] = snaps.position.map(POS_GROUP)
    snaps = snaps[snaps.group.notna()].copy()
    snaps["pct"] = np.where(snaps.group == "OL", snaps.offense_pct, snaps.defense_pct)
    snaps["dsnaps"] = snaps.defense_snaps
    gd = prep["games"].set_index("game_id").gameday
    snaps = snaps[snaps.game_id.isin(gd.index)]
    snaps["gameday"] = snaps.game_id.map(gd)
    info = snaps.sort_values("gameday").drop_duplicates("pfr_player_id", keep="last").set_index(
        "pfr_player_id")[["player", "group"]]

    rows = []
    for team, ts in snaps.groupby("team"):
        order = ts.drop_duplicates("game_id").sort_values("gameday").game_id.tolist()
        piv = ts.pivot_table(index="pfr_player_id", columns="game_id", values="pct",
                             aggfunc="max", fill_value=0)[order]
        for j, gid in enumerate(order):
            reg = _regulars(piv, j)
            if reg.empty:
                continue
            miss = reg[piv.loc[reg.index, gid] <= 0.05]
            rows += [(gid, team, pid, a) for pid, a in miss.items()]
    lm = pd.DataFrame(rows, columns=["game_id", "team", "pfr", "avg_pct"])
    lm["group"] = lm.pfr.map(info.group)
    r = rosters.dropna(subset=["gsis_id", "pfr_id"]).drop_duplicates("pfr_id", keep="last")
    # was the absence announced before kickoff? (what a prediction can know)
    gw = prep["games"].set_index("game_id")[["season", "week"]]
    lm = lm.join(gw, on="game_id")
    lm["gsis"] = lm.pfr.map(r.set_index("pfr_id").gsis_id)
    st = rosters[["season", "week", "gsis_id", "team", "status"]].drop_duplicates(
        ["season", "week", "gsis_id"], keep="last").set_index(["season", "week", "gsis_id"])
    key = pd.MultiIndex.from_frame(lm[["season", "week", "gsis"]].rename(columns={"gsis": "gsis_id"}))
    rs = st.reindex(key)
    moved = rs.team.notna().to_numpy() & (rs.team.to_numpy() != lm.team.to_numpy())
    ann = np.where(rs.status.isin(["RES", "INA"]).to_numpy() | moved, 1.0, 0.0)
    if injuries is not None and len(injuries):
        ij = injuries.drop_duplicates(["season", "week", "gsis_id"], keep="last").set_index(
            ["season", "week", "gsis_id"]).report_status.map(INJ_WEIGHT)
        ann = np.maximum(ann, ij.reindex(key).fillna(0).to_numpy())
    lm["announced"] = ann
    prep["line_snaps"] = snaps[["game_id", "gameday", "season", "team", "pfr_player_id", "player",
                                "group", "pct", "dsnaps"]]
    prep["line_missing"] = lm
    prep["line_info"] = info
    prep["p2g"] = r.set_index("pfr_id").gsis_id
    prep["g2p"] = r.drop_duplicates("gsis_id", keep="last").set_index("gsis_id").pfr_id
    return lm


def _line_quality(prep, as_of):
    """Quality multiplier (0.5-2.0) per defender: pass rushers by pressure events per
    snap, DBs by coverage plays per snap, both shrunk toward their group average."""
    sn = prep["line_snaps"]
    sn = sn[(sn.gameday < as_of) & (sn.group.isin(DEF_GROUPS))]
    dp = prep["def_prod"]
    dp = dp[dp.game_id.isin(sn.game_id.unique())].copy()
    dp["pfr"] = dp.gsis.map(prep["g2p"])
    prod = dp.dropna(subset=["pfr"]).groupby("pfr")[["press", "cov"]].sum()
    snp = sn.groupby("pfr_player_id").agg(snaps=("dsnaps", "sum"), group=("group", "last"))
    q = snp.join(prod).fillna(0)
    out = pd.Series(1.0, index=q.index)
    for grp, col in (("DL", "press"), ("LB", "press"), ("CB", "cov"), ("S", "cov")):
        g = q[q.group == grp]
        if g.empty:
            continue
        mean = g[col].sum() / max(g.snaps.sum(), 1)
        rate = (g[col] + mean * 300) / (g.snaps + 300)  # ~300 snaps of league-average prior
        qq = (rate / mean).clip(0.5, 2.0)
        out[g.index] = qq.clip(lower=1.0) if grp == "LB" else qq  # off-ball LBs aren't rushers
    return out


def _fit_lines(m, prep, tg, as_of):
    """What a missing starter costs, in EPA/play, by position group: regress each
    game's offensive EPA residual on the OL starters the offense was missing and the
    defensive starters the defense was missing."""
    m["lines_ok"] = "line_missing" in prep
    if not m["lines_ok"]:
        return
    q = _line_quality(prep, as_of)
    m["line_q"] = q
    lm = prep["line_missing"].copy()
    lm["v"] = lm.avg_pct * lm.pfr.map(q).fillna(1.0)
    val = lm.pivot_table(index=["game_id", "team"], columns="group", values="v", aggfunc="sum",
                         fill_value=0).reindex(columns=list(LINE_GROUPS), fill_value=0)
    lm["va"] = lm.v * (lm["announced"] if "announced" in lm else 1.0)
    val_a = lm.pivot_table(index=["game_id", "team"], columns="group", values="va", aggfunc="sum",
                           fill_value=0).reindex(columns=list(LINE_GROUPS), fill_value=0)
    d = tg.copy()
    ol = val["OL"].reindex(pd.MultiIndex.from_frame(d[["game_id", "posteam"]])).fillna(0).to_numpy()
    dv = val[list(DEF_GROUPS)].reindex(pd.MultiIndex.from_frame(d[["game_id", "defteam"]])).fillna(0).to_numpy()
    has = d.game_id.isin(prep["line_snaps"].game_id.unique()).to_numpy()
    e = m["epa"]
    hfa = np.where(d.neutral, 0, np.where(d.home, 0.5, -0.5)) * m["epa_hfa"]
    resid = (d.epa / d.plays - (m["epa_mu"] + d.posteam.map(e["off"]) - d.defteam.map(e["def"]) + hfa)).to_numpy()
    A = np.column_stack([np.ones(len(d)), ol, dv])[has]
    coef, t = _ols_shrunk(A, resid[has])
    m["line_coef"] = pd.Series(coef[1:], index=list(LINE_GROUPS))
    m["line_t"] = pd.Series(t[1:], index=list(LINE_GROUPS))
    # each team's typical *announced* missing load in the rating window: the same thing a
    # prediction can see, so "now vs usual" compares like with like
    ol_a = val_a["OL"].reindex(pd.MultiIndex.from_frame(d[["game_id", "posteam"]])).fillna(0).to_numpy()
    dv_a = val_a[list(DEF_GROUPS)].reindex(pd.MultiIndex.from_frame(d[["game_id", "defteam"]])).fillna(0).to_numpy()
    w = d.w.to_numpy()[has]
    avg = {}
    for i, g in enumerate(LINE_GROUPS):
        col = ol_a if g == "OL" else dv_a[:, i - 1]
        side = d.posteam if g == "OL" else d.defteam
        df = pd.DataFrame({"team": side.to_numpy()[has], "v": col[has], "w": w})
        avg[g] = df.groupby("team").apply(lambda x: np.average(x.v, weights=x.w), include_groups=False)
    m["line_avg_miss"] = pd.DataFrame(avg).fillna(0)
    m["line_as_of"] = as_of


def line_status(m, prep, team, ctx=None, out_players=None, lookback=8):
    """Current regular OL/defensive starters for a team and who's missing now."""
    sn = prep["line_snaps"]
    ts = sn[(sn.team == team) & (sn.gameday < m["as_of"])]
    # starters = this season's regulars once 2+ games are played. Before that, last
    # season's regulars, but offseason departures don't count as "missing" (their
    # replacements are unknown until they play).
    cur = ts[ts.season == ts.season.max()] if len(ts) else ts
    in_season = cur.game_id.nunique() >= 2
    if in_season:
        ts = cur
    order = ts.drop_duplicates("game_id").sort_values("gameday").game_id.tolist()[-lookback:]
    if not order:
        return pd.DataFrame()
    piv = ts[ts.game_id.isin(order)].pivot_table(index="pfr_player_id", columns="game_id",
                                                 values="pct", aggfunc="max", fill_value=0)[order]
    reg = _regulars(piv, len(order), lookback)
    miss = (ctx or {}).get("miss", {})
    team_now = (ctx or {}).get("team_now")
    outs = {n.lower() for n in (out_players or [])}
    rows = []
    for pid, a in reg.items():
        gsis = prep["p2g"].get(pid)
        name = prep["line_info"].player.get(pid, pid)
        grp = prep["line_info"].group.get(pid)
        w, why = 0.0, ""
        if gsis is not None and miss.get(gsis, 0) > 0:
            w, why = miss[gsis], (ctx or {}).get("why", {}).get(gsis, "injury report")
        if in_season and team_now is not None and gsis is not None and team_now.get(gsis, None) not in (team,):
            w, why = 1.0, (f"now {team_now.get(gsis)}" if isinstance(team_now.get(gsis), str) else "not on roster")
        if name.lower() in outs or _short(name) in outs:
            w, why = 1.0, "your OUT list"
        rows.append({"player": name, "group": grp, "snap_share": round(a, 2),
                     "quality": round(float(m.get("line_q", pd.Series(dtype=float)).get(pid, 1.0)), 2),
                     "missing_weight": w, "why": why})
    return pd.DataFrame(rows).sort_values(["group", "snap_share"], ascending=[True, False])


def _short(name):
    parts = str(name).split()
    return (f"{parts[0][0]}.{' '.join(parts[1:])}" if len(parts) > 1 else str(name)).lower()


def _line_delta(m, prep, team, ctx, out_players):
    """Starter-equivalents missing now minus the team's typical missing load, centered on
    the league this week (so reporting timing can't push every team the same way)."""
    key = (id(ctx), tuple(out_players or []))
    cache = m.setdefault("_line_cache", {})
    if key not in cache:
        rows, names = {}, {}
        for t in m["line_avg_miss"].index:
            rows[t], names[t] = _line_delta_raw(m, prep, t, ctx, out_players)
        df = pd.DataFrame(rows).T
        cache[key] = (df - df.mean(), names)
    df, names = cache[key]
    if team not in df.index:
        return {g: 0.0 for g in LINE_GROUPS}, {"OL": [], "DEF": []}
    return df.loc[team].to_dict(), names[team]


def _line_delta_raw(m, prep, team, ctx, out_players):
    st = line_status(m, prep, team, ctx, out_players)
    base = m["line_avg_miss"].loc[team] if team in m["line_avg_miss"].index else pd.Series(0, index=LINE_GROUPS)
    now = {g: 0.0 for g in LINE_GROUPS}
    names = {"OL": [], "DEF": []}
    if len(st):
        for _, r in st[st.missing_weight > 0].iterrows():
            now[r.group] += r.snap_share * r.quality * r.missing_weight
            if r.missing_weight >= 0.5:
                names["OL" if r.group == "OL" else "DEF"].append(f"{r.player} ({r.group}, {r.why})")
    return {g: now[g] - base.get(g, 0) for g in LINE_GROUPS}, names


# ============================================================ game features
# ======================================================== rosters / availability
def load_rosters(season, data_dir="nfl_data"):
    """Weekly rosters (status incl. IR / PUP / NFI) and the latest depth chart."""
    rw = pd.read_parquet(_cached(f"{REL}/weekly_rosters/roster_weekly_{season}.parquet",
                                 f"{data_dir}/roster_weekly_{season}.parquet"))
    rw["team"] = rw["team"].replace(TEAM_FIX)
    try:
        dc = pd.read_parquet(_cached(f"{REL}/depth_charts/depth_charts_{season}.parquet",
                                     f"{data_dir}/depth_charts_{season}.parquet"))
        dc["team"] = dc["team"].replace(TEAM_FIX)
    except Exception:
        dc = None
    return rw, dc


RESERVE_LABELS = {"R01": "IR", "R48": "IR (designated to return)", "R04": "PUP",
                  "R05": "NFI", "R40": "NFI", "R02": "retired", "R27": "suspended"}


def build_context(season, week, rosters, injuries, depth=None, carry_forward=False):
    """Who is on each team and who is unavailable for `week`.

    Reserve lists (IR, PUP, NFI, suspended) = out. Injury report statuses are used
    as-is for that week; if the week's report isn't out yet (carry_forward=True),
    the most recent report is used at reduced weight."""
    rw = rosters[(rosters.season == season) & (rosters.week <= week)]
    snap = rw[rw.week == rw.week.max()]
    on_team = snap[snap.status.isin(["ACT", "RES", "INA", "DEV"])]
    team_now = on_team.drop_duplicates("gsis_id", keep="last").set_index("gsis_id").team
    miss, why = {}, {}
    for _, r in snap[snap.status.isin(["RES", "INA"])].iterrows():
        miss[r.gsis_id] = 1.0
        why[r.gsis_id] = RESERVE_LABELS.get(r.status_description_abbr, "reserve/inactive")
    inj = injuries[(injuries.season == season) & (injuries.week <= week)]
    if len(inj):
        wk = inj.week.max()
        rep = inj[inj.week == wk]
        scale = {"Out": 0.7, "Doubtful": 0.5, "Questionable": 0.15} if (carry_forward or wk < week) \
            else INJ_WEIGHT
        for _, r in rep.iterrows():
            wt = scale.get(r.report_status, 0)
            if wt > miss.get(r.gsis_id, 0):
                miss[r.gsis_id] = wt
                why[r.gsis_id] = f"{r.report_status} (wk {wk} report)"
    qb1, k1 = {}, {}
    if depth is not None and len(depth):
        d = depth[depth.dt == depth.groupby("team").dt.transform("max")]
        k1 = {}
        kd = d[d.pos_abb == "PK"].sort_values("pos_rank")
        for team, x in kd.groupby("team"):
            k1[team] = x.gsis_id.tolist()
        d = d[d.pos_abb == "QB"].sort_values("pos_rank")
        for team, x in d.groupby("team"):
            qb1[team] = x.gsis_id.tolist()
    names = snap.drop_duplicates("gsis_id", keep="last").set_index("gsis_id").full_name
    return {"team_now": team_now, "miss": miss, "why": why, "qb_depth": qb1, "kickers": k1,
            "names": names, "label": f"{season} week {week}"}


def _availability(m, team, ctx, out_players):
    miss = dict(ctx["miss"]) if ctx else {}
    for name in out_players or []:
        keys = {name.lower(), _short(name)}
        hit = m["qbs"][m["qbs"].name.str.lower().isin(keys)].index.tolist()
        hit += m["skill"][m["skill"].name.str.lower().isin(keys)].pid.tolist()
        for pid in hit:
            miss[pid] = 1.0
    return miss


def pick_qb(m, team, ctx=None, miss=None, override=None):
    q = m["qbs"]
    miss = miss or {}
    blank = pd.Series({"name": "replacement QB", "epa_db": m["qb_repl"], "to_db": m["qb_lg_to"] * 1.2})
    if override:
        ov = str(override).strip()
        if override in q.index:
            return override, q.loc[override]
        # accept "C.Keenum" or "Case Keenum"
        parts = ov.split()
        short = f"{parts[0][0]}.{' '.join(parts[1:])}" if len(parts) > 1 else ov
        hit = q[(q.name.str.lower() == ov.lower()) | (q.name.str.lower() == short.lower())]
        if len(hit):
            return hit.index[0], hit.iloc[0]
        # no dropbacks in the rating window: still start him, rated as a backup-level QB
        return ov, pd.Series({"name": f"{ov} (no recent data, rated as backup)",
                              "epa_db": m["qb_repl"], "to_db": m["qb_lg_to"] * 1.2})
    order = list((ctx or {}).get("qb_depth", {}).get(team, []))
    st = m["starters"].get(team)
    order += [st] if st is not None else []
    team_now = (ctx or {}).get("team_now")
    for pid in order:
        if miss.get(pid, 0) >= 0.5:
            continue
        if team_now is not None and team_now.get(pid, team) != team:
            continue
        if pid in q.index:
            return pid, q.loc[pid]
        nm = (ctx or {}).get("names", pd.Series(dtype=str)).get(pid, pid)
        return pid, pd.Series({"name": f"{nm} (no recent data, rated as backup)",
                               "epa_db": m["qb_repl"], "to_db": m["qb_lg_to"] * 1.2})
    return None, blank


def _skill_changes(m, team, ctx, miss):
    """Value lost/gained per game from injuries, reserve lists and roster moves."""
    sk = m["skill"]
    if not len(sk):
        return 0.0, 0.0, []
    team_now = (ctx or {}).get("team_now")
    why = (ctx or {}).get("why", {})
    pass_d = rush_d = 0.0
    notes = []
    # players whose production is baked into this team's ratings
    mine = sk[(sk.posteam == team) & (sk.value_pg > 0)]
    for _, r in mine.iterrows():
        now = team_now.get(r.pid) if team_now is not None else team
        gone = team_now is not None and now != team
        wt = 1.0 if gone else miss.get(r.pid, 0)
        if wt <= 0:
            continue
        loss = 0.6 * wt * r.value_pg * min(r.share, 1.0)
        if r.kind == "rec":
            pass_d -= loss
        else:
            rush_d -= loss
        tag = f"now {now}" if gone and isinstance(now, str) else ("left team" if gone else why.get(r.pid, f"{wt:.0%} out"))
        if loss > 0.3:
            notes.append(f"-{r['name']} ({tag})")
    # arrivals: production earned elsewhere that the ratings don't know about
    if team_now is not None:
        new = team_now[team_now == team].index
        arr = sk[sk.pid.isin(new) & (sk.posteam != team) & (sk.value_pg > 0)]
        here = sk[(sk.posteam == team)].groupby(["pid", "kind"]).share.sum()
        for (pid, kind), d in arr.groupby(["pid", "kind"]):
            r = d.iloc[0]
            avail = 1 - miss.get(pid, 0)
            gain = 0.6 * avail * r.value_pg * max(0.0, 1 - here.get((pid, kind), 0))
            if r.kind == "rec":
                pass_d += gain
            else:
                rush_d += gain
            if gain > 0.3:
                notes.append(f"+{r['name']} (from {', '.join(d.posteam.unique())})")
    return pass_d / m["pass_pg"], rush_d / m["rush_pg"], notes


def team_inputs(m, off, dfn, home_flag, feats, qb_override=None, out_players=None, ctx=None):
    """Expected EPA/play, points, turnover rate and 4th-down style for one offense."""
    f = set(feats)
    notes = {}
    miss = _availability(m, off, ctx, out_players)

    # ---- game plan: pass rate vs defenses like this one
    prate = m["pass_rate"].get(off, m["lg_pass_rate"])
    style_resid = 0.0
    if ("gameplan" in f or "style" in f) and dfn in m["def_profile"].index:
        sg = m["style_games"][m["style_games"].posteam == off]
        prof = m["def_profile"]
        if len(sg):
            dist = ((prof.loc[sg.defteam].to_numpy() - prof.loc[dfn].to_numpy()) ** 2).sum(1)
            sim = np.exp(-dist / 2) * sg.w.to_numpy() * np.where(sg.defteam == dfn, 2.0, 1.0)
            # games called by the current head coach count fully, older regimes at 40%
            sim = sim * np.where(sg.coach.to_numpy() == m["team_coach"].get(off), 1.0, 0.4)
            eff_n = sim.sum() ** 2 / max((sim ** 2).sum(), 1e-9)
            if "gameplan" in f:
                pr = np.average(sg.prate, weights=sim * sg.plays)
                prate = prate + (pr - prate) * eff_n / (eff_n + 6)
                notes["pass_rate_vs_similar"] = round(prate, 3)
            if "style" in f:
                r = np.average(sg.resid, weights=sim * sg.plays)
                style_resid = r * eff_n / (eff_n + 12)

    # ---- QB
    qb_delta_epa, qb_delta_to = 0.0, 0.0
    qb_id, qb = pick_qb(m, off, ctx, miss, qb_override)
    notes["qb"] = qb["name"]
    base = m["qb_base"].loc[off] if off in m["qb_base"].index else None
    if "qb" in f and base is not None:
        qb_delta_epa = (qb["epa_db"] - base.epa_db)  # per dropback
        qb_delta_to = (qb["to_db"] - base.to_db)
        notes["qb_vs_team_norm_epa"] = round(qb_delta_epa, 3)

    # ---- skill players: injuries, reserve lists (IR/PUP/NFI), roster moves
    skill_pass = skill_rush = 0.0
    if "skill" in f:
        skill_pass, skill_rush, moves = _skill_changes(m, off, ctx, miss)
        if moves:
            notes["roster_changes"] = moves

    hfa = 0 if home_flag is None else (0.5 if home_flag else -0.5)
    e = m["epa"]
    # ---- EPA/play: blend of overall and pass/run split under this game plan
    if "gameplan" in f:
        pe = m["pass_mu"] + m["pass"]["off"].get(off, 0) - m["pass"]["def"].get(dfn, 0)
        re = m["rush_mu"] + m["rush"]["off"].get(off, 0) - m["rush"]["def"].get(dfn, 0)
        split = prate * pe + (1 - prate) * re
        overall = m["epa_mu"] + e["off"].get(off, 0) - e["def"].get(dfn, 0)
        epa = 0.5 * split + 0.5 * overall
    else:
        epa = m["epa_mu"] + e["off"].get(off, 0) - e["def"].get(dfn, 0)
    epa += hfa * m["epa_hfa"]
    epa_base = epa  # team ratings only; everything added below is a matchup/player adjustment
    epa += style_resid
    epa += prate * (qb_delta_epa + skill_pass) + (1 - prate) * skill_rush

    # ---- offensive line and defensive starters missing (vs the team's usual load)
    if "lines" in f and m.get("lines_ok"):
        prep = m["prep"]
        d_off, n_off = _line_delta(m, prep, off, ctx, out_players)
        d_def, n_def = _line_delta(m, prep, dfn, ctx, out_players)
        c = m["line_coef"]
        adj = c["OL"] * d_off["OL"] + sum(c[g] * d_def[g] for g in DEF_GROUPS)
        epa += adj
        notes["line_epa_adj"] = round(adj, 3)
        if n_off["OL"]:
            notes["ol_out"] = n_off["OL"]
        if n_def["DEF"]:
            notes["opp_def_out"] = n_def["DEF"]

    # ---- pass rush vs pass protection: compounding beyond what additive ratings assume
    if "pass_rush" in f:
        add, log5 = pressure_matchup(m, off, dfn)
        epa += prate * (log5 - add) * m["pressure_cost"]
        notes["pressure_rate"] = f"{log5:.1%} of dropbacks (league {m['pr_mu']:.1%})"

    # ---- points: EPA-based projection blended with the points model
    pts_model = m["pts_mu"] + m["pts"]["off"].get(off, 0) - m["pts"]["def"].get(dfn, 0) \
        + hfa * m["pts_hfa"]
    pts_epa = m["pts_mu"] + m["K"] * m["plays_pg"] * (epa - m["epa_mu"])
    if "epa" in f:
        # Player/matchup adjustments ride inside the blend (35% weight). Tested: applying
        # them at full weight made 2025 predictions worse and pushed totals 1.5-6 pts high;
        # single-player effects are noisier than they look, so the blend acts as shrinkage.
        pts = m["w_epa"] * pts_epa + (1 - m["w_epa"]) * pts_model
    else:
        base_epa = (m["epa_mu"] + hfa * m["epa_hfa"]
                    + m["epa"]["off"].get(off, 0) - m["epa"]["def"].get(dfn, 0))
        pts = pts_model + m["K"] * m["plays_pg"] * (epa - base_epa)

    # ---- turnovers per play
    to = m["to_mu"] + m["to"]["off"].get(off, 0) - m["to"]["def"].get(dfn, 0)
    to += prate * qb_delta_to
    to_ratio = max(to, 0.004) / m["to_mu"] if "turnovers" in f else 1.0

    # ---- 4th-down aggressiveness
    coach = m["team_coach"].get(off)
    goe = {z: (m["coach_goe"].get((coach, z), 0.0) if "fourth" in f else 0.0)
           for z in ("fg", "punt")}
    conv = m["team_conv"].get(off, m["lg_conv"])

    # ---- halftime swing (offense coach vs defense coach), EPA/play -> points/half
    half = 0.0
    if "halftime" in f:
        dc = m["team_coach"].get(dfn)
        half = m["half_off"].get(coach, 0) + m["half_def"].get(dc, 0)
    half_pts = m["K"] * m["plays_pg"] / 2 * half

    out = {"to_ratio": to_ratio, "goe": goe, "conv": conv, "half_pts": half_pts, "epa": epa}

    # ---- red zone: TD vs FG finishing, heavily shrunk
    if "redzone" in f:
        pred = m["rz_lg"] + m["rz_off"].get(off, 0) + m["rz_def"].get(dfn, 0)
        pred = float(np.clip(pred, 0.3, 0.8))
        lg = m["rz_lg"]
        out["rz"] = (pred / lg, (1 - pred) / (1 - lg))
        pts += (pred - lg) * m["rz_trips_pg"] * 4.0
        notes["red_zone_td_rate"] = f"{pred:.1%} (league {lg:.1%})"

    # ---- kicker: depth chart K1 (or last team kicker), accuracy vs league by distance
    if "kicker" in f:
        kid = None
        for c in (ctx or {}).get("kickers", {}).get(off, []):
            if miss.get(c, 0) < 0.5:
                kid = c
                break
        kid = kid or m["team_kicker"].get(off)
        kk = m["kickers"]
        out["kick_off"] = float(kk.offset.get(kid, 0.0)) if kid is not None else 0.0
        if kid in kk.index:
            r = kk.loc[kid]
            notes["kicker"] = f"{r['name']} ({int(r.made)}/{int(r.att)}, {out['kick_off']:+.2f} vs avg)"

    notes.update({"coach": coach, "go_over_exp_punt_zone": round(goe["punt"], 3),
                  "go_over_exp_fg_zone": round(goe["fg"], 3)})
    out.update({"pts": pts, "notes": notes})
    return out


def h2h_adjust(m, home, away, neutral):
    """Head-to-head residual (home perspective) from meetings within 2 years."""
    g = m["games"]
    g = g[(g.gameday < m["as_of"]) & (g.gameday >= m["as_of"] - pd.Timedelta(days=730))
          & g.result.notna() & (((g.home_team == home) & (g.away_team == away))
                                | ((g.home_team == away) & (g.away_team == home)))]
    if g.empty:
        return 0.0, 0
    p = m["pts"]
    res = []
    for _, x in g.iterrows():
        hf = 0 if x.location == "Neutral" else m["pts_hfa"]
        pred = (p["off"].get(x.home_team, 0) - p["def"].get(x.away_team, 0)) - \
               (p["off"].get(x.away_team, 0) - p["def"].get(x.home_team, 0)) + hf
        r = x.result - pred
        res.append(r if x.home_team == home else -r)
    n = len(res)
    return float(np.clip(np.mean(res) * n / (n + 24), -1.5, 1.5)), n


# ============================================================== game engine
def _tilt(base, theta, tau, rz=(1.0, 1.0)):
    """Tilt league drive-result odds: theta raises scoring, tau scales turnovers,
    rz shifts scoring drives between TDs and field goals (red-zone finishing)."""
    p = base.copy()
    p[..., TD] *= np.exp(theta) * rz[0]
    p[..., FG] *= np.exp(0.55 * theta) * rz[1]
    p[..., MISS] *= np.exp(0.55 * theta) * rz[1]
    p[..., TO] *= tau
    return p / p.sum(-1, keepdims=True)


def _solve_theta(m, pts_per_drive, tau, rz=(1.0, 1.0)):
    base, w = m["base_probs"], m["bin_weights"]
    val = np.array([6.94, 3, 0, 0, 0, 0])
    lo, hi = -3.0, 3.0
    for _ in range(40):
        mid = (lo + hi) / 2
        ev = (w[:, None] * _tilt(base, mid, tau, rz) * val).sum()
        lo, hi = (mid, hi) if ev < pts_per_drive else (lo, mid)
    return (lo + hi) / 2


def _team_engine(m, t):
    """Per-half result probability tables for one offense."""
    # offensive points only (defensive/return TDs come from the engine)
    drives = m["drives_pg"]
    if not t.get("pre_cal") and m.get("cal"):  # single-team fallback
        a, b, c = m["cal"]
        t = dict(t, pts=(t["pts"] - a - c * m["pts_mu"]) / b, half_pts=t["half_pts"] / b)
    lg_def_pts = 7 * m["def_td_rate"] * m["base_probs"][:, TO] @ m["bin_weights"] * drives
    out = []
    for sgn in (-1, 1):  # 1st half, 2nd half
        target = (t["pts"] - lg_def_pts + sgn * t["half_pts"]) / drives
        rz = t.get("rz", (1.0, 1.0))
        th = _solve_theta(m, max(target, 0.3), t["to_ratio"], rz)
        out.append(_tilt(m["base_probs"], th, t["to_ratio"], rz))
    return out


def _sample_pool(m, rng, code, bins, which):
    """Vectorised draw of next-start or duration from empirical (result, bin) pools."""
    out = np.empty(len(bins))
    for b in np.unique(bins):
        idx = np.where(bins == b)[0]
        pool = m["pools"].get((code, int(b))) or m["pools"][(code, -1)]
        arr = pool[which] if len(pool[which]) else m["pools"][(code, -1)][which]
        out[idx] = arr[rng.integers(0, len(arr), len(idx))]
    return out


def simulate_game(m, home, away, n_sims=10000, neutral=False, seed=None,
                  features=ALL_FEATURES, home_qb=None, away_qb=None,
                  out_players=None, ctx=None, weather=None, referee=None):
    """Possession-by-possession Monte Carlo of one game.
    weather: optional {"temp": F, "wind": mph} forecast; otherwise games.csv / stadium average."""
    rng = np.random.default_rng(seed)
    hf = None if neutral else True
    af = None if neutral else False
    th = team_inputs(m, home, away, hf, features, home_qb, out_players, ctx)
    ta = team_inputs(m, away, home, af, features, away_qb, out_players, ctx)
    h2h, n_h2h = h2h_adjust(m, home, away, neutral) if "h2h" in features else (0.0, 0)
    th["pts"] += h2h / 2
    ta["pts"] -= h2h / 2
    margin_adj, total_adj, gnote = game_context(m, home, away, neutral, weather)
    if "rest_travel" not in features:
        margin_adj = 0.0
    if "weather" not in features:
        total_adj = 0.0
    ref_m = ref_t = 0.0
    if "referee" in features:
        ref_m, ref_t, ref_note = referee_effect(m, referee or gnote.get("referee_name"))
        asked = referee or gnote.get("referee_name")
        gnote["referee"] = ref_note or (f"'{asked}' not recognized, see referee list" if asked
                                        else "crew not assigned yet (assignments come out mid-week)")
    th["pts"] += total_adj / 2 + margin_adj / 2 + ref_t / 2 + ref_m / 2
    ta["pts"] += total_adj / 2 - margin_adj / 2 + ref_t / 2 - ref_m / 2
    gnote["referee_pts"] = round(ref_m, 2)

    th["proj_pts"], ta["proj_pts"] = th["pts"], ta["pts"]
    _apply_cal(m, th, ta)
    tabs = [_team_engine(m, th), _team_engine(m, ta)]  # [team][half] -> (bins, 6)
    teams = [th, ta]
    n = n_sims
    score = np.zeros((2, n))
    half_score = np.zeros((2, n))
    poss_count = np.zeros((2, n))
    to_count = np.zeros((2, n))

    first = rng.integers(0, 2, n)  # coin toss: who receives first
    _play_halves(m, rng, tabs, teams, score, half_score, poss_count, to_count, first)

    # overtime, 2025 rules: both teams get a possession, then sudden death, 10 minutes
    tied = score[0] == score[1]
    if tied.any():
        _overtime(m, rng, tabs, teams, score, tied)

    hs, as_ = score[0], score[1]
    margin = hs - as_
    top = pd.Series(list(zip(hs.astype(int), as_.astype(int)))).value_counts(normalize=True).head(5)
    return {
        "home": home, "away": away,
        "home_win_pct": float((margin > 0).mean()), "away_win_pct": float((margin < 0).mean()),
        "tie_pct": float((margin == 0).mean()),
        "proj_home_pts": float(hs.mean()), "proj_away_pts": float(as_.mean()),
        "proj_spread": float(-margin.mean()), "proj_total": float((hs + as_).mean()),
        "half_home": float(half_score[0].mean()), "half_away": float(half_score[1].mean()),
        "poss_home": float(poss_count[0].mean()), "poss_away": float(poss_count[1].mean()),
        "to_home": float(to_count[0].mean()), "to_away": float(to_count[1].mean()),
        "top_scores": [(f"{home} {h} - {away} {a}", round(p * 100, 1)) for (h, a), p in top.items()],
        "home_scores": hs, "away_scores": as_,
        "inputs": {"home": th, "away": ta, "h2h": h2h, "h2h_games": n_h2h,
                   "game": dict(gnote, rest_travel_pts=round(margin_adj, 2),
                                weather_total_pts=round(total_adj, 2))},
    }


def _fourth_down(m, rng, t, res, bins, tab):
    """Coach aggressiveness: turn some punts/FGs into go attempts (or the reverse)."""
    out = res.copy()
    p_base = tab[bins]
    for zone, kick_code, zbins in (("punt", PUNT, (bins >= 3) & (bins <= 5)),
                                   ("fg", FG, bins <= 3)):
        d = t["goe"][zone]
        if d == 0:
            continue
        rate = m["zone_rate"].get(zone, 0)
        lg = m["lg_go"].get(zone, 0.5)
        if d > 0:  # would have kicked/punted, now goes for it
            pk = p_base[:, kick_code].clip(1e-3)
            flip = (out == kick_code) & zbins & (rng.random(len(out)) < rate * d / pk / max(1 - lg, 0.2))
            ok = rng.random(len(out)) < t["conv"]
            if zone == "fg":
                td = rng.random(len(out)) < 0.55
                out = np.where(flip & ok, np.where(td, TD, FG), np.where(flip, DOWNS, out))
            else:
                cont = rng.random(len(out))
                new = np.where(cont < 0.33, TD, np.where(cont < 0.55, FG, PUNT))
                out = np.where(flip & ok, new, np.where(flip, DOWNS, out))
        else:  # would have gone for it, now kicks/punts
            flip = zbins & np.isin(out, [TD, FG, DOWNS]) & (rng.random(len(out)) < rate * -d / max(lg, 0.2) * 0.5)
            out = np.where(flip, kick_code, out)
    return out


def _drive(m, rng, tabs, teams, team_arr, half, spot, idx):
    """Run one drive for sims `idx` where `team_arr[idx]` has the ball."""
    k = idx.size
    res = np.empty(k, int)
    bins = ((spot - 1) // 10).clip(0, N_BINS - 1).astype(int)
    for team in (0, 1):
        s = team_arr == team
        if not s.any():
            continue
        p = tabs[team][half][bins[s]]
        u = rng.random(s.sum())
        r = (u[:, None] > p.cumsum(1)).sum(1).clip(0, 5)
        r = _fourth_down(m, rng, teams[team], r, bins[s], tabs[team][half])
        kick = np.isin(r, [FG, MISS])
        if kick.any() and "kick_off" in teams[team]:
            dist = _sample_kick_dist(m, rng, bins[s][kick])
            made = rng.random(kick.sum()) < _fg_prob(m, dist, teams[team]["kick_off"])
            r[kick] = np.where(made, FG, MISS)
        res[s] = r
    secs = np.zeros(k)
    nxt = np.zeros(k)
    for code in range(6):
        c = res == code
        if c.any():
            secs[c] = _sample_pool(m, rng, code, bins[c], 1)
            nxt[c] = _sample_pool(m, rng, code, bins[c], 0)
    return res, secs, nxt


def _kick(m, rng, k):
    return m["kick_starts"][rng.integers(0, len(m["kick_starts"]), k)]


def _play_halves(m, rng, tabs, teams, score, half_score, poss_count, to_count, first):
    n = len(first)
    for half, receiver in ((0, first), (1, 1 - first)):
        pos = receiver.copy()
        spot = _kick(m, rng, n)
        clock = np.full(n, 1800.0)
        live = np.ones(n, bool)
        for _ in range(30):
            idx = np.where(live)[0]
            if idx.size == 0:
                break
            res, secs, nxt = _drive(m, rng, tabs, teams, pos[idx], half, spot[idx], idx)
            cl = clock[idx]
            cut = (secs > cl) & (rng.random(idx.size) > cl / secs)
            res = np.where(cut, -1, res)
            team = pos[idx]
            poss_count[team, idx] += 1
            pts = np.where(res == TD, 6 + (rng.random(idx.size) < 0.94), np.where(res == FG, 3, 0))
            score[team, idx] += pts
            turnover = res == TO
            to_count[team, idx] += turnover
            pick6 = turnover & (rng.random(idx.size) < m["def_td_rate"])
            score[1 - team, idx] += 7 * pick6
            kicked = np.isin(res, [TD, FG]) | pick6
            pos[idx] = np.where(pick6, team, 1 - team)
            spot[idx] = np.where(kicked, _kick(m, rng, idx.size), np.clip(nxt, 1, 99))
            clock[idx] = cl - secs
            live[idx] = (clock[idx] > 0) & (res != -1)
        if half == 0:
            half_score[:] = score


def _overtime(m, rng, tabs, teams, score, tied):
    idx0 = np.where(tied)[0]
    n = idx0.size
    pos = rng.integers(0, 2, n)
    spot = _kick(m, rng, n)
    clock = np.full(n, 600.0)
    had = np.zeros((2, n), bool)
    live = np.ones(n, bool)
    s = score[:, idx0].copy()
    for _ in range(12):
        idx = np.where(live)[0]
        if idx.size == 0:
            break
        res, secs, nxt = _drive(m, rng, tabs, teams, pos[idx], 1, spot[idx], idx0[idx])
        cl = clock[idx]
        cut = (secs > cl) & (rng.random(idx.size) > cl / secs)
        res = np.where(cut, -1, res)
        team = pos[idx]
        pts = np.where(res == TD, 7, np.where(res == FG, 3, 0))
        s[team, idx] += pts
        pick6 = (res == TO) & (rng.random(idx.size) < m["def_td_rate"])
        s[1 - team, idx] += 7 * pick6
        had[team, idx] = True
        both = had[0, idx] & had[1, idx]
        done = pick6 | (both & (s[0, idx] != s[1, idx])) | (res == -1)
        clock[idx] = cl - secs
        done |= clock[idx] <= 0
        pos[idx] = 1 - team
        spot[idx] = np.where(np.isin(res, [TD, FG]), _kick(m, rng, idx.size), np.clip(nxt, 1, 99))
        live[idx] = ~done
    score[:, idx0] = s


# ============================================================= reporting
def print_game(r):
    h, a = r["home"], r["away"]
    fav = h if r["proj_spread"] < 0 else a
    ih, ia = r["inputs"]["home"]["notes"], r["inputs"]["away"]["notes"]
    print(f"{a} ({ia['qb']}) @ {h} ({ih['qb']})")
    print(f"  Win prob:    {h} {r['home_win_pct']:.1%} | {a} {r['away_win_pct']:.1%} | tie {r['tie_pct']:.1%}")
    print(f"  Final proj:  {h} {r['proj_home_pts']:.1f} - {a} {r['proj_away_pts']:.1f}   "
          f"(halftime {r['half_home']:.1f} - {r['half_away']:.1f})")
    print(f"  Line:        {fav} -{abs(r['proj_spread']):.1f}   Total {r['proj_total']:.1f}")
    print(f"  Possessions: {h} {r['poss_home']:.1f}, {a} {r['poss_away']:.1f}   "
          f"Turnovers: {h} {r['to_home']:.2f}, {a} {r['to_away']:.2f}")
    print("  Most likely finals:", ", ".join(f"{s} ({p}%)" for s, p in r["top_scores"][:3]))
    gi = r["inputs"]["game"]
    print(f"  Conditions:  {gi['weather']} (total {gi['weather_total_pts']:+.1f}) | rest: {gi['rest']}"
          + (f" | {gi['travel']}" if "travel" in gi else "") + f" ({h} {gi['rest_travel_pts']:+.1f})")
    if "referee" in gi:
        print(f"  Referee:     {gi['referee']} ({h} {gi['referee_pts']:+.2f})")
    for side, t in (("home", h), ("away", a)):
        n_ = r["inputs"][side]["notes"]
        bits = [n_[k] for k in ("kicker",) if k in n_]
        if "pressure_rate" in n_:
            bits.append(f"QB pressured {n_['pressure_rate']}")
        if "red_zone_td_rate" in n_:
            bits.append(f"red-zone TD {n_['red_zone_td_rate']}")
        if bits:
            print(f"  {t} offense: " + " | ".join(bits))
    for side, t in (("home", h), ("away", a)):
        n_ = r["inputs"][side]["notes"]
        if n_.get("ol_out"):
            print(f"  {t} OL starters out: {', '.join(n_['ol_out'])}")
    for side, t, opp in (("home", h, a), ("away", a, h)):
        n_ = r["inputs"][side]["notes"]
        if n_.get("opp_def_out"):
            print(f"  {opp} defensive starters out: {', '.join(n_['opp_def_out'])}")
    for side, t in (("home", h), ("away", a)):
        ch = r["inputs"][side]["notes"].get("roster_changes")
        if ch:
            print(f"  {t} roster/injury impact: {', '.join(ch)}")


def factor_breakdown(m, home, away, n_sims=6000, seed=7, **kw):
    """How much each feature moves the spread, adding them one at a time."""
    rows, feats = [], []
    base = simulate_game(m, home, away, n_sims, seed=seed, features=(), **kw)
    prev = base["proj_spread"]
    rows.append(("Team points ratings only", round(-prev, 1), 0.0, round(base["home_win_pct"] * 100, 1)))
    labels = {"epa": "EPA efficiency (pass/run)", "qb": "Starting QB", "skill": "Skill players out",
              "gameplan": "Game plan vs similar D", "style": "Style matchup residual",
              "turnovers": "Turnover tendencies / QB mistakes", "fourth": "4th-down aggressiveness",
              "halftime": "Halftime adjustments", "h2h": "Head-to-head (2 yrs)",
              "weather": "Weather / roof", "rest_travel": "Rest and travel",
              "pass_rush": "Pass rush vs protection", "kicker": "Kicker accuracy",
              "redzone": "Red-zone TD rate", "referee": "Referee crew tendencies",
              "lines": "OL / defensive starters out"}
    for f in ALL_FEATURES:
        feats.append(f)
        r = simulate_game(m, home, away, n_sims, seed=seed, features=tuple(feats), **kw)
        rows.append((labels[f], round(-r["proj_spread"], 1), round(prev - r["proj_spread"], 1),
                     round(r["home_win_pct"] * 100, 1)))
        prev = r["proj_spread"]
    return pd.DataFrame(rows, columns=["factor added", f"{home} margin", "change", f"{home} win %"])


# ============================================================== backtest
def backtest(games, prep, injuries, rosters, season, start_week=4, n_sims=3000,
             features=ALL_FEATURES, K=None, w_epa=None, weeks=None):
    """Refit before every week using only prior data, with that week's roster and
    injury report, and the actual starting QBs; grade against results and Vegas."""
    s = games[(games.season == season) & (games.game_type == "REG") & games.result.notna()]
    rows = []
    for wk in weeks or range(start_week, s.week.max() + 1):
        slate = s[s.week == wk]
        if slate.empty:
            continue
        m = fit_model(prep, slate.gameday.min(), K=K, w_epa=w_epa)
        ctx = build_context(season, wk, rosters, injuries) if rosters is not None else None
        for _, g in slate.iterrows():
            r = simulate_game(m, g.home_team, g.away_team, n_sims, neutral=g.location == "Neutral",
                              seed=1, features=features, home_qb=g.home_qb_id,
                              away_qb=g.away_qb_id, ctx=ctx)
            p = r["home_win_pct"] + r["tie_pct"] / 2
            won = 1.0 if g.result > 0 else (0.5 if g.result == 0 else 0.0)
            rows.append({"week": wk, "game": f"{g.away_team}@{g.home_team}", "p_home": p,
                         "home_won": won, "model_margin": -r["proj_spread"],
                         "vegas_margin": g.spread_line, "actual_margin": g.result,
                         "model_total": r["proj_total"], "actual_total": g.total,
                         "vegas_total": g.total_line})
    bt = pd.DataFrame(rows)
    dec = bt[bt.home_won != 0.5]
    return bt, {
        "games": len(bt),
        "pick_acc": ((dec.p_home > 0.5) == (dec.home_won == 1)).mean(),
        "vegas_pick_acc": ((dec.vegas_margin > 0) == (dec.home_won == 1)).mean(),
        "brier": ((bt.p_home - bt.home_won) ** 2).mean(),
        "margin_MAE": (bt.model_margin - bt.actual_margin).abs().mean(),
        "vegas_margin_MAE": (bt.vegas_margin - bt.actual_margin).abs().mean(),
        "total_MAE": (bt.model_total - bt.actual_total).abs().mean(),
        "vegas_total_MAE": (bt.vegas_total - bt.actual_total).abs().mean(),
        "vegas_brier": ((1 / (1 + np.exp(-bt.vegas_margin / 6.5)) - bt.home_won) ** 2).mean(),
    }


# ============================================================ convenience
def simulate_week(m, season, week, ctx=None, n_sims=6000, seed=0, features=ALL_FEATURES):
    g = m["games"]
    slate = g[(g.season == season) & (g.week == week)]
    rows = []
    for _, x in slate.iterrows():
        r = simulate_game(m, x.home_team, x.away_team, n_sims, neutral=x.location == "Neutral",
                          seed=seed, features=features, ctx=ctx)
        rows.append({"matchup": f"{x.away_team} @ {x.home_team}",
                     "QBs": f"{r['inputs']['away']['notes']['qb']} / {r['inputs']['home']['notes']['qb']}",
                     "home_win_%": round(r["home_win_pct"] * 100, 1),
                     "model_line": round(r["proj_spread"], 1),
                     "vegas_line": -x.spread_line if pd.notna(x.spread_line) else np.nan,
                     "model_total": round(r["proj_total"], 1), "vegas_total": x.total_line})
    return pd.DataFrame(rows)


def coach_table(m):
    """4th-down go-rate over expected and halftime swing, current head coaches."""
    rows = []
    for team, coach in m["team_coach"].items():
        rows.append({"team": team, "coach": coach,
                     "4th_GOE_punt_zone": m["coach_goe"].get((coach, "punt"), 0),
                     "4th_GOE_fg_zone": m["coach_goe"].get((coach, "fg"), 0),
                     "halftime_off_swing": m["half_off"].get(coach, 0),
                     "halftime_def_swing": -m["half_def"].get(coach, 0)})
    return pd.DataFrame(rows).set_index("team").round(3).sort_values("4th_GOE_punt_zone", ascending=False)


def qb_table(m, min_db=150):
    """Active QBs (played in the rating window); db includes any older history added."""
    q = m["qbs"][(m["qbs"].db >= min_db) & m["qbs"]["last"].notna()]
    return (q.assign(epa_per_db=q.epa_db, turnover_pct=q.to_db * 100)
             [["name", "team", "db", "epa_per_db", "turnover_pct"]]
             .sort_values("epa_per_db", ascending=False).round(3))


# ============================================================ grading
def grade_vs_vegas(bt):
    """Model vs Vegas closing lines: winners, spread/total error, ATS and over/under
    records (model picks the side it likes relative to the Vegas number)."""
    from math import comb
    def p_at_least(k, n):
        return sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n if n else np.nan
    dec = bt[bt.home_won != 0.5]
    ats = bt.actual_margin - bt.vegas_margin
    pick = np.sign(bt.model_margin - bt.vegas_margin)
    a = bt[(ats != 0) & (pick != 0) & bt.vegas_margin.notna()]
    aw = int((np.sign(ats[a.index]) == pick[a.index]).sum())
    ou = bt.actual_total - bt.vegas_total
    op = np.sign(bt.model_total - bt.vegas_total)
    o = bt[(ou != 0) & (op != 0) & bt.vegas_total.notna()]
    ow = int((np.sign(ou[o.index]) == op[o.index]).sum())
    return pd.Series({
        "games": len(bt),
        "winner_pct_model": ((dec.p_home > .5) == (dec.home_won == 1)).mean(),
        "winner_pct_vegas": ((dec.vegas_margin > 0) == (dec.home_won == 1)).mean(),
        "spread_miss_model": (bt.model_margin - bt.actual_margin).abs().mean(),
        "spread_miss_vegas": (bt.vegas_margin - bt.actual_margin).abs().mean(),
        "total_miss_model": (bt.model_total - bt.actual_total).abs().mean(),
        "total_miss_vegas": (bt.vegas_total - bt.actual_total).abs().mean(),
        "ATS_record": f"{aw}-{len(a) - aw}", "ATS_pct": aw / max(len(a), 1),
        "ATS_p_value": p_at_least(aw, len(a)),
        "OU_record": f"{ow}-{len(o) - ow}", "OU_pct": ow / max(len(o), 1),
        "OU_p_value": p_at_least(ow, len(o)),
        "model_total_bias": (bt.model_total - bt.actual_total).mean(),
    })


def ats_detail(bt):
    """Game-by-game picks against the spread and total."""
    d = bt.copy()
    away, home = d.game.str.split("@").str[0], d.game.str.split("@").str[1]
    d["model_line"], d["vegas_line"] = (-d.model_margin).round(1), -d.vegas_margin
    d["ATS_pick"] = np.where(d.model_margin > d.vegas_margin, home, away)
    ats = d.actual_margin - d.vegas_margin
    d["ATS"] = np.where(ats == 0, "push", np.where(np.sign(ats) == np.sign(d.model_margin - d.vegas_margin), "W", "L"))
    d["OU_pick"] = np.where(d.model_total > d.vegas_total, "over", "under")
    ou = d.actual_total - d.vegas_total
    d["OU"] = np.where(ou == 0, "push", np.where(np.sign(ou) == np.sign(d.model_total - d.vegas_total), "W", "L"))
    return d[["week", "game", "model_line", "vegas_line", "actual_margin", "ATS_pick", "ATS",
              "model_total", "vegas_total", "actual_total", "OU_pick", "OU"]].round(1)
