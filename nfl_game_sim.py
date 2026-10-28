"""
NFL game simulator v2: play-by-play powered, possession-based.
Data: nflverse (github.com/nflverse). Usage: see NFL_Game_Simulator_v2.ipynb.

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
"""

from pathlib import Path
import numpy as np
import pandas as pd

GAMES_URL = "https://github.com/nflverse/nfldata/raw/master/data/games.csv"
REL = "https://github.com/nflverse/nflverse-data/releases/download"
TEAM_FIX = {"OAK": "LV", "SD": "LAC", "STL": "LA", "LAR": "LA"}
INJ_WEIGHT = {"Out": 1.0, "Doubtful": 0.85, "Questionable": 0.25}

ALL_FEATURES = ("epa", "qb", "skill", "gameplan", "style", "turnovers",
                "fourth", "halftime", "h2h")

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

    tg = off.groupby(["game_id", "posteam", "defteam"]).agg(
        plays=("epa", "size"), epa=("epa", "sum"),
        pass_n=("is_pass", "sum"), pass_epa=("pass_epa", "sum"),
        rush_epa=("rush_epa", "sum"), to=("to", "sum"),
        h1=("h1", "sum"), h1_epa=("h1_epa", "sum"),
        h2=("h2", "sum"), h2_epa=("h2_epa", "sum"),
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
        season=("season", "first")).reset_index().sort_values(["game_id", "fixed_drive"])
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

    return {"games": games, "tg": tg, "qb": qb, "skill": skill, "fd": fd, "drives": dr}


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
    _calibrate(m)
    m["games"] = prep["games"]
    return m


def _calibrate(m, n=6000):
    """Game flow (possessions, field position) compresses scoring toward the mean.
    Measure that and invert it so simulated points match the projected points."""
    m["cal"] = (0.0, 1.0)
    rng = np.random.default_rng(123)
    mu = m["pts_mu"]
    avg = {"pts": mu, "to_ratio": 1.0, "goe": {"fg": 0, "punt": 0}, "conv": m["lg_conv"], "half_pts": 0}
    xs, ys = [], []
    for tgt in (mu - 9, mu, mu + 9):
        t = dict(avg, pts=tgt)
        tabs = [_team_engine(m, t), _team_engine(m, avg)]
        sc, hs, pc, tc = (np.zeros((2, n)) for _ in range(4))
        _play_halves(m, rng, tabs, [t, avg], sc, hs, pc, tc, rng.integers(0, 2, n))
        xs.append(tgt); ys.append(sc[0].mean())
    b, a = np.polyfit(xs, ys, 1)
    m["cal"] = (a, b)


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
    g["epa_db"] = _shrink(g.we, g.wd, repl, 150)
    g["to_db"] = _shrink(g.wt, g.wd, lg_to * 1.1, 250)
    m["qbs"], m["qb_repl"], m["qb_lg_to"] = g, repl, lg_to

    # team's recency-weighted QB baseline (what its ratings were built with)
    qb["epa_db"] = qb.qb_id.map(g.epa_db)
    qb["to_db"] = qb.qb_id.map(g.to_db)
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
    s = tg[["posteam", "defteam", "plays", "pass_n", "w", "gameday"]].copy()
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
    qb1 = {}
    if depth is not None and len(depth):
        d = depth[depth.dt == depth.groupby("team").dt.transform("max")]
        d = d[d.pos_abb == "QB"].sort_values("pos_rank")
        for team, x in d.groupby("team"):
            qb1[team] = x.gsis_id.tolist()
    names = snap.drop_duplicates("gsis_id", keep="last").set_index("gsis_id").full_name
    return {"team_now": team_now, "miss": miss, "why": why, "qb_depth": qb1,
            "names": names, "label": f"{season} week {week}"}


def _availability(m, team, ctx, out_players):
    miss = dict(ctx["miss"]) if ctx else {}
    for name in out_players or []:
        hit = m["qbs"][m["qbs"].name.str.lower() == name.lower()].index.tolist()
        hit += m["skill"][m["skill"].name.str.lower() == name.lower()].pid.tolist()
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
    epa += hfa * m["epa_hfa"] + style_resid
    epa += prate * (qb_delta_epa + skill_pass) + (1 - prate) * skill_rush

    # ---- points: EPA-based projection blended with the points model
    pts_model = m["pts_mu"] + m["pts"]["off"].get(off, 0) - m["pts"]["def"].get(dfn, 0) \
        + hfa * m["pts_hfa"]
    pts_epa = m["pts_mu"] + m["K"] * m["plays_pg"] * (epa - m["epa_mu"])
    if "epa" in f:
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

    notes.update({"coach": coach, "go_over_exp_punt_zone": round(goe["punt"], 3),
                  "go_over_exp_fg_zone": round(goe["fg"], 3)})
    return {"pts": pts, "to_ratio": to_ratio, "goe": goe, "conv": conv,
            "half_pts": half_pts, "epa": epa, "notes": notes}


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
def _tilt(base, theta, tau):
    """Tilt league drive-result odds: theta raises scoring, tau scales turnovers."""
    p = base.copy()
    p[..., TD] *= np.exp(theta)
    p[..., FG] *= np.exp(0.55 * theta)
    p[..., MISS] *= np.exp(0.55 * theta)
    p[..., TO] *= tau
    return p / p.sum(-1, keepdims=True)


def _solve_theta(m, pts_per_drive, tau):
    base, w = m["base_probs"], m["bin_weights"]
    val = np.array([6.94, 3, 0, 0, 0, 0])
    lo, hi = -3.0, 3.0
    for _ in range(40):
        mid = (lo + hi) / 2
        ev = (w[:, None] * _tilt(base, mid, tau) * val).sum()
        lo, hi = (mid, hi) if ev < pts_per_drive else (lo, mid)
    return (lo + hi) / 2


def _team_engine(m, t):
    """Per-half result probability tables for one offense."""
    # offensive points only (defensive/return TDs come from the engine)
    drives = m["drives_pg"]
    a, b = m.get("cal", (0.0, 1.0))
    t = dict(t, pts=(t["pts"] - a) / b, half_pts=t["half_pts"] / b)
    lg_def_pts = 7 * m["def_td_rate"] * m["base_probs"][:, TO] @ m["bin_weights"] * drives
    out = []
    for sgn in (-1, 1):  # 1st half, 2nd half
        target = (t["pts"] - lg_def_pts + sgn * t["half_pts"]) / drives
        th = _solve_theta(m, max(target, 0.3), t["to_ratio"])
        out.append(_tilt(m["base_probs"], th, t["to_ratio"]))
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
                  out_players=None, ctx=None):
    """Possession-by-possession Monte Carlo of one game."""
    rng = np.random.default_rng(seed)
    hf = None if neutral else True
    af = None if neutral else False
    th = team_inputs(m, home, away, hf, features, home_qb, out_players, ctx)
    ta = team_inputs(m, away, home, af, features, away_qb, out_players, ctx)
    h2h, n_h2h = h2h_adjust(m, home, away, neutral) if "h2h" in features else (0.0, 0)
    th["pts"] += h2h / 2
    ta["pts"] -= h2h / 2

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
        "inputs": {"home": th, "away": ta, "h2h": h2h, "h2h_games": n_h2h},
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
        res[s] = _fourth_down(m, rng, teams[team], r, bins[s], tabs[team][half])
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
              "halftime": "Halftime adjustments", "h2h": "Head-to-head (2 yrs)"}
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
                         "model_total": r["proj_total"], "actual_total": g.total})
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
    q = m["qbs"][m["qbs"].db >= min_db]
    return (q.assign(epa_per_db=q.epa_db, turnover_pct=q.to_db * 100)
             [["name", "team", "db", "epa_per_db", "turnover_pct"]]
             .sort_values("epa_per_db", ascending=False).round(3))
