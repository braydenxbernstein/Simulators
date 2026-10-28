"""
Player prop projections on top of the NFL game simulator (v6).

Read-only on the game simulator: props use a game simulation's output (projected points,
starting QB) but never change it, so game predictions are identical with or without props.

How a projection is built
  1. Team volume: pass attempts, targets and carries per game (recency-weighted),
     adjusted for game script (teams expected to lead run more, trailing teams pass more).
  2. Player role: target share, carry share, QB attempt share over his recent games,
     with teammates who are out redistributing part of their share.
  3. Player efficiency: yards per target / carry / attempt, catch rate, completion %,
     shrunk toward position averages and adjusted for the opponent's defense.
  4. Touchdowns: the simulator's projected points -> team TDs -> each player's share.

Markets: passing yards, pass attempts, completions, passing TDs, rushing yards,
rush attempts, receiving yards, receptions, anytime TD (probability).
"""

import numpy as np
import pandas as pd

import nfl_game_sim_v6 as ns

RZ_GAMES = 17    # games of each player's history used for his red-zone share
RZ_K = 12        # team inside-10 carries of prior (the back's overall carry share) per back
USE_RZ_SHARE = True  # split rushing TDs by red-zone (inside-10) carry share, not total carries
REC_TD_K = 40   # targets of league-average TD rate added to each receiver (shrinkage)
# Players not on the injury report who missed their team's recent games often don't play
# (benched, inactive, practice squad, unreported injury). Chance they play, by how much of the
# offense they had (targets + carries share), measured on 2025-26 games:
RECENT_MISS_ADJ = True
MISSED_LAST_P = ((0.03, 0.47), (0.08, 0.66), (0.15, 0.75), (0.30, 0.83), (9.0, 1.0))   # missed last game
MISSED_TWO_P = ((0.15, 0.27), (0.30, 0.42), (9.0, 0.51))                              # missed last 2+
ROLE_GAMES, ROLE_HALFLIFE = 8, 3   # games (and recency half-life) behind each player's role and rates
# efficiency shrinkage: weighted targets / carries / attempts of position-average added to each player
SHRINK_K = {"ypt": 15, "catch": 15, "ypc": 50, "ypa": 150, "comp": 150}   # ypt/catch were 40 before the Oct 2026 refit

MARKETS = {  # our column -> actual stat column in nflverse weekly stats
    "pass_yds": "passing_yards", "pass_att": "attempts", "completions": "completions",
    "pass_tds": "passing_tds", "rush_yds": "rushing_yards", "rush_att": "carries",
    "rec_yds": "receiving_yards", "receptions": "receptions",
}
# Volume calibration fitted on the 2025 backtest (projected totals ran below actual for
# rushing/receiving), validated out of sample on 2026 weeks 1-3.
CAL = {"rush_att": 1.144, "rush_yds": 1.173, "rec_yds": 1.097, "receptions": 1.075, "targets": 1.075}

ODDS_API_MARKETS = {  # The Odds API market key -> our column
    "player_pass_yds": "pass_yds", "player_pass_attempts": "pass_att",
    "player_pass_completions": "completions", "player_pass_tds": "pass_tds",
    "player_rush_yds": "rush_yds", "player_rush_attempts": "rush_att",
    "player_reception_yds": "rec_yds", "player_receptions": "receptions",
    "player_anytime_td": "anytime_td",
}


# ------------------------------------------------------------------ data
def load_player_stats(seasons, games, data_dir="nfl_data"):
    frames = []
    for y in seasons:
        p = ns._cached(f"{ns.REL}/stats_player/stats_player_week_{y}.parquet",
                       f"{data_dir}/stats_player_week_{y}.parquet")
        frames.append(pd.read_parquet(p))
    s = pd.concat(frames, ignore_index=True)
    s["team"] = s.team.replace(ns.TEAM_FIX)
    s["opponent_team"] = s.opponent_team.replace(ns.TEAM_FIX)
    gd = games.set_index("game_id").gameday
    s = s[s.game_id.isin(gd.index)].copy()
    s["gameday"] = s.game_id.map(gd)
    s["tds"] = s.rushing_tds.fillna(0) + s.receiving_tds.fillna(0)
    return s


def _recent(df, key, n, halflife):
    """Last n games per `key`, with recency weights (most recent = 1)."""
    df = df.sort_values("gameday")
    df["rk"] = df.groupby(key).cumcount(ascending=False)
    df = df[df.rk < n].copy()
    df["w"] = 0.5 ** (df.rk / halflife)
    return df


# ------------------------------------------------------------------ fitting
def fit_props(m, stats):
    """Everything projections need, using only games before the model's as_of date."""
    as_of = m["as_of"]
    s = stats[stats.gameday < as_of]
    s = s[s.gameday >= as_of - pd.Timedelta(days=500)]  # this season + last season
    P = {"as_of": as_of}

    # team volume per game
    team = s.groupby(["game_id", "team", "opponent_team", "gameday"]).agg(
        att=("attempts", "sum"), tgt=("targets", "sum"), car=("carries", "sum"),
        pyds=("passing_yards", "sum"), comp=("completions", "sum"),
        ryds=("rushing_yards", "sum"), recyds=("receiving_yards", "sum"),
        rec=("receptions", "sum"), ptd=("passing_tds", "sum"), rtd=("rushing_tds", "sum")).reset_index()
    rt = _recent(team, "team", 10, 4)
    lg = team[["att", "tgt", "car"]].mean()
    vol = rt.groupby("team").apply(lambda d: pd.Series({
        c: (np.average(d[c], weights=d.w) * d.w.sum() + lg[c] * 3) / (d.w.sum() + 3)
        for c in ("att", "tgt", "car")}), include_groups=False)
    P["volume"], P["lg_volume"] = vol, lg

    # opponent defense ratios (yards per target / carry / attempt allowed vs league), shrunk
    dfn = _recent(team, "opponent_team", 12, 6)
    lg_ypt = team.recyds.sum() / team.tgt.sum()
    lg_ypc = team.ryds.sum() / team.car.sum()
    lg_ypa = team.pyds.sum() / team.att.sum()
    lg_cr = team.rec.sum() / team.tgt.sum()

    def ratio(num, den, lgv, k):
        g = dfn.groupby("opponent_team").apply(
            lambda d: ((d[num] * d.w).sum() + lgv * k) / ((d[den] * d.w).sum() + k), include_groups=False)
        return (g / lgv).clip(0.8, 1.2)
    P["def_ypt"] = ratio("recyds", "tgt", lg_ypt, 150)
    P["def_ypc"] = ratio("ryds", "car", lg_ypc, 120)
    P["def_ypa"] = ratio("pyds", "att", lg_ypa, 150)
    P["def_cr"] = ratio("rec", "tgt", lg_cr, 150)

    # touchdown conversion
    pts = m["games"][(m["games"].gameday < as_of) & m["games"].result.notna()]
    pts = pts[pts.gameday >= as_of - pd.Timedelta(days=500)]
    tot_pts = pts.home_score.sum() + pts.away_score.sum()
    P["td_per_pt"] = (team.ptd.sum() + team.rtd.sum()) / max(tot_pts, 1)
    P["lg_pass_td_share"] = team.ptd.sum() / max(team.ptd.sum() + team.rtd.sum(), 1)
    ts = rt.groupby("team")[["ptd", "rtd"]].sum()
    P["pass_td_share"] = ((ts.ptd + P["lg_pass_td_share"] * 8) / (ts.ptd + ts.rtd + 8))

    # game script: pass-rate shift per point of expected margin, from pregame spreads
    g = m["games"][(m["games"].gameday < as_of) & m["games"].spread_line.notna()]
    tm = team.merge(g[["game_id", "home_team", "spread_line"]], on="game_id")
    tm["exp_margin"] = np.where(tm.team == tm.home_team, tm.spread_line, -tm.spread_line)
    tm["prate"] = tm.att / (tm.att + tm.car)
    tm["prate_dev"] = tm.prate - tm.groupby("team").prate.transform("mean")
    ok = tm.prate_dev.notna() & np.isfinite(tm.exp_margin)
    P["script_slope"] = float(np.polyfit(tm.exp_margin[ok], tm.prate_dev[ok], 1)[0])
    # inside a simulation the final margin is known, and trailing teams throw more as the game
    # goes; this slope ties pass rate to the simulated final margin
    res = m["games"].set_index("game_id").result
    tm["final_margin"] = np.where(tm.team == tm.home_team, tm.game_id.map(res), -tm.game_id.map(res))
    ok2 = tm.prate_dev.notna() & np.isfinite(tm.final_margin)
    P["script_slope_final"] = float(np.polyfit(tm.final_margin[ok2], tm.prate_dev[ok2], 1)[0])

    # players: recent role and efficiency (last 8 games they played, any team)
    ps = s.merge(team[["game_id", "team", "att", "tgt", "car"]].rename(
        columns={"att": "t_att", "tgt": "t_tgt", "car": "t_car"}), on=["game_id", "team"])
    pr = _recent(ps, "player_id", ROLE_GAMES, ROLE_HALFLIFE)
    agg = pr.assign(**{f"w_{c}": pr[c] * pr.w for c in (
        "targets", "t_tgt", "carries", "t_car", "attempts", "t_att", "receptions",
        "receiving_yards", "rushing_yards", "passing_yards", "completions",
        "receiving_tds", "rushing_tds")})
    wcols = [c for c in agg.columns if c.startswith("w_")]
    pl = agg.groupby("player_id")[wcols + ["w"]].sum()
    info = pr.sort_values("gameday").groupby("player_id")[[
        "player_display_name", "position", "team"]].last()
    pl = pl.join(info)
    # shares (shrunk a little toward 0 for small samples)
    pl["tgt_share"] = pl.w_targets / (pl.w_t_tgt + 2)
    pl["car_share"] = pl.w_carries / (pl.w_t_car + 2)
    pl["att_share"] = pl.w_attempts / (pl.w_t_att + 2)
    # efficiency, shrunk to position averages
    for pos_col, num, den, k, name in (
            ("position", "w_receiving_yards", "w_targets", SHRINK_K["ypt"], "ypt"),
            ("position", "w_receptions", "w_targets", SHRINK_K["catch"], "catch"),
            ("position", "w_rushing_yards", "w_carries", SHRINK_K["ypc"], "ypc"),
            ("position", "w_passing_yards", "w_attempts", SHRINK_K["ypa"], "ypa"),
            ("position", "w_completions", "w_attempts", SHRINK_K["comp"], "comp")):
        pos_mean = pl.groupby(pos_col).apply(
            lambda d: d[num].sum() / max(d[den].sum(), 1e-9), include_groups=False)
        prior = pl[pos_col].map(pos_mean).fillna(pl[num].sum() / max(pl[den].sum(), 1e-9))
        pl[name] = (pl[num] + prior * k) / (pl[den] + k)
    # receiving TDs per target, heavily shrunk to league (TD rates barely repeat)
    P["lg_rec_td_per_tgt"] = team.ptd.sum() / max(team.tgt.sum(), 1)
    P["lg_rush_td_per_car"] = team.rtd.sum() / max(team.car.sum(), 1)
    pl["rec_td_rate"] = (pl.w_receiving_tds + P["lg_rec_td_per_tgt"] * REC_TD_K) / (pl.w_targets + REC_TD_K)
    # red-zone role: each ball carrier's share of his team's inside-10 carries in the games he
    # played (last RZ_GAMES), shrunk toward his overall carry share with RZ_K carries of prior
    rz = m.get("prep", {}).get("rz_rush") if isinstance(m.get("prep"), dict) else None
    pl["rz_share"] = pl["car_share"]
    if rz is not None and len(rz):
        pr_rz = _recent(ps, "player_id", RZ_GAMES, RZ_GAMES / 2)   # longer window: red-zone carries are sparse
        g_rz = rz[rz.game_id.isin(pr_rz.game_id)]
        team_rz = g_rz.groupby(["game_id", "team"]).rz_car.sum()
        mine = pr_rz[["game_id", "team", "player_id", "w"]].merge(
            g_rz[["game_id", "player_id", "rz_car"]], on=["game_id", "player_id"], how="left").fillna({"rz_car": 0})
        mine["t_rz"] = team_rz.reindex(pd.MultiIndex.from_frame(mine[["game_id", "team"]])).fillna(0).to_numpy()
        agg_rz = mine.assign(wr=mine.rz_car * mine.w, wt=mine.t_rz * mine.w).groupby("player_id")[["wr", "wt"]].sum()
        agg_rz = agg_rz.reindex(pl.index).fillna(0)
        pl["rz_share"] = (agg_rz.wr + RZ_K * pl.car_share) / (agg_rz.wt + RZ_K)
    pl["rz_ratio"] = (pl.rz_share / pl.car_share.replace(0, np.nan)).clip(0.15, 3.0).fillna(1.0)
    P["players"] = pl
    # who played in each team's last two games of this season (for the recent-miss adjustment)
    wk = s[["season", "week", "player_id"]].drop_duplicates()
    P["player_weeks"] = set(zip(wk.season, wk.week, wk.player_id))
    tw = team.merge(s[["game_id", "season", "week"]].drop_duplicates(), on="game_id")
    tw = tw.sort_values("gameday").groupby("team").tail(2)
    P["team_last_weeks"] = {t: list(zip(x.season, x.week))[::-1] for t, x in tw.groupby("team")}
    P["season_now"] = int(s.season.max()) if len(s) else None

    # team passing efficiency over recent games (what receivers' rates were earned with)
    tp = rt.groupby("team").apply(lambda d: pd.Series({
        "ypa": (d.pyds * d.w).sum() / max((d.att * d.w).sum(), 1),
        "comp": (d.comp * d.w).sum() / max((d.att * d.w).sum(), 1)}), include_groups=False)
    P["team_ypa"], P["team_comp"] = tp.ypa, tp.comp
    P["lg_ypa"], P["lg_comp"] = lg_ypa, team.comp.sum() / max(team.att.sum(), 1)
    # QB quality -> passing efficiency (for QBs rated by the game model's EPA history only)
    q = s[s.position == "QB"].groupby("player_id").agg(att=("attempts", "sum"), yds=("passing_yards", "sum"),
                                                       comp=("completions", "sum"), epa=("passing_epa", "sum"))
    q = q[q.att >= 150]
    x = q.epa / q.att
    P["qb_slope_ypa"] = float(np.polyfit(x, q.yds / q.att, 1)[0]) if len(q) > 10 else 4.3
    P["qb_slope_comp"] = float(np.polyfit(x, q.comp / q.att, 1)[0]) if len(q) > 10 else 0.15
    return P


# ------------------------------------------------------------------ projection
def project_game(m, P, home, away, ctx=None, sim=None, n_sims=4000, out_players=None, **sim_kw):
    """Projected stat lines for every relevant player in one game."""
    if sim is None:
        sim = ns.simulate_game(m, home, away, n_sims, seed=3, ctx=ctx, out_players=out_players, **sim_kw)
    rows = []
    for side, team, opp in (("home", home, away), ("away", away, home)):
        pts = sim[f"proj_{side}_pts"]
        margin = pts - sim[f"proj_{'away' if side == 'home' else 'home'}_pts"]
        rows += _project_team(m, P, team, opp, pts, margin, sim["inputs"][side], ctx, out_players)
    df = pd.DataFrame(rows)
    df["home"], df["away"] = home, away
    return df


def _team_setup(m, P, team, opp, t_inputs, ctx, out_players):
    """Everything about one offense that doesn't depend on the score: volume, roles, QB."""
    vol = P["volume"].loc[team] if team in P["volume"].index else P["lg_volume"]
    miss = dict((ctx or {}).get("miss", {}))
    for n in out_players or []:
        for pid, r in P["players"].iterrows():
            if str(r.player_display_name).lower() == n.lower() or ns._short(r.player_display_name) == n.lower():
                miss[pid] = 1.0
    team_now = (ctx or {}).get("team_now")
    pl = P["players"]
    if team_now is not None:
        cand = pl[pl.index.map(lambda p: team_now.get(p, None)) == team]
    else:
        cand = pl[pl.team == team]
    cand = cand.copy()
    # same rule as the game simulation: 50%+ likely to miss = out; below that, share is trimmed
    mw = cand.index.map(lambda p: miss.get(p, 0.0)).astype(float)
    cand["avail"] = np.where(mw >= 0.5, 0.0, 1 - mw)
    if RECENT_MISS_ADJ:
        cand["avail"] = cand["avail"] * _recent_miss_factor(P, team, cand, mw, (ctx or {}).get("cleared", set()))

    # starting QB = exactly the QB the game simulation started (overrides included)
    qb_name = str(t_inputs["notes"].get("qb", "")).split(" (")[0].strip()

    def _match(df):
        hit = df[(df.position == "QB") & ((df.player_display_name.str.lower() == qb_name.lower())
                                          | (df.player_display_name.map(ns._short) == qb_name.lower()))]
        return hit.index[0] if len(hit) else None

    qb_id = _match(cand[cand.avail > 0])
    if qb_id is None and qb_name:
        other = _match(P["players"])  # e.g. a QB forced in from another roster
        if other is not None:
            cand.loc[other] = P["players"].loc[other]
            cand.loc[other, "avail"] = 1.0
            cand.loc[other, "tgt_share"] = 0.0
            qb_id = other
    if qb_id is None and qb_name:
        # no recent stats (e.g. a veteran backup): rate him at QB position averages
        qbp = P["players"][P["players"].position == "QB"]
        wts = qbp.w_attempts.clip(lower=0)
        generic = {c: np.average(qbp[c], weights=wts) for c in ("comp", "ypa", "ypc", "catch", "ypt", "rec_td_rate")}
        generic.update(player_display_name=qb_name, position="QB", team=team, tgt_share=0.0,
                       car_share=0.04, att_share=0.97, avail=1.0)
        for c in cand.columns:
            generic.setdefault(c, 0.0)
        cand.loc[f"generic_qb_{team}"] = pd.Series(generic)
        qb_id = f"generic_qb_{team}"
    if qb_id is None:
        qbs = cand[(cand.position == "QB") & (cand.avail > 0)]
        qb_id = qbs.sort_values("att_share").index[-1] if len(qbs) else None

    # Shares must add up to the team's real volume: players who changed teams bring their
    # old share and absent players free theirs up, so rescale the available players to the
    # share of targets/carries a team's known players typically get (the rest goes to
    # backups we can't see yet).
    for share, cover in (("tgt_share", 1.0), ("car_share", 1.0)):
        live = (cand[share] * cand.avail).sum()
        if live > 0:
            cand[share] = cand[share] * cand.avail * (cover / live)
    return {"vol": vol, "cand": cand, "qb_id": qb_id,
            "d_ypt": P["def_ypt"].get(opp, 1.0), "d_ypc": P["def_ypc"].get(opp, 1.0),
            "d_ypa": P["def_ypa"].get(opp, 1.0), "d_cr": P["def_cr"].get(opp, 1.0)}


def _recent_miss_factor(P, team, cand, mw, cleared):
    """Chance each player plays, for players with no injury designation who missed the
    team's most recent game(s) this season. ESPN-cleared ("Active") players are exempt."""
    last = P.get("team_last_weeks", {}).get(team, [])
    if not last or last[0][0] != P.get("season_now"):
        return np.ones(len(cand))          # new season (week 1): last year's games don't count
    pw = P.get("player_weeks", set())
    same = [w for w in last if w[0] == last[0][0]]
    share = (cand.tgt_share.fillna(0) + cand.car_share.fillna(0)).to_numpy()
    out = np.ones(len(cand))
    for i, (pid, pos) in enumerate(zip(cand.index, cand.position)):
        if pos == "QB" or mw[i] > 0 or pid in cleared:
            continue
        m1 = (same[0][0], same[0][1], pid) not in pw
        m2 = m1 and len(same) > 1 and (same[1][0], same[1][1], pid) not in pw
        if not m1:
            continue
        table = MISSED_TWO_P if m2 else MISSED_LAST_P
        out[i] = next(p for cap, p in table if share[i] < cap)
    return out


def _project_team(m, P, team, opp, pts, exp_margin, t_inputs, ctx, out_players):
    S = _team_setup(m, P, team, opp, t_inputs, ctx, out_players)
    vol, cand, qb_id = S["vol"], S["cand"], S["qb_id"]
    base_rate = vol.att / (vol.att + vol.car)
    prate = float(np.clip(base_rate + P["script_slope"] * exp_margin, 0.35, 0.75))
    plays = vol.att + vol.car
    att, car = plays * prate, plays * (1 - prate)
    tgt = vol.tgt * att / vol.att

    pass_tds = pts * P["td_per_pt"] * P["pass_td_share"].get(team, P["lg_pass_td_share"])
    rush_tds = pts * P["td_per_pt"] - pass_tds
    d_ypt, d_ypc = P["def_ypt"].get(opp, 1.0), P["def_ypc"].get(opp, 1.0)
    d_ypa, d_cr = P["def_ypa"].get(opp, 1.0), P["def_cr"].get(opp, 1.0)

    out = []
    tot_rec_td_w = (cand.tgt_share * cand.rec_td_rate / P["lg_rec_td_per_tgt"] * cand.avail).sum()
    rzw = cand.car_share * (cand.rz_ratio if USE_RZ_SHARE and "rz_ratio" in cand else 1.0)
    tot_car = (rzw * cand.avail).sum()
    for pid, r in cand.iterrows():
        if r.avail <= 0 or (r.position == "QB" and pid != qb_id):
            continue
        rec = {"player_id": pid, "player": r.player_display_name, "pos": r.position, "team": team, "opp": opp}
        if r.position == "QB":
            a = att * 0.97
            rec.update(pass_att=a, completions=a * r.comp * d_cr, pass_yds=a * r.ypa * d_ypa,
                       pass_tds=pass_tds)
        c = car * r.car_share
        t = tgt * r.tgt_share
        rec.update(rush_att=c, rush_yds=c * r.ypc * d_ypc, targets=t,
                   receptions=t * r.catch * d_cr, rec_yds=t * r.ypt * d_ypt)
        rec_td = pass_tds * (r.tgt_share * r.rec_td_rate / P["lg_rec_td_per_tgt"]) / max(tot_rec_td_w, 1e-9) \
            if r.position != "QB" else 0.0
        rush_td = rush_tds * (rzw.loc[pid] / max(tot_car, 1e-9))
        for k, f in CAL.items():
            if k in rec:
                rec[k] *= f
        rec["exp_tds"] = rec_td + rush_td
        rec["anytime_td"] = 1 - np.exp(-(rec_td + rush_td))
        rb = r.position in ("RB", "FB")
        if (rec.get("pass_att", 0) > 5) or t >= 2.0 or c >= 4.0 or (rb and (c >= 1.5 or t >= 1.0)):
            out.append(rec)
    return out


# ------------------------------------------------------------------ simulated props
# Game-to-game noise the averages can't see, tuned so simulated ranges match real ones.
# Oct 2026 refit (backtest 2025 wk 3-18 + 2026 wk 1-5, ~330 games): simulated passing was too
# right-skewed and ~12% of targets went to players who then didn't play, so yardage medians
# ran low (actual went over the model's median 59-62% of the time). After the refit:
# rec yds 51%, pass yds 51%, rush att 50%; vs FanDuel lines the average gap went from
# -17.7 to +0.3 (QB pass yds) and -8.5 to -2.5 (rec yds).
ROLE_CV = 0.25       # a receiver's target share swings ~25% game to game
CARRY_CV = 0.35      # carry shares swing more (game script, hot hand, goal-line work)
QB_ATT_CV = 0.21     # team pass attempts per game
YPR_CV = 0.70        # yards per catch vary play to play (0.95 made yardage medians too low; refit Oct 2026)
YPC_SD = 6.5         # rushing yards per carry, standard deviation
TGT_GAMMA = 1.0      # >1 concentrates targets on main receivers (fit on 2025, see notes)
CAR_GAMMA = 1.0      # same for carries
TD_FLATTEN = 0.85    # <1 spreads touchdowns more evenly between teammates (fit on 2025: best Brier, top-end less overconfident)
PASS_EFF_BETA = 0.15 # links passing efficiency to the simulated score (refit Oct 2026)
RUSH_EFF_BETA = 0.05 # links rushing efficiency to the simulated score
PASS_CAL = 1.04      # level calibration of simulated passing yards (fit on 2025-26 backtest)
ATT_NOISE = "normal" # game-to-game noise in pass attempts: "normal" or "gamma" (old: right-skewed)
EFF_FORM = "linear"  # how passing efficiency follows the score: "linear" (symmetric) or "exp" (old)


class SimProps:
    """Player stat lines simulated inside every simulated game, internally consistent:
    the QB's completions, passing yards and passing TDs are the sum of what his receivers
    caught in that same simulated game, and every stat is tied to that game's score.

    .means    average of every stat across all simulations
    .medians  middle outcome (sportsbook lines sit near this)
    .draws    {(player, market): array of simulated values}
    .prob_over(player, market, line)
    .joint_prob(legs)   same-game combos, e.g.
        [("Jalen Hurts", "pass_yds", "over", 220.5), ("DeVonta Smith", "rec_yds", "over", 60.5),
         ("PHI", "win"), ("GAME", "total", "over", 44.5), ("PHI", "spread", -2.5),
         ("DeVonta Smith", "anytime_td")]"""

    def __init__(self, means, medians, draws, home, away, home_scores, away_scores, team=None):
        self.means, self.medians, self.draws = means, medians, draws
        self.team = team or {}   # team totals per simulation (pass/rush yards, attempts, points)
        self.home, self.away = home, away
        self.home_scores, self.away_scores = np.asarray(home_scores), np.asarray(away_scores)
        self._names = {}
        for (name, _mk) in draws:
            self._names[_norm(name)] = name
            self._names[ns._short(name)] = name

    def _name(self, player):
        k = _norm(player)
        hit = self._names.get(k) or self._names.get(str(player).lower())
        if hit:
            return hit
        # nickname / spelling differences between sportsbooks and rosters ("Kenneth" vs "Kenny"):
        # same last name and first initial, and only one such player in this game
        parts = k.split()
        if len(parts) >= 2:
            cands = {v for kk, v in self._names.items()
                     if kk.split()[-1:] == parts[-1:] and kk[:1] == k[:1]}
            if len(cands) == 1:
                return cands.pop()
        return player

    def prob_over(self, player, market, line):
        d = self.draws.get((self._name(player), market))
        if d is None:
            return np.nan
        return float((d >= 1).mean()) if market == "anytime_td" else float((d > line).mean())

    def distribution(self, player, market, qs=(0.1, 0.25, 0.5, 0.75, 0.9)):
        d = self.draws.get((self._name(player), market))
        return None if d is None else pd.Series(np.quantile(d, qs), index=[f"{int(q * 100)}th" for q in qs])

    def _mask(self, leg):
        h, a = self.home_scores, self.away_scores
        name, market = leg[0], leg[1]
        if market == "win":
            return (h > a) if name == self.home else (a > h)
        if market == "spread":            # ("PHI", "spread", -2.5): PHI covers -2.5
            margin = (h - a) if name == self.home else (a - h)
            return margin + float(leg[2]) > 0
        if market == "total":             # ("GAME", "total", "over", 44.5)
            t = h + a
            return t > float(leg[3]) if leg[2] == "over" else t < float(leg[3])
        d = self.draws.get((self._name(name), market))
        if d is None:
            raise KeyError(f"No simulated {market} for {name}")
        if market == "anytime_td":
            return d >= 1
        side, line = leg[2], float(leg[3])
        return d > line if side == "over" else d < line

    def joint_prob(self, legs, dk_price=None):
        """Probability every leg hits in the same simulated game, vs. treating legs as
        independent (what a naive parlay assumes). dk_price: DraftKings' same-game-parlay
        price in American odds (e.g. "+450"), to get expected value."""
        masks = [self._mask(l) for l in legs]
        singles = [float(mk.mean()) for mk in masks]
        joint = float(np.logical_and.reduce(masks).mean())
        indep = float(np.prod(singles))
        out = {"legs": [" ".join(str(x) for x in l) for l in legs], "leg_probs": [round(p, 3) for p in singles],
               "joint_prob": round(joint, 4), "if_independent": round(indep, 4),
               "correlation_lift": round(joint / indep, 2) if indep > 0 else np.nan,
               "fair_odds": _american_from_prob(joint)}
        if dk_price is not None:
            dec = _decimal_from_american(dk_price)
            out["dk_price"] = str(dk_price)
            out["ev_per_unit"] = round(joint * dec - 1, 3)
        return out


def _american_from_prob(p):
    p = min(max(float(p), 1e-4), 1 - 1e-4)
    return f"-{round(100 * p / (1 - p))}" if p >= 0.5 else f"+{round(100 * (1 - p) / p)}"


def _decimal_from_american(x):
    x = float(str(x).replace("+", ""))
    return 1 + (x / 100 if x > 0 else 100 / -x)


def _qb_rates(P, S, team, t_inputs):
    """Starting QB's completion % and yards/attempt, and how they compare with the
    team's recent passing (receivers' rates were earned with that)."""
    cand, qb = S["cand"], S["qb_id"]
    t_ypa = P["team_ypa"].get(team, P["lg_ypa"])
    t_comp = P["team_comp"].get(team, P["lg_comp"])
    if qb is None:
        return t_comp, t_ypa, 1.0, 1.0
    r = cand.loc[qb]
    if str(qb).startswith("generic_qb_") or float(r.get("w_attempts", 0) or 0) < 50:
        # little or no passing on record: use the game model's QB rating vs the team's
        # usual QB (it knows first-time starters play far below a typical backup)
        d = float(t_inputs["notes"].get("qb_vs_team_norm_epa", 0.0) or 0.0)
        ypa, comp = t_ypa + P["qb_slope_ypa"] * d, t_comp + P["qb_slope_comp"] * d
    else:
        ypa, comp = float(r.ypa), float(r.comp)
    ypa, comp = float(np.clip(ypa, 4.5, 9.5)), float(np.clip(comp, 0.45, 0.75))
    return comp, ypa, float(np.clip(comp / t_comp, 0.8, 1.2)), float(np.clip(ypa / t_ypa, 0.7, 1.35))


def simulate_props(m, P, home, away, sim=None, ctx=None, out_players=None, n_sims=10000, seed=7, **sim_kw):
    """Play out every player's stats inside each simulated game (the same scores the game
    simulation produced), then summarize. Game predictions are not changed."""
    if sim is None:
        sim = ns.simulate_game(m, home, away, n_sims, seed=3, ctx=ctx, out_players=out_players, **sim_kw)
    rng = np.random.default_rng(seed)
    base = project_game(m, P, home, away, ctx=ctx, sim=sim, out_players=out_players)
    keep = set(zip(base.team, base.player_id))
    means, meds, draws, team_tot = [], [], {}, {}
    hs, as_ = np.asarray(sim["home_scores"], float), np.asarray(sim["away_scores"], float)
    for side, team, opp, own, oth in (("home", home, away, hs, as_), ("away", away, home, as_, hs)):
        S = _team_setup(m, P, team, opp, sim["inputs"][side], ctx, out_players)
        n = len(own)
        vol, qb = S["vol"], S["qb_id"]
        cand = S["cand"]
        cand = cand[(cand.avail > 0) & ((cand.position != "QB") | (cand.index == qb))].copy()
        J = len(cand)
        ids = list(cand.index)
        z = (own - own.mean()) / max(own.std(), 1e-6)   # how good this simulated game went

        # --- volume
        base_rate = vol.att / (vol.att + vol.car)
        prate = np.clip(base_rate + P["script_slope_final"] * (own - oth), 0.30, 0.80)
        plays = (vol.att + vol.car) * np.exp(rng.normal(0, 0.09, n))
        # starting QB throws ~97% of team attempts (trick plays, garbage-time backups)
        att_noise = (np.clip(rng.normal(1, QB_ATT_CV, n), 0.3, 1.9) if ATT_NOISE == "normal"
                     else rng.gamma(1 / QB_ATT_CV ** 2, QB_ATT_CV ** 2, n))
        pa = np.maximum(np.round(plays * prate * 0.97 * att_noise), 0).astype(int)
        carries_team = rng.poisson(plays * (1 - prate))
        targets_team = rng.binomial(pa, min(vol.tgt / max(vol.att, 1), 1.0))

        # --- targets -> receptions -> yards, per receiver (sums equal the team's passing)
        comp, ypa, f_c, f_y = _qb_rates(P, S, team, sim["inputs"][side])
        tsh = np.where(cand.position.to_numpy() == "QB", 0.0, cand.tgt_share.to_numpy()) ** TGT_GAMMA
        role = rng.gamma(1 / ROLE_CV ** 2, ROLE_CV ** 2, (n, J))
        pt = tsh * role
        pt = pt / np.maximum(pt.sum(1, keepdims=True), 1e-12)
        targets = rng.multinomial(targets_team, pt) if tsh.sum() > 0 else np.zeros((n, J), int)
        # Team passing is anchored to the QB (his completion % and yards/attempt vs this
        # defense, which projects better than summing receivers' noisier rates); receivers'
        # own rates decide how it's split among them.
        tgt_rate = min(vol.tgt / max(vol.att, 1), 1.0)
        p_mean = tsh / max(tsh.sum(), 1e-12)
        raw_catch = np.clip(cand.catch.to_numpy() * S["d_cr"], 0.05, 0.95)
        raw_ypt = np.maximum(cand.ypt.to_numpy() * S["d_ypt"], 0.5)
        want_comp = comp * S["d_cr"] / tgt_rate                     # catches per target
        want_ypt = ypa * S["d_ypa"] * PASS_CAL / tgt_rate           # yards per target
        c_scale = want_comp / max((p_mean * raw_catch).sum(), 1e-6)
        y_scale = want_ypt / max((p_mean * raw_ypt).sum(), 1e-6)
        catch = np.clip(raw_catch * c_scale, 0.05, 0.95)
        rec = rng.binomial(targets, catch)
        eff_p = (np.exp(PASS_EFF_BETA * z - PASS_EFF_BETA ** 2 / 2) if EFF_FORM == "exp"
                 else np.clip(1 + PASS_EFF_BETA * z, 0.3, None))[:, None]
        ypr = np.maximum(raw_ypt * y_scale / catch, 0.5)
        k = 1 / YPR_CV ** 2
        rec_yds = np.where(rec > 0, rng.gamma(np.maximum(rec, 1) * k, ypr * eff_p / k), 0.0)

        # --- carries -> rushing yards
        csh = cand.car_share.to_numpy() ** CAR_GAMMA
        crole = rng.gamma(1 / CARRY_CV ** 2, CARRY_CV ** 2, (n, J))
        pc = csh * crole
        pc = pc / np.maximum(pc.sum(1, keepdims=True), 1e-12)
        carries = rng.multinomial(carries_team, pc) if csh.sum() > 0 else np.zeros((n, J), int)
        eff_r = np.exp(RUSH_EFF_BETA * z - RUSH_EFF_BETA ** 2 / 2)[:, None]
        ypc = cand.ypc.to_numpy() * S["d_ypc"]
        rush_yds = carries * ypc * eff_r + np.sqrt(carries) * YPC_SD * rng.standard_normal((n, J))

        # --- touchdowns: from this simulated score, split by who got the ball
        tds = np.floor(own * P["td_per_pt"] + rng.random(n)).astype(int)
        pass_tds_team = rng.binomial(tds, P["pass_td_share"].get(team, P["lg_pass_td_share"]))
        rush_tds_team = tds - pass_tds_team
        w_rec = (targets * (cand.rec_td_rate.to_numpy() / P["lg_rec_td_per_tgt"])) ** TD_FLATTEN
        prt = w_rec / np.maximum(w_rec.sum(1, keepdims=True), 1e-12)
        prt[w_rec.sum(1) == 0] = 1.0 / max(J, 1)
        rec_td = rng.multinomial(pass_tds_team, prt)
        rzr = cand.rz_ratio.to_numpy() if (USE_RZ_SHARE and "rz_ratio" in cand) else np.ones(J)
        cw = (carries.astype(float) * rzr) ** TD_FLATTEN   # goal-line backs get more of the rushing TDs
        prr = cw / np.maximum(cw.sum(1, keepdims=True), 1e-12)
        prr[carries.sum(1) == 0] = 1.0 / max(J, 1)
        rush_td = rng.multinomial(rush_tds_team, prr)
        any_td = rec_td + rush_td

        team_tot[team] = {"points": own, "pass_att": pa, "completions": rec.sum(1), "pass_yds": rec_yds.sum(1),
                          "carries": carries.sum(1), "rush_yds": rush_yds.sum(1), "pass_tds": rec_td.sum(1),
                          "rush_tds": rush_td.sum(1), "listed_tgt_share": None}

        # --- QB line = what his receivers did
        qb_cols = {}
        if qb is not None and qb in ids:
            qb_cols = {"pass_att": pa, "completions": rec.sum(1), "pass_yds": rec_yds.sum(1),
                       "pass_tds": rec_td.sum(1)}

        for j, pid in enumerate(ids):
            r = cand.loc[pid]
            if (team, pid) not in keep and pid != qb:
                continue
            stats_j = {"rush_att": carries[:, j], "rush_yds": rush_yds[:, j], "targets": targets[:, j],
                       "receptions": rec[:, j], "rec_yds": rec_yds[:, j], "anytime_td": any_td[:, j]}
            if pid == qb:
                stats_j.update(qb_cols)
            row = {"player_id": pid, "player": r.player_display_name, "pos": r.position, "team": team, "opp": opp}
            mean_row, med_row = dict(row), dict(row)
            for mk, arr in stats_j.items():
                draws[(r.player_display_name, mk)] = arr
                if mk == "anytime_td":
                    mean_row[mk] = med_row[mk] = float((arr >= 1).mean())
                else:
                    mean_row[mk], med_row[mk] = float(arr.mean()), float(np.median(arr))
            mean_row["exp_tds"] = float(any_td[:, j].mean())
            means.append(mean_row)
            meds.append(med_row)
    mdf, ddf = pd.DataFrame(means), pd.DataFrame(meds)
    for d in (mdf, ddf):
        d["home"], d["away"] = home, away
    return SimProps(mdf, ddf, draws, home, away, hs, as_, team_tot)


# ------------------------------------------------------------------ backtest
def naive_baseline(stats, as_of, player_ids):
    """What a simple average of each player's last 8 games would have said."""
    s = stats[(stats.gameday < as_of) & stats.player_id.isin(player_ids)]
    s = s.sort_values("gameday").groupby("player_id").tail(8)
    b = s.groupby("player_id")[list(MARKETS.values()) + ["tds"]].mean()
    b["anytime_td_rate"] = s.assign(any_td=(s.tds > 0).astype(float)).groupby("player_id").any_td.mean()
    return b


def backtest_props(games, prep, injuries, rosters, stats, season, weeks, n_sims=3000):
    rows = []
    g = games[(games.season == season) & games.result.notna()]
    for wk in weeks:
        slate = g[g.week == wk]
        if slate.empty:
            continue
        m = ns.fit_model(prep, slate.gameday.min())
        P = fit_props(m, stats)
        ctx = ns.build_context(season, wk, rosters, injuries)
        for _, x in slate.iterrows():
            pj = project_game(m, P, x.home_team, x.away_team, ctx=ctx, n_sims=n_sims,
                              home_qb=x.home_qb_id, away_qb=x.away_qb_id,
                              neutral=x.location == "Neutral")
            pj["game_id"], pj["week"], pj["season"] = x.game_id, wk, season
            rows.append(pj)
    pj = pd.concat(rows, ignore_index=True)
    act = stats[stats.game_id.isin(pj.game_id)].set_index(["game_id", "player_id"])
    act = act[list(MARKETS.values()) + ["tds"]].add_prefix("act_")
    pj = pj.join(act, on=["game_id", "player_id"])
    pj = pj[pj.act_tds.notna()].copy()  # player actually played (props are void otherwise)
    base = []
    for wk, d in pj.groupby("week"):
        as_of = games[(games.season == season) & (games.week == wk)].gameday.min()
        b = naive_baseline(stats, as_of, d.player_id.unique())
        base.append(d[["game_id", "player_id"]].join(b, on="player_id"))
    b = pd.concat(base).set_index(["game_id", "player_id"])
    b.columns = [f"naive_{c}" for c in b.columns]
    return pj.join(b, on=["game_id", "player_id"])


def grade_props(pj):
    """Model vs naive last-8-game average, per market: average miss, bias, correlation."""
    out = {}
    for col, stat in MARKETS.items():
        stat_a = f"act_{stat}"
        d = pj[pj[col].notna() & pj[stat_a].notna() & pj[f"naive_{stat}"].notna()]
        if col in ("pass_yds", "pass_att", "completions", "pass_tds"):
            d = d[d.pos == "QB"]
        elif col in ("rec_yds", "receptions"):
            d = d[d.targets >= 2]
        else:
            d = d[d.rush_att >= 4]
        if len(d) < 20:
            continue
        out[col] = {
            "players_graded": len(d),
            "model_avg_miss": (d[col] - d[stat_a]).abs().mean(),
            "naive_avg_miss": (d[f"naive_{stat}"] - d[stat_a]).abs().mean(),
            "model_bias": (d[col] - d[stat_a]).mean(),
            "model_corr": d[col].corr(d[stat_a]),
            "naive_corr": d[f"naive_{stat}"].corr(d[stat_a]),
        }
    d = pj[pj.anytime_td.notna() & pj.naive_anytime_td_rate.notna()]
    y = (d.act_tds > 0).astype(float)
    out["anytime_td"] = {
        "players_graded": len(d),
        "model_avg_miss": ((d.anytime_td - y) ** 2).mean(),   # Brier score
        "naive_avg_miss": ((d.naive_anytime_td_rate - y) ** 2).mean(),
        "model_bias": d.anytime_td.mean() - y.mean(),
        "model_corr": d.anytime_td.corr(y), "naive_corr": d.naive_anytime_td_rate.corr(y),
    }
    return pd.DataFrame(out).T


# ------------------------------------------------------------------ probabilities
# Spread of actual results around a projection: sd = k * sqrt(projection), fitted on 2025.
SD_K = {"pass_yds": 5.15, "pass_att": 1.573, "completions": 1.346, "pass_tds": 0.926,
        "rush_yds": 4.596, "rush_att": 1.501, "rec_yds": 4.779, "receptions": 1.134}


def prob_over(proj, line, market):
    """Model probability the stat goes over `line` (normal approximation)."""
    from math import erf, sqrt
    sd = SD_K[market] * np.sqrt(np.maximum(proj, 0.5))
    z = (np.asarray(line, float) - np.asarray(proj, float)) / sd
    return 0.5 * (1 - np.vectorize(lambda v: erf(v / sqrt(2)))(z))


def project_week(m, P, season, week, ctx=None, n_sims=4000):
    """Prop projections for every game in a week (game predictions are untouched)."""
    g = m["games"]
    rows = []
    for _, x in g[(g.season == season) & (g.week == week)].iterrows():
        pj = project_game(m, P, x.home_team, x.away_team, ctx=ctx, n_sims=n_sims,
                          neutral=x.location == "Neutral")
        pj["season"], pj["week"], pj["game_id"] = season, week, x.game_id
        rows.append(pj)
    return pd.concat(rows, ignore_index=True)


# ------------------------------------------------------------------ vs DraftKings
TEAM_NAMES = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL", "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR", "Chicago Bears": "CHI", "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL", "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX", "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC", "Los Angeles Rams": "LA", "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN", "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT", "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB", "Tennessee Titans": "TEN", "Washington Commanders": "WAS"}


def _norm(name):
    import re
    n = re.sub(r"[^a-z ]", "", str(name).lower().replace(".", " ").replace("-", " "))
    return " ".join(w for w in n.split() if w not in ("jr", "sr", "ii", "iii", "iv", "v"))


def load_dk_lines(path):
    """Read a CSV written by fetch_dk_props.py (one row per outcome) and return one row
    per player/market/line with over and under prices (decimal odds)."""
    d = pd.read_csv(path)
    d["home"] = d.home_team.map(TEAM_NAMES)
    d["away"] = d.away_team.map(TEAM_NAMES)
    d["market"] = d.market.map(ODDS_API_MARKETS)
    d = d[d.market.notna()].copy()
    d["key"] = d.player.map(_norm)
    keys = ["season", "week", "home", "away", "market", "key", "player"]
    yn = d[d.market == "anytime_td"]
    ou = d[d.market != "anytime_td"]
    ou = ou.pivot_table(index=keys + ["point"], columns="name", values="price", aggfunc="first").reset_index()
    ou = ou.rename(columns={"Over": "over_price", "Under": "under_price", "point": "line"})
    yes = yn[yn.name.str.lower() == "yes"][keys + ["price"]].rename(columns={"price": "yes_price"})
    return ou, yes


def auto_dk_lines(home=None, away=None, path="dk_props_current.csv", max_age_hours=6, refresh=False):
    """DraftKings current lines, downloaded automatically when missing or stale.
    With home/away, only that game is checked and fetched (~6 credits); without them,
    every game in the next 8 days (~100 credits). Returns (over/under lines, anytime-TD lines)."""
    import os
    import datetime as _dt
    import fetch_dk_props as fdk
    need = refresh or not os.path.exists(path)
    if not need:
        d = pd.read_csv(path)
        if home and away:
            d = d[(d.home_team == fdk.TEAM_NAMES.get(home)) & (d.away_team == fdk.TEAM_NAMES.get(away))]
        if d.empty:
            need = True
        else:
            snap = pd.to_datetime(d.snapshot, utc=True).max()
            age = (pd.Timestamp.now(tz="UTC") - snap).total_seconds() / 3600
            need = age > max_age_hours
    if need:
        try:
            fdk.fetch_current(out=path, teams=[home, away] if home and away else None)
        except fdk.OddsAPIError as e:
            print(f"Couldn't pull DraftKings lines: {e}")
            if not os.path.exists(path):
                return None, None
    if not os.path.exists(path):
        return None, None
    ou, yes = load_dk_lines(path)
    d = pd.read_csv(path)
    if len(d):
        snap = pd.to_datetime(d.snapshot, utc=True).max().tz_convert("America/Phoenix")
        print(f"DraftKings lines as of {snap:%a %b %d, %I:%M %p} Arizona time")
    return ou, yes


def dk_line(dk_ou, dk_yes, player, market):
    """A player's current DraftKings line and prices for one market (main line)."""
    if dk_ou is None:
        return None
    key = _norm(player)
    if market == "anytime_td":
        r = dk_yes[dk_yes.key == key]
        return None if r.empty else {"line": None, "over_price": float(r.yes_price.iloc[0]), "under_price": None}
    r = dk_ou[(dk_ou.market == market) & (dk_ou.key == key)].dropna(subset=["over_price", "under_price"])
    if r.empty:
        return None
    r = r.assign(gap=(r.over_price - r.under_price).abs()).sort_values("gap").iloc[0]  # main line
    return {"line": float(r.line), "over_price": float(r.over_price), "under_price": float(r.under_price)}


def prop_check(game_props, dk_ou, dk_yes, player, market, line=None, sims=None):
    """Model vs DraftKings for one prop: projection, line, probabilities, and expected value.
    Pass sims=<SimProps> to count probabilities straight from the simulations."""
    row = game_props[game_props.player.map(_norm) == _norm(player)]
    if row.empty:
        return f"{player} isn't in this game's projections (check spelling or team)."
    proj = float(row[market].iloc[0])
    dk = dk_line(dk_ou, dk_yes, player, market)
    if market == "anytime_td":
        if not dk:
            return f"{player}: model anytime TD {proj:.1%} (no DraftKings price found)"
        ev = proj * dk["over_price"] - 1
        return (f"{player}: model anytime TD {proj:.1%} | DraftKings {_american(dk['over_price'])} "
                f"(implied {1 / dk['over_price']:.1%}) | expected value {ev:+.1%} per $1")
    src = "your line"
    if line is None:
        if not dk:
            return f"{player}: projection {proj:.1f} (no DraftKings line found; pass line= to test one)"
        line, src = dk["line"], "DraftKings line"
    name = row.player.iloc[0]
    if sims is not None and (name, market) in sims.draws:
        p = sims.prob_over(name, market, line)
        how = "counted from simulations"
    else:
        p = float(prob_over(proj, line, market))
        how = "normal approximation"
    out = (f"{player} {market}: projection {proj:.1f} vs {src} {line} -> P(over) {p:.1%}, "
           f"P(under) {1 - p:.1%} ({how})")
    if dk and line == dk["line"]:
        ev_o, ev_u = p * dk["over_price"] - 1, (1 - p) * dk["under_price"] - 1
        out += (f"\n  DraftKings prices: over {_american(dk['over_price'])}, under {_american(dk['under_price'])}"
                f" | expected value per $1: over {ev_o:+.1%}, under {ev_u:+.1%}")
    return out


def _american(dec):
    return f"+{round((dec - 1) * 100)}" if dec >= 2 else f"{round(-100 / (dec - 1))}"


def compare_to_dk(proj, dk_ou, dk_yes):
    """This week's model projections next to DraftKings lines, with the model's
    probability and expected value for the side it prefers."""
    pj = proj.copy()
    pj["key"] = pj.player.map(_norm)
    rows = []
    for market in dk_ou.market.unique():
        l = dk_ou[dk_ou.market == market].merge(pj[["season", "week", "home", "away", "key", "team", market]],
                                               on=["season", "week", "home", "away", "key"])
        l = l[l[market].notna()]
        if l.empty:
            continue
        po = prob_over(l[market], l.line, market)
        l = l.assign(model=l[market].round(1), p_over=po.round(3))
        l["pick"] = np.where(po > 0.5, "over", "under")
        p = np.where(po > 0.5, po, 1 - po)
        price = np.where(po > 0.5, l.over_price, l.under_price)
        l["model_prob"] = p.round(3)
        l["ev_per_unit"] = (p * price - 1).round(3)
        rows.append(l[["player", "team", "market", "line", "model", "pick", "model_prob", "over_price",
                       "under_price", "ev_per_unit"]])
    if len(dk_yes):
        t = dk_yes.merge(pj[["season", "week", "home", "away", "key", "team", "anytime_td"]],
                         on=["season", "week", "home", "away", "key"])
        t = t.assign(market="anytime_td", line=np.nan, model=t.anytime_td.round(3), pick="yes",
                     model_prob=t.anytime_td.round(3), over_price=t.yes_price, under_price=np.nan,
                     ev_per_unit=(t.anytime_td * t.yes_price - 1).round(3))
        rows.append(t[["player", "team", "market", "line", "model", "pick", "model_prob", "over_price",
                       "under_price", "ev_per_unit"]])
    return pd.concat(rows, ignore_index=True).sort_values("ev_per_unit", ascending=False) if rows else pd.DataFrame()


def grade_vs_dk(pj, dk_ou, dk_yes):
    """Backtest: model vs DraftKings on the same historical props. Whose number was
    closer to the result, and how the model's picks did at DK's actual prices."""
    pj = pj.copy()
    pj["key"] = pj.player.map(_norm)
    res = {}
    for market in dk_ou.market.unique():
        stat = "act_" + MARKETS[market]
        l = dk_ou[dk_ou.market == market].merge(
            pj[["season", "week", "home", "away", "key", market, stat]],
            on=["season", "week", "home", "away", "key"])
        l = l[l[stat].notna() & l[market].notna() & l.over_price.notna() & l.under_price.notna()]
        if l.empty:
            continue
        po = prob_over(l[market], l.line, market)
        pick_over = po > 0.5
        push = l[stat] == l.line
        won = (pick_over == (l[stat] > l.line)) & ~push
        price = np.where(pick_over, l.over_price, l.under_price)
        n = int((~push).sum())
        profit = np.where(push, 0.0, np.where(won, price - 1, -1.0))
        edge = np.where(pick_over, po, 1 - po) * price - 1 > 0.05  # model sees 5%+ value
        res[market] = {
            "props": len(l),
            "model_avg_miss": (l[market] - l[stat]).abs().mean(),
            "dk_line_avg_miss": (l.line - l[stat]).abs().mean(),
            "model_record": f"{int(won.sum())}-{n - int(won.sum())}",
            "model_win_pct": won.sum() / max(n, 1),
            "roi_all_picks": profit.sum() / max(n, 1),
            "roi_5pct_edge_picks": profit[edge].mean() if edge.any() else np.nan,
            "n_5pct_edge_picks": int(edge.sum()),
        }
    if len(dk_yes):
        t = dk_yes.merge(pj[["season", "week", "home", "away", "key", "anytime_td", "act_tds"]],
                         on=["season", "week", "home", "away", "key"])
        t = t[t.act_tds.notna()]
        if len(t):
            y = (t.act_tds > 0).to_numpy(float)
            imp = 1 / t.yes_price
            val = t.anytime_td * t.yes_price - 1 > 0.05
            res["anytime_td"] = {
                "props": len(t),
                "model_avg_miss": ((t.anytime_td - y) ** 2).mean(),  # Brier score
                "dk_line_avg_miss": ((imp - y) ** 2).mean(),        # vig-inclusive implied prob
                "model_record": "", "model_win_pct": np.nan, "roi_all_picks": np.nan,
                "roi_5pct_edge_picks": np.where(y[val] == 1, t.yes_price[val] - 1, -1).mean() if val.any() else np.nan,
                "n_5pct_edge_picks": int(val.sum()),
            }
    return pd.DataFrame(res).T
