"""
Player prop projections on top of the NFL game simulator (v5).

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

import nfl_game_sim_v5 as ns

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
    pr = _recent(ps, "player_id", 8, 3)
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
            ("position", "w_receiving_yards", "w_targets", 40, "ypt"),
            ("position", "w_receptions", "w_targets", 40, "catch"),
            ("position", "w_rushing_yards", "w_carries", 50, "ypc"),
            ("position", "w_passing_yards", "w_attempts", 150, "ypa"),
            ("position", "w_completions", "w_attempts", 150, "comp")):
        pos_mean = pl.groupby(pos_col).apply(
            lambda d: d[num].sum() / max(d[den].sum(), 1e-9), include_groups=False)
        prior = pl[pos_col].map(pos_mean).fillna(pl[num].sum() / max(pl[den].sum(), 1e-9))
        pl[name] = (pl[num] + prior * k) / (pl[den] + k)
    # receiving TDs per target, heavily shrunk to league (TD rates barely repeat)
    P["lg_rec_td_per_tgt"] = team.ptd.sum() / max(team.tgt.sum(), 1)
    P["lg_rush_td_per_car"] = team.rtd.sum() / max(team.car.sum(), 1)
    pl["rec_td_rate"] = (pl.w_receiving_tds + P["lg_rec_td_per_tgt"] * 40) / (pl.w_targets + 40)
    P["players"] = pl
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
    tot_car = (cand.car_share * cand.avail).sum()
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
        rush_td = rush_tds * (r.car_share / max(tot_car, 1e-9))
        for k, f in CAL.items():
            if k in rec:
                rec[k] *= f
        rec["exp_tds"] = rec_td + rush_td
        rec["anytime_td"] = 1 - np.exp(-(rec_td + rush_td))
        if (rec.get("pass_att", 0) > 5) or t >= 2.0 or c >= 4.0:
            out.append(rec)
    return out


# ------------------------------------------------------------------ simulated props
# Game-to-game noise the averages can't see, tuned so simulated spreads match real ones.
ROLE_CV = 0.25      # a player's share of targets swings ~25% game to game
CARRY_CV = 0.35     # carry shares swing more (game script, hot hand, goal-line work)
QB_ATT_CV = 0.21    # QB pass attempts per game
TD_CAL = 1.15       # simulated TD chances ran ~15% low for players who actually played (2025)
YPR_CV = 0.95       # yards per catch / per completion vary a lot play to play
YPC_SD = 6.5        # rushing yards per carry, standard deviation


class SimProps:
    """Player stat lines simulated inside every simulated game.
    .means   average of every stat across all simulations (same columns as project_game)
    .medians middle outcome (sportsbook lines sit near this)
    .draws   {(player, market): array of simulated values}"""

    def __init__(self, means, medians, draws):
        self.means, self.medians, self.draws = means, medians, draws

    def prob_over(self, player, market, line):
        d = self.draws.get((player, market))
        if d is None:
            return np.nan
        if market == "anytime_td":
            return float((d >= 1).mean())
        return float((d > line).mean())

    def distribution(self, player, market, qs=(0.1, 0.25, 0.5, 0.75, 0.9)):
        d = self.draws.get((player, market))
        return None if d is None else pd.Series(np.quantile(d, qs), index=[f"{int(q * 100)}th" for q in qs])


def simulate_props(m, P, home, away, sim=None, ctx=None, out_players=None, n_sims=10000, seed=7, **sim_kw):
    """Play out every player's stats in each simulated game (same scores the game
    simulation produced), then average. Game predictions are not changed."""
    if sim is None:
        sim = ns.simulate_game(m, home, away, n_sims, seed=3, ctx=ctx, out_players=out_players, **sim_kw)
    rng = np.random.default_rng(seed)
    base = project_game(m, P, home, away, ctx=ctx, sim=sim, out_players=out_players)
    keep = set(zip(base.team, base.player_id))
    means, meds, draws = [], [], {}
    hs, as_ = np.asarray(sim["home_scores"], float), np.asarray(sim["away_scores"], float)
    for side, team, opp, own, oth in (("home", home, away, hs, as_), ("away", away, home, as_, hs)):
        S = _team_setup(m, P, team, opp, sim["inputs"][side], ctx, out_players)
        n = len(own)
        vol, cand = S["vol"], S["cand"]
        cand = cand[(cand.avail > 0) & ((cand.position != "QB") | (cand.index == S["qb_id"]))]
        base_rate = vol.att / (vol.att + vol.car)
        prate = np.clip(base_rate + P["script_slope_final"] * (own - oth), 0.30, 0.80)
        plays = (vol.att + vol.car) * np.exp(rng.normal(0, 0.09, n))
        att, car = plays * prate, plays * (1 - prate)
        tgt = vol.tgt / vol.att * att
        J = len(cand)
        role = rng.gamma(1 / ROLE_CV ** 2, ROLE_CV ** 2, (n, J))
        tgt_share = cand.tgt_share.to_numpy() * np.where(cand.position.to_numpy() == "QB", 0, 1)
        targets = rng.poisson(tgt[:, None] * tgt_share * role * CAL["targets"])
        catch = np.clip(cand.catch.to_numpy() * S["d_cr"], 0.05, 0.95)
        rec = rng.binomial(targets, catch)
        ypr = np.maximum(cand.ypt.to_numpy() * S["d_ypt"] / catch * CAL["rec_yds"] / CAL["receptions"], 0.5)
        k = 1 / YPR_CV ** 2
        rec_yds = np.where(rec > 0, rng.gamma(np.maximum(rec, 1) * k, ypr / k), 0.0)
        crole = rng.gamma(1 / CARRY_CV ** 2, CARRY_CV ** 2, (n, J))
        carries = rng.poisson(car[:, None] * cand.car_share.to_numpy() * crole * CAL["rush_att"])
        ypc = cand.ypc.to_numpy() * S["d_ypc"] * CAL["rush_yds"] / CAL["rush_att"]
        rush_yds = carries * ypc + np.sqrt(carries) * YPC_SD * rng.standard_normal((n, J))
        # touchdowns: team TDs from each simulated score, split by who got the ball
        # stochastic rounding keeps TDs consistent with the score (24 pts -> ~2-3 TDs)
        tds = np.floor(own * P["td_per_pt"] + rng.random(n))
        pass_tds = rng.binomial(tds.astype(int), P["pass_td_share"].get(team, P["lg_pass_td_share"]))
        rush_tds = tds - pass_tds
        w_rec = targets * (cand.rec_td_rate.to_numpy() / P["lg_rec_td_per_tgt"])
        rec_td = rng.poisson(TD_CAL * pass_tds[:, None] * w_rec / np.maximum(w_rec.sum(1, keepdims=True), 1e-9))
        rush_td = rng.poisson(TD_CAL * rush_tds[:, None] * carries / np.maximum(carries.sum(1, keepdims=True), 1e-9))
        any_td = rec_td + rush_td
        qb_cols = {}
        if S["qb_id"] is not None and S["qb_id"] in cand.index:
            r = cand.loc[S["qb_id"]]
            pa = np.round(att * 0.97 * rng.gamma(1 / QB_ATT_CV ** 2, QB_ATT_CV ** 2, n)).astype(int)
            comp = rng.binomial(pa, np.clip(r.comp * S["d_cr"], 0.3, 0.8))
            ypcmp = max(r.ypa * S["d_ypa"] / np.clip(r.comp * S["d_cr"], 0.3, 0.8), 1.0)
            pyds = np.where(comp > 0, rng.gamma(np.maximum(comp, 1) * k, ypcmp / k), 0.0)
            qb_cols = {"pass_att": pa, "completions": comp, "pass_yds": pyds, "pass_tds": pass_tds}
        for j, (pid, r) in enumerate(cand.iterrows()):
            if (team, pid) not in keep:
                continue
            stats_j = {"rush_att": carries[:, j], "rush_yds": rush_yds[:, j], "targets": targets[:, j],
                       "receptions": rec[:, j], "rec_yds": rec_yds[:, j], "anytime_td": any_td[:, j]}
            if pid == S["qb_id"]:
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
    return SimProps(mdf, ddf, draws)


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
