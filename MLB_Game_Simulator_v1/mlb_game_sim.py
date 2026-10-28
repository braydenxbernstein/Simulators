"""
MLB game simulator: plate-appearance Monte Carlo.

Every simulated game is played one plate appearance at a time. Each PA's outcome
(K, BB, HBP, 1B, 2B, 3B, HR, out in play) comes from the batter's and pitcher's
regressed platoon rates combined by the odds-ratio (log5) method, then adjusted for
the game environment. Baserunners advance probabilistically, starters get pulled on
pitch count, and bullpens are rebuilt for every game from who is actually rested.

Factors (F = fitted from data and shrunk by sample size, A = assumed constant)
  Batter talent         F  3 seasons of platoon splits, weighted toward this year, regressed per outcome
  Pitcher talent        F  same, vs left/right batters; switch hitters bat opposite the pitcher
  Platoon               F  league L/R split plus each player's own split, heavily shrunk
  Statcast luck         F  xwOBA vs wOBA moves hit rates toward expected (batters and pitchers)
  Park                  F  per-outcome park factors (HR, 2B, 3B, 1B, BB, K) from 2 seasons of boxscores
  Weather               F  temperature and wind blowing out/in -> HR and hit rates (roof closed = neutral)
  Home plate umpire     F  strikeout and walk rates in his games vs expected
  Home field            F  per-PA batting edge for the home team
  Travel / rest         F  cross-country trips on 0-1 days rest, day game after night game
  Team defense          F  outs above average converts to fewer hits on balls in play
  Catcher framing       A  framing runs -> strikeout and walk rates
  Times through order   A  starters get worse the 3rd and 4th time through
  Starter leash         F  each starter's recent pitch counts; short rest and postseason shorten it
  Pitch counts          F  each pitcher's pitches per PA, by PA outcome
  Bullpen               F  relievers used in the last 30 days, minus anyone who pitched back-to-back
                           days or threw 25+ pitches yesterday; best 3 rested arms close games late
  Baserunning           A  advancement odds on hits and outs, adjusted by Statcast sprint speed
  Double plays, sac flies, errors, wild pitches/steals   A
  Extra innings         -  automatic runner in the regular season, none in the postseason
  Walk-offs             -  game ends the moment the home team leads in the 9th or later
"""

import datetime as dt
import math
from collections import defaultdict

import numpy as np
import pandas as pd

import mlb_data as md

OUT = md.OUTCOMES                       # K BB HBP 1B 2B 3B HR OIP
K_, BB_, HBP_, S1, S2, S3, HR_, OIP_ = range(8)
HITS = [S1, S2, S3, HR_]
POSTSEASON = {"F", "D", "L", "W"}

# stabilization points (PA) for regression to the mean
STAB_BAT = np.array([60, 120, 240, 400, 900, 900, 170, 450], float)
STAB_PIT = np.array([70, 170, 400, 700, 1300, 1300, 900, 700], float)
SEASON_W = {0: 1.0, 1: 0.6, 2: 0.35}    # this season, last season, two seasons ago

# assumed constants
TTO = {  # multipliers by times through the order: (K, hits/BB)
    0: (1.04, 0.97), 1: (1.00, 1.00), 2: (0.95, 1.05), 3: (0.92, 1.08)}
PITCHES_BY_OUTCOME = np.array([4.8, 5.7, 3.4, 3.4, 3.5, 3.6, 3.5, 3.3])
FRAMING_PITCHES_PER_GAME = 31.0
LG_SPEED = 27.0
LINEAR_W = np.array([0, .69, .72, .89, 1.27, 1.62, 2.10, 0])   # wOBA-style weights, for quality ranking only


def _w(x):
    return np.asarray(x, float)


# ================================================================== fitting helpers
def _rate_table(df, stab, league, weights):
    """Per-player regressed overall rates + raw vs-hand rates.
    df: rows pid, vs, PA, outcomes, season_age."""
    df = df.copy()
    w = df.season_age.map(weights).fillna(0.0)
    for c in ["PA"] + OUT + ["GO", "AO", "GIDP", "pitches"]:
        df[c] = df[c] * w
    tot = df.groupby("pid")[["PA"] + OUT + ["GO", "AO", "GIDP", "pitches"]].sum()
    hand = df.groupby(["pid", "vs"])[["PA"] + OUT].sum()
    return tot, hand


def _regress(counts, pa, prior, stab):
    return (counts + stab * prior) / (pa + stab).reshape(-1, 1) if np.ndim(pa) else (counts + stab * prior) / (pa + stab)


def _norm(v):
    v = np.clip(v, 1e-6, None)
    return v / v.sum(axis=-1, keepdims=True)


def _poisson_irls(y, offset, X, iters=25, ridge=1e-3):
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        eta = offset + X @ b
        mu = np.exp(eta)
        z = eta - offset + (y - mu) / mu
        W = mu
        A = X.T @ (X * W[:, None]) + ridge * np.eye(X.shape[1])
        b = np.linalg.solve(A, X.T @ (W * z))
    return b


# ================================================================== model
class Model:
    """Everything fitted as of a date. Build with fit(asof)."""

    def describe(self):
        print(f"MLB model as of {self.asof} (season {self.season})")
        print(f"  batters: {len(self.bat)}  pitchers: {len(self.pit)}  parks: {len(self.park)}  umps: {len(self.ump)}")
        print(f"  league per PA: " + ", ".join(f"{o} {r:.3f}" for o, r in zip(OUT, self.lg_all)))
        print(f"  weather: HR {self.wx['hr_temp']*100:+.2f}%/degF, {self.wx['hr_wind']*100:+.2f}%/mph out; "
              f"hits {self.wx['hit_temp']*100:+.2f}%/degF, {self.wx['hit_wind']*100:+.2f}%/mph out")
        print(f"  home batting multipliers: " + ", ".join(f"{o} {m:.3f}" for o, m in zip(OUT, self.home_mult)))
        for k, v in self.travel.items():
            print(f"  {k}: batting x{v:.3f}")
        print(f"  postseason starter leash: x{getattr(self, 'post_leash', 0.93):.3f} (from {getattr(self, 'post_leash_n', 0)} playoff starts)")
        if hasattr(self, "post_vec"):
            print(f"  postseason environment ({self.post_n} games): " + ", ".join(f"{o} x{v:.3f}" for o, v in zip(OUT[:7], self.post_vec[:7])))
        if getattr(self, "level_vec", None) is not None:
            print("  recent-environment multipliers: " + ", ".join(f"{o} x{v:.3f}" for o, v in zip(OUT[:7], self.level_vec[:7])))


def fit(asof=None, season=None, savant_year=None, verbose=True):
    """Fit everything using only games before `asof`.
    savant_year: which year's Statcast leaderboards to use (defaults to the season when asof is
    today; for backtests pass the prior year so no future data leaks in)."""
    asof = md.to_date(asof)
    season = season or (asof.year if asof.month >= 3 else asof.year - 1)
    live = asof >= dt.date.today()
    if savant_year is None:
        savant_year = season if live else season - 1
    m = Model()
    m.asof, m.season, m.savant_year = asof, season, savant_year

    # ---------------- games and boxscores (this season + last season) before asof
    scheds, boxes = [], {}
    for s in (season - 1, season):
        sc = md.season_schedule(s)
        fin = md.final_games(sc)
        fin = fin[fin.date < asof]
        boxes.update(md.boxscores(fin.game_pk.tolist(), s))
        scheds.append(fin)
    sched = pd.concat(scheds, ignore_index=True)
    m.sched_hist = sched
    m.box = boxes
    gf = md.game_frame(sched, boxes)
    gf = gf[gf.b_plateAppearances > 0].copy()
    gf["K"] = gf.b_strikeOuts
    gf["BB"] = gf.b_baseOnBalls
    gf["HBP"] = gf.b_hitByPitch
    gf["2B"], gf["3B"], gf["HR"] = gf.b_doubles, gf.b_triples, gf.b_homeRuns
    gf["1B"] = gf.b_hits - gf["2B"] - gf["3B"] - gf["HR"]
    gf["PA"] = gf.b_plateAppearances
    gf["OIP"] = (gf.PA - gf[OUT[:-1]].sum(axis=1)).clip(lower=0)
    m.gf = gf

    # ---------------- player splits, 3 seasons
    if verbose:
        print("Loading platoon splits...")
    bat_rows, pit_rows = [], []
    for age in (0, 1, 2):
        yr = season - age
        b = md.splits(yr, "hitting", asof if age == 0 else None)
        p = md.splits(yr, "pitching", asof if age == 0 else None)
        b["season_age"], p["season_age"] = age, age
        bat_rows.append(b)
        pit_rows.append(p)
    bat_df = pd.concat(bat_rows, ignore_index=True)
    pit_df = pd.concat(pit_rows, ignore_index=True)
    ids = set(bat_df.pid) | set(pit_df.pid)
    for g in boxes.values():
        for side in g["teams"].values():
            ids.update(side["lineup"])
            ids.update(p["id"] for p in side["pitchers"])
    m.people = md.people(list(ids))

    # league rates by batter hand x pitcher hand (current + last season, from batter splits)
    cur = bat_df[bat_df.season_age <= 1].copy()
    cur["bats"] = cur.pid.map(lambda i: m.people.get(i, {}).get("bats", "R"))
    cur["eff"] = np.where(cur.bats == "S", np.where(cur.vs == "L", "R", "L"), cur.bats)
    lg = {}
    for B in "LR":
        for P in "LR":
            sub = cur[(cur.eff == B) & (cur.vs == P)]
            lg[(B, P)] = _norm(sub[OUT].sum().values / sub.PA.sum())
    m.lg = lg
    m.lg_all = _norm(cur[OUT].sum().values / cur.PA.sum())
    m.lg_bat_mix = {B: _norm(cur[cur.eff == B][OUT].sum().values / cur[cur.eff == B].PA.sum()) for B in "LR"}
    pc = pit_df[pit_df.season_age <= 1].copy()
    pc["throws"] = pc.pid.map(lambda i: m.people.get(i, {}).get("throws", "R"))
    m.lg_pit_mix = {T: _norm(pc[pc.throws == T][OUT].sum().values / pc[pc.throws == T].PA.sum()) for T in "LR"}
    m.lg_p_per_pa = pc.pitches.sum() / pc.PA.sum()
    m.lg_gb = pc.GO.sum() / (pc.GO.sum() + pc.AO.sum())

    # ---------------- Statcast (no leakage in backtests: savant_year defaults to prior season)
    sv = md.savant(savant_year)
    m.speed = dict(zip(sv["speed"].player_id, sv["speed"].sprint_speed)) if len(sv["speed"]) else {}
    xf_b, xf_p = {}, {}
    for key, out in (("xbat", xf_b), ("xpit", xf_p)):
        x = sv[key]
        if len(x):
            ratio = (x.est_woba / x.woba.replace(0, np.nan)).clip(0.8, 1.2).fillna(1.0)
            w = x.bip / (x.bip + 350.0)
            out.update(dict(zip(x.player_id, ratio ** w)))
    m.xf_bat, m.xf_pit = xf_b, xf_p
    oaa = sv["oaa"]
    games_by_team = sched[sched.date.map(lambda d: d.year) == savant_year]
    gcount = pd.concat([games_by_team.home_id, games_by_team.away_id]).value_counts()
    m.oaa_pg = {}
    if len(oaa):
        for r in oaa.itertuples():
            n = gcount.get(r.team_id, 0) or 162
            m.oaa_pg[r.team_id] = r.outs_above_average / max(n, 40)
    fr = sv["framing"]
    m.framing_rpg = dict(zip(fr.id, fr.rv_tot / fr.pitches.clip(lower=1) * FRAMING_PITCHES_PER_GAME)) if len(fr) else {}

    # ---------------- batters
    tot, hand = _rate_table(bat_df, STAB_BAT, m.lg_all, SEASON_W)
    m.bat = {}
    for pid, r in tot.iterrows():
        info = m.people.get(pid, {"bats": "R", "name": str(pid)})
        bats = info["bats"]
        effs = {"L": ("R" if bats == "S" else bats), "R": ("L" if bats == "S" else bats)}   # eff hand vs P
        prior_mix = _norm(np.mean([m.lg[(effs[P], P)] for P in "LR"], axis=0))
        overall = _norm(_regress(r[OUT].values, r.PA, prior_mix, STAB_BAT))
        xf = xf_b.get(pid, 1.0)
        rates = {}
        for P in "LR":
            B = effs[P]
            lg_ratio = m.lg[(B, P)] / prior_mix
            try:
                h = hand.loc[(pid, P)]
                obs, n = h[OUT].values, h.PA
            except KeyError:
                obs, n = np.zeros(8), 0.0
            base = overall * lg_ratio
            if n > 0:
                own = (obs + 1e-9) / n
                dev = (own / overall) / lg_ratio
                shrink = n / (n + 4 * STAB_BAT)
                dev = np.exp(np.log(np.clip(dev, 0.2, 5)) * shrink)
                base = base * dev
            base = base.copy()
            base[HITS] *= xf
            rates[P] = _norm(base)
        sp = m.speed.get(pid, LG_SPEED)
        m.bat[pid] = {"name": info["name"], "bats": bats, "vs": rates, "speed": sp if np.isfinite(sp) else LG_SPEED,
                      "PA": r.PA}

    # ---------------- pitchers
    tot, hand = _rate_table(pit_df, STAB_PIT, m.lg_all, SEASON_W)
    m.pit = {}
    for pid, r in tot.iterrows():
        info = m.people.get(pid, {"throws": "R", "name": str(pid)})
        T = info["throws"]
        mix = m.lg_pit_mix.get(T, m.lg_all)
        overall = _norm(_regress(r[OUT].values, r.PA, mix, STAB_PIT))
        xf = xf_p.get(pid, 1.0)
        rates = {}
        for B in "LR":
            lg_ratio = m.lg[(B, T)] / mix
            try:
                h = hand.loc[(pid, B)]
                obs, n = h[OUT].values, h.PA
            except KeyError:
                obs, n = np.zeros(8), 0.0
            base = overall * lg_ratio
            if n > 0:
                dev = ((obs + 1e-9) / n / overall) / lg_ratio
                shrink = n / (n + 4 * STAB_PIT)
                base = base * np.exp(np.log(np.clip(dev, 0.2, 5)) * shrink)
            base = base.copy()
            base[HITS] *= xf
            rates[B] = _norm(base)
        gb = (r.GO + 40 * m.lg_gb) / (r.GO + r.AO + 40)
        ppa = (r.pitches + 60 * m.lg_p_per_pa) / (r.PA + 60) if r.pitches > 0 else m.lg_p_per_pa
        m.pit[pid] = {"name": info["name"], "throws": T, "vs": rates, "gb": gb, "ppa": ppa, "PA": r.PA}

    _fit_environment(m, verbose)
    _fit_usage(m)
    if verbose:
        print(f"Fitted on {len(sched)} games before {asof}.")
    return m


def _expected(gf, rate_cols):
    """Expected outcome counts for each team-game from batting team and opponent pitching rates."""
    bat = gf.groupby("team_id")[OUT + ["PA"]].sum()
    pit = gf.groupby("opp_id")[OUT + ["PA"]].sum()
    lg = gf[OUT].sum().values / gf.PA.sum()
    br = (bat[OUT].values + 200 * lg) / (bat.PA.values[:, None] + 200)
    pr = (pit[OUT].values + 200 * lg) / (pit.PA.values[:, None] + 200)
    br = pd.DataFrame(br, index=bat.index, columns=OUT)
    pr = pd.DataFrame(pr, index=pit.index, columns=OUT)
    e = br.loc[gf.team_id].values * pr.loc[gf.opp_id].values / lg
    e = e / e.sum(axis=1, keepdims=True)
    return e * gf.PA.values[:, None], lg


def _fit_environment(m, verbose):
    gf = m.gf.copy()
    exp, lg = _expected(gf, OUT)
    obs = gf[OUT].values

    # park factors by outcome (k = prior pseudo-counts per outcome)
    kpark = np.array([400, 200, 40, 400, 150, 30, 80, 600], float)
    m.park = {}
    for v, idx in gf.groupby("venue_id").groups.items():
        o, e = obs[idx].sum(0), exp[idx].sum(0)
        m.park[v] = (o + kpark) / (e + kpark)
    # remove the park from expectations before fitting weather/ump/home
    pf = np.vstack([m.park.get(v, np.ones(8)) for v in gf.venue_id])
    exp_p = exp * pf

    # postseason environment: playoff PA outcomes vs what the same teams' rates predict (shrunk)
    post = gf.game_type.isin(list(POSTSEASON)).values
    kpost = np.array([600, 300, 60, 600, 200, 40, 120, 900], float)
    m.post_vec = (obs[post].sum(0) + kpost) / (exp_p[post].sum(0) + kpost) if post.any() else np.ones(8)
    m.post_n = int(post.sum() // 2)

    # home field per PA
    hm = gf.home.values
    kh = np.array([2000, 1000, 200, 2000, 800, 150, 500, 3000], float)
    oh, eh = obs[hm].sum(0), exp_p[hm].sum(0)
    oa, ea = obs[~hm].sum(0), exp_p[~hm].sum(0)
    m.home_mult = np.sqrt(((oh + kh) / (eh + kh)) / ((oa + kh) / (ea + kh)))

    # umpire K / BB
    kump = np.array([500, 250], float)
    m.ump = {}
    g2 = gf.dropna(subset=["hp_ump_id"])
    for u, idx in g2.groupby("hp_ump_id").groups.items():
        ii = gf.index.get_indexer(idx)
        o = obs[ii][:, [K_, BB_]].sum(0)
        e = exp_p[ii][:, [K_, BB_]].sum(0)
        m.ump[int(u)] = (o + kump) / (e + kump)

    # weather: Poisson regression of HR and hits on temperature and wind blowing out
    w = gf[gf.temp.notna() & ~gf.roof_closed.astype(bool)]
    ii = gf.index.get_indexer(w.index)
    X = np.column_stack([(w.temp.values - 72) / 10, w.wind_out.values / 10])
    b_hr = _poisson_irls(obs[ii][:, HR_], np.log(exp_p[ii][:, HR_]), X)
    hits = [S1, S2, S3]
    b_h = _poisson_irls(obs[ii][:, hits].sum(1), np.log(exp_p[ii][:, hits].sum(1)), X)
    m.wx = {"hr_temp": b_hr[0] / 10, "hr_wind": b_hr[1] / 10, "hit_temp": b_h[0] / 10, "hit_wind": b_h[1] / 10,
            "n_games": len(w)}
    # park factors already contain each park's usual weather, so weather is applied relative to it
    vm = w.groupby("venue_id")[["temp", "wind_out"]].mean()
    m.venue_wx = {v: (r.temp, r.wind_out) for v, r in vm.iterrows()}
    m.wx_default = (float(w.temp.mean()), float(w.wind_out.mean()))

    # travel and rest
    sc = m.sched_hist.sort_values("start")
    lon = {}
    rows = []
    last = {}
    for g in sc.itertuples():
        for side, tid in (("away", g.away_id), ("home", g.home_id)):
            prev = last.get(tid)
            flags = {"long_trip": False, "day_after_night": False}
            if prev is not None:
                days = (g.date - prev["date"]).days
                if days <= 1 and g.lon is not None and prev["lon"] is not None and abs((g.lon or 0) - (prev["lon"] or 0)) > 25:
                    flags["long_trip"] = True
                if days == 1 and prev["dn"] == "night" and g.day_night == "day":
                    flags["day_after_night"] = True
            rows.append({"game_pk": g.game_pk, "team_id": tid, **flags})
            last[tid] = {"date": g.date, "lon": g.lon, "dn": g.day_night}
    fl = pd.DataFrame(rows)
    gfl = gf.merge(fl, on=["game_pk", "team_id"], how="left")
    lw = LINEAR_W
    m.travel = {}
    for f in ("long_trip", "day_after_night"):
        sel = gfl[f].fillna(False).values.astype(bool)
        o = (obs[sel] * lw).sum()
        e = (exp_p[sel] * lw).sum()
        k = 1500.0
        m.travel[f] = (o + k) / (e + k)
    m.flags = fl


def _fit_usage(m):
    """Pitcher appearances and lineups from boxscores, for leashes, bullpens and projected lineups."""
    app, lus = [], []
    gmeta = m.sched_hist.set_index("game_pk")
    for pk, g in m.box.items():
        if pk not in gmeta.index:
            continue
        meta = gmeta.loc[pk]
        if isinstance(meta, pd.DataFrame):
            meta = meta.iloc[0]
        for side, opp in (("away", "home"), ("home", "away")):
            t = m.box[pk]["teams"][side]
            o = m.box[pk]["teams"][opp]
            for p in t["pitchers"]:
                app.append({"game_pk": pk, "date": meta.date, "team_id": t["team_id"], **p})
            opp_sp = o["pitchers"][0]["id"] if o["pitchers"] else None
            lus.append({"game_pk": pk, "date": meta.date, "start": meta.start, "team_id": t["team_id"],
                        "opp_sp": opp_sp, "opp_throws": m.people.get(opp_sp, {}).get("throws", "R"),
                        "lineup": t["lineup"], "pos": t["lineup_pos"]})
    m.app = pd.DataFrame(app)
    m.lineups = pd.DataFrame(lus).sort_values("start")
    st = m.app[m.app.start & (m.app.bf >= 9)].sort_values("date")
    m.lg_leash = st.pitches.mean() if len(st) else 88.0
    # postseason hook: starters' playoff pitch counts vs their regular-season norm (fitted, shrunk to 1)
    gt = dict(zip(m.sched_hist.game_pk, m.sched_hist.game_type))
    a = m.app[m.app.start].copy()
    a["type"] = a.game_pk.map(gt)
    reg = a[a.type == "R"].sort_values("date")
    post = a[a.type.isin(list(POSTSEASON))]
    norm = reg.groupby("id").pitches.apply(lambda x: x.tail(12).mean())
    post = post.assign(norm=post.id.map(norm)).dropna(subset=["norm"])
    n = len(post)
    ratio = post.pitches.sum() / post.norm.sum() if n else 1.0
    m.post_leash = float((n * ratio + 30 * 1.0) / (n + 30)) if n else 0.93
    m.post_leash_n = n


# ================================================================== per-game inputs
def leash(m, pid, date, postseason=False):
    a = m.app[(m.app.id == pid) & m.app.start & (m.app.date < date)].sort_values("date")
    last = a.tail(12)
    n = len(last)
    mu = (last.pitches.sum() + 4 * m.lg_leash) / (n + 4)
    rest = (date - a.date.iloc[-1]).days if n else 5
    if rest <= 3:
        mu -= 10
    if postseason:
        mu *= getattr(m, "post_leash", 0.93)
    return float(mu), rest


def bullpen(m, team_id, date, exclude=(), postseason=False):
    """Rested relievers. Returns (hi-leverage ids, other ids, weights, list of tired arms)."""
    a = m.app[(m.app.team_id == team_id) & (m.app.date < date)]
    recent = a[a.date >= date - dt.timedelta(days=30)]
    rel = recent[~recent.start]
    if not len(rel):
        rel = a[~a.start].tail(60)
    usage = rel.groupby("id").outs.sum()
    starters = set(a[a.start & (a.date >= date - dt.timedelta(days=30))].id)
    y1 = a[a.date == date - dt.timedelta(days=1)]
    y2 = a[a.date == date - dt.timedelta(days=2)]
    tired = set(y1[y1.pitches >= 25].id) | (set(y1.id) & set(y2.id))
    pool = [p for p in usage.index if p not in starters and p not in exclude and usage[p] >= 3]
    avail = [p for p in pool if p not in tired] or pool
    def qual(p):
        r = m.pit.get(p)
        if r is None:
            return 0.33
        v = 0.5 * (r["vs"]["L"] + r["vs"]["R"])
        return float((v * LINEAR_W).sum())
    ranked = sorted([p for p in avail if usage[p] >= 9], key=qual)
    hi = ranked[:3] if len(ranked) >= 3 else ranked
    rest = [p for p in avail if p not in hi] or hi
    if postseason:   # playoff pens shrink to the trusted arms
        rest = [p for p in ranked[3:7] if p in avail] or rest
    return hi, rest, usage.to_dict(), sorted(tired & set(pool))


def unit_rates(m, ids, usage, B):
    if not ids:
        return m.lg_all
    v = np.zeros(8)
    wsum = 0.0
    for p in ids:
        r = m.pit.get(p)
        rr = r["vs"][B] if r else m.lg[(B, "R")]
        wt = max(usage.get(p, 3), 3)
        v += wt * rr
        wsum += wt
    return _norm(v / wsum)


def projected_lineup(m, team_id, opp_throws, date, out=()):
    lu = m.lineups[(m.lineups.team_id == team_id) & (m.lineups.date < date)]
    lu = lu[lu.lineup.map(len) == 9]
    if not len(lu):
        return None, None
    same = lu[lu.opp_throws == opp_throws]
    row = (same if len(same) and (date - same.date.iloc[-1]).days <= 10 else lu).iloc[-1]
    ids, pos = list(row.lineup), list(row.pos)
    out_ids = {_resolve(m, o) for o in out}
    if out_ids & set(ids):
        # replace with the most-used recent bench bat not already in the lineup
        recent = lu.tail(15)
        counts = defaultdict(int)
        for l in recent.lineup:
            for p in l:
                counts[p] += 1
        bench = [p for p, _ in sorted(counts.items(), key=lambda x: -x[1]) if p not in ids and p not in out_ids]
        for i, p in enumerate(ids):
            if p in out_ids:
                ids[i] = bench.pop(0) if bench else p
    return ids, pos


def _resolve(m, who):
    if isinstance(who, (int, np.integer)):
        return int(who)
    key = md.norm_name(who)
    for pid, info in m.people.items():
        if md.norm_name(info["name"]) == key:
            return pid
    for pid, info in m.people.items():
        n = md.norm_name(info["name"])
        if n.endswith(key) or key.endswith(n):
            return pid
    return None


# ================================================================== game setup
def build_game(m, g, lineups=None, starters=None, out=(), weather=None, sims_note=True):
    """Turn a schedule row into simulation tables. g: Series/namedtuple from md.schedule().
    lineups: {"away": [ids], "home": [ids]} override; starters: {"away": id, "home": id} override."""
    date = g.date if isinstance(g.date, dt.date) else md.to_date(str(g.date)[:10])
    post = g.game_type in POSTSEASON
    notes = []
    sides = ("away", "home")
    team = {"away": g.away_id, "home": g.home_id}
    sp = {"away": g.away_sp, "home": g.home_sp}
    if starters:
        for s in sides:
            if starters.get(s) is not None:
                sp[s] = _resolve(m, starters[s])
    sp = {s: (int(sp[s]) if sp[s] is not None and pd.notna(sp[s]) else None) for s in sides}
    throws = {s: (m.pit.get(sp[s], {}).get("throws") or m.people.get(sp[s], {}).get("throws", "R")) if sp[s] else "R"
              for s in sides}

    lu, pos = {}, {}
    for s, o in (("away", "home"), ("home", "away")):
        given = (lineups or {}).get(s) or getattr(g, f"{s}_lineup", None)
        if given is not None and isinstance(given, (list, tuple)) and len(given) == 9:
            lu[s] = [_resolve(m, x) for x in given]
            pos[s] = [m.people.get(p, {}).get("pos", "") for p in lu[s]]
            notes.append(f"{s} lineup: confirmed")
        else:
            lu[s], pos[s] = projected_lineup(m, team[s], throws[o], date, out)
            notes.append(f"{s} lineup: projected from last game vs {throws[o]}HP")
        if lu[s] is None:
            raise ValueError(f"No lineup for {s}")

    # environment
    temp, wind = g.temp, g.wind_out
    roof_closed = bool(g.roof_closed)
    if weather:
        temp, wind = weather.get("temp", temp), weather.get("wind_out", wind)
        roof_closed = weather.get("roof_closed", roof_closed)
    if (temp is None or not np.isfinite(temp)) and not roof_closed and g.lat is not None:
        temp, wind = md.forecast_weather(g.lat, g.lon, g.start, g.azimuth)
        notes.append(f"weather: forecast {temp:.0f}F, wind out {wind:+.1f} mph")
    if str(g.roof).lower() in ("dome",):
        roof_closed = True
    if roof_closed or temp is None or not np.isfinite(temp):
        wx_hr = wx_hit = 1.0
        if roof_closed:
            notes.append("weather: roof closed / dome")
    else:
        t0, w0 = m.venue_wx.get(g.venue_id, m.wx_default)
        wx_hr = math.exp(m.wx["hr_temp"] * (temp - t0) + m.wx["hr_wind"] * (wind - w0))
        wx_hit = math.exp(m.wx["hit_temp"] * (temp - t0) + m.wx["hit_wind"] * (wind - w0))
    park = m.park.get(g.venue_id, np.ones(8))
    ump = m.ump.get(int(g.hp_ump_id), np.ones(2)) if g.hp_ump_id is not None and pd.notna(g.hp_ump_id) else np.ones(2)
    if g.hp_ump_id is not None and pd.notna(g.hp_ump_id):
        notes.append(f"umpire {g.hp_ump}: K x{ump[0]:.3f}, BB x{ump[1]:.3f}")

    # travel flags for this game (computed against each team's previous game)
    trav = {}
    for s in sides:
        prev = m.sched_hist[((m.sched_hist.home_id == team[s]) | (m.sched_hist.away_id == team[s])) &
                            (m.sched_hist.date < date)].sort_values("start").tail(1)
        mult = 1.0
        if len(prev):
            p = prev.iloc[0]
            days = (date - p.date).days
            if days <= 1 and p.lon is not None and g.lon is not None and abs(p.lon - g.lon) > 25:
                mult *= m.travel["long_trip"]
                notes.append(f"{s}: long trip on short rest x{m.travel['long_trip']:.3f}")
            if days == 1 and p.day_night == "night" and g.day_night == "day":
                mult *= m.travel["day_after_night"]
                notes.append(f"{s}: day game after night game x{m.travel['day_after_night']:.3f}")
        trav[s] = mult

    # per-team fielding effects (defense and catcher framing apply to the OTHER team's batting)
    fld = {}
    for s in sides:
        oaa = m.oaa_pg.get(team[s], 0.0)
        hit_mult = 1 - oaa / 5.8
        catcher = [p for p, ps in zip(lu[s], pos[s]) if ps == "C"]
        rpg = m.framing_rpg.get(catcher[0], 0.0) if catcher else 0.0
        fld[s] = {"hit": hit_mult, "k": 1 + 0.27 * rpg, "bb": 1 - 0.40 * rpg}

    # pitching units
    units, ppa, leashes, pen_info = {}, {}, {}, {}
    for s in sides:
        hi, rest, usage, tired = bullpen(m, team[s], date, exclude=[sp[s]] if sp[s] else [], postseason=post)
        pen_info[s] = {"hi": hi, "rest": rest, "tired": tired}
        if tired:
            notes.append(f"{s} tired relievers: " + ", ".join(m.people.get(p, {}).get('name', str(p)) for p in tired))
        if sp[s] and sp[s] in m.pit:
            st = {B: m.pit[sp[s]]["vs"][B] for B in "LR"}
            sppa = m.pit[sp[s]]["ppa"] / m.lg_p_per_pa
            L, rest_days = leash(m, sp[s], date, post)
            if rest_days <= 3:
                notes.append(f"{s} starter on short rest ({rest_days} days)")
        elif sp[s]:
            T = throws[s]
            st = {B: m.lg[(B, T)] for B in "LR"}
            sppa, L = 1.0, m.lg_leash * 0.9
            notes.append(f"{s} starter {m.people.get(sp[s], {}).get('name', sp[s])}: no MLB history, league average")
        else:
            st = {B: unit_rates(m, rest, usage, B) for B in "LR"}
            sppa, L = 1.0, 35.0
            notes.append(f"{s}: no probable starter, bullpen game")
        units[s] = [st, {B: unit_rates(m, rest, usage, B) for B in "LR"}, {B: unit_rates(m, hi, usage, B) for B in "LR"}]
        ppa[s] = sppa
        leashes[s] = L

    # probability tables P[bat_team, slot, unit, tto, outcome]
    P = np.zeros((2, 9, 3, 4, 8))
    speeds = np.zeros((2, 9))
    gb = np.zeros(2)
    for ti, s in enumerate(sides):
        o = sides[1 - ti]
        env = park.copy()
        env[HR_] *= wx_hr
        env[[S1, S2, S3]] *= wx_hit
        env[K_] *= ump[0] * fld[o]["k"]
        env[BB_] *= ump[1] * fld[o]["bb"]
        env[[S1, S2, S3]] *= fld[o]["hit"]
        env[[BB_, HBP_, S1, S2, S3, HR_]] *= trav[s]
        lvl = getattr(m, "level_vec", None)
        if lvl is not None:
            env = env * lvl
        if post:
            env = env * getattr(m, "post_vec", np.ones(8))
        if s == "home":
            env = env * m.home_mult
        for slot, pid in enumerate(lu[s]):
            b = m.bat.get(pid)
            bats = (b or {}).get("bats") or m.people.get(pid, {}).get("bats", "R")
            speeds[ti, slot] = (b or {}).get("speed", LG_SPEED)
            for u in range(3):
                T = throws[o] if u == 0 else "R"
                for tto in range(4):
                    if u > 0 and tto > 0:
                        P[ti, slot, u, tto] = P[ti, slot, u, 0]
                        continue
                    B = ("R" if T == "L" else "L") if bats == "S" else bats
                    pr = units[o][u][B]
                    br = b["vs"][T] if b else m.lg[(B, T)]
                    l = m.lg[(B, T)]
                    v = br * pr / l
                    v = v * env
                    if u == 0:
                        kf, hf = TTO[tto]
                        v[K_] *= kf
                        v[[BB_, HBP_, S1, S2, S3, HR_]] *= hf
                    P[ti, slot, u, tto] = _norm(v)
        spid = sp[o]
        gb[ti] = m.pit.get(spid, {}).get("gb", m.lg_gb) if spid else m.lg_gb

    names = {s: [m.people.get(p, {}).get("name", str(p)) for p in lu[s]] for s in sides}
    return {"game_pk": g.game_pk, "date": date, "away": g.away, "home": g.home, "away_id": g.away_id,
            "home_id": g.home_id, "postseason": post, "P": P, "speed": speeds, "gb": gb,
            "ppa": np.array([ppa["away"], ppa["home"]]), "leash": np.array([leashes["away"], leashes["home"]]),
            "lineup_ids": lu, "lineup_names": names, "starters": sp,
            "starter_names": {s: (m.people.get(sp[s], {}).get("name") if sp[s] else "bullpen") for s in sides},
            "throws": throws, "pen": pen_info, "notes": notes,
            "env": {"park": dict(zip(OUT, np.round(park, 3))), "temp": temp, "wind_out": wind,
                    "roof_closed": roof_closed, "wx_hr": wx_hr, "wx_hit": wx_hit, "ump": g.hp_ump}}


# ================================================================== simulator
NB = 12   # batter stats: PA H 1B 2B 3B HR BB K R RBI TB HBP
B_PA, B_H, B_1B, B_2B, B_3B, B_HR, B_BB, B_K, B_R, B_RBI, B_TB, B_HBP = range(NB)
NP = 8    # starter stats: BF K BB H R OUTS HR PITCHES
P_BF, P_K, P_BB, P_H, P_R, P_OUTS, P_HR, P_PIT = range(NP)


def simulate(G, n=6000, seed=None):
    rng = np.random.default_rng(seed)
    P, speed, gbp = G["P"], G["speed"], G["gb"]
    cumP = np.cumsum(P, axis=-1)
    post = G["postseason"]
    hi_inn = 7 if post else 8

    inning = np.ones(n, int)
    half = np.zeros(n, int)
    outs = np.zeros(n, int)
    base = -np.ones((n, 3), int)       # lineup slot of runner on 1st/2nd/3rd
    resp = np.full((n, 3), 2, int)     # 0 = charged to starter, 1 = bullpen, 2 = nobody (ghost runner)
    score = np.zeros((n, 2), int)
    bidx = np.zeros((n, 2), int)
    st_in = np.ones((n, 2), bool)
    pitches = np.zeros((n, 2))
    st_bf = np.zeros((n, 2), int)
    lim = G["leash"][None, :] + rng.normal(0, 13, (n, 2))
    done = np.zeros(n, bool)
    bat = np.zeros((n, 2, 9, NB), np.int16)
    pit = np.zeros((n, 2, NP), np.float32)   # indexed by FIELDING team (its starter)
    f5 = np.zeros((n, 2), int)
    i1 = np.zeros((n, 2), int)
    first = -np.ones(n, int)
    lead_exit = np.zeros((n, 2), bool)
    lead_lost = np.zeros((n, 2), bool)
    exited = np.zeros((n, 2), bool)
    ar = np.arange(n)

    for step in range(400):
        act = ~done
        if not act.any():
            break
        a = np.nonzero(act)[0]
        bt = half[a]
        ft = 1 - bt
        # ---- pitching change
        sin = st_in[a, ft]
        pull = sin & ((pitches[a, ft] >= lim[a, ft]) | (pit[a, ft, P_R] >= 6) | (st_bf[a, ft] >= 36))
        if pull.any():
            ia, it = a[pull], ft[pull]
            st_in[ia, it] = False
            exited[ia, it] = True
            lead = score[ia, it] - score[ia, 1 - it]
            lead_exit[ia, it] = (lead > 0) & (pit[ia, it, P_OUTS] >= 15)
        sin = st_in[a, ft]
        lead_f = score[a, ft] - score[a, bt]
        hi = (~sin) & (inning[a] >= hi_inn) & (lead_f >= -1) & (lead_f <= 3)
        unit = np.where(sin, 0, np.where(hi, 2, 1))
        tto = np.where(sin, np.minimum(st_bf[a, ft] // 9, 3), 0)
        slot = bidx[a, bt]
        cp = cumP[bt, slot, unit, tto]
        u = rng.random(len(a))
        o = (u[:, None] > cp).sum(1)
        o = np.minimum(o, 7)

        # ---- pitch count
        fac = np.where(sin, G["ppa"][ft], 1.0)
        pitches[a, ft] += rng.poisson(PITCHES_BY_OUTCOME[o] * fac)

        # ---- base / out resolution
        b1, b2, b3 = base[a, 0].copy(), base[a, 1].copy(), base[a, 2].copy()
        r1, r2, r3 = resp[a, 0].copy(), resp[a, 1].copy(), resp[a, 2].copy()
        o0 = outs[a].copy()
        runs = np.zeros(len(a), int)
        rbi = np.zeros(len(a), int)
        new_resp = np.where(sin, 0, 1)
        score_events = []   # (mask, runner slot array, resp array, counts_rbi)

        def score_r(mask, rs, rr, counts_rbi=True):
            mk = mask & (rs >= 0)
            score_events.append((mk, rs.copy(), rr.copy(), counts_rbi))

        nb1, nb2, nb3 = b1.copy(), b2.copy(), b3.copy()
        nr1, nr2, nr3 = r1.copy(), r2.copy(), r3.copy()
        add_outs = np.zeros(len(a), int)
        bsp = speed[bt, slot]

        # walks / HBP: forced advances
        m_bb = (o == BB_) | (o == HBP_)
        if m_bb.any():
            full = m_bb & (b1 >= 0) & (b2 >= 0) & (b3 >= 0)
            score_r(full, b3, r3)
            f12 = m_bb & (b1 >= 0) & (b2 >= 0)
            nb3 = np.where(f12, b2, nb3); nr3 = np.where(f12, r2, nr3)
            f1 = m_bb & (b1 >= 0)
            nb2 = np.where(f1, b1, nb2); nr2 = np.where(f1, r1, nr2)
            nb1 = np.where(m_bb, slot, nb1); nr1 = np.where(m_bb, new_resp, nr1)
        # single
        m1 = o == S1
        if m1.any():
            score_r(m1, b3, r3)
            sp2 = speed[bt, np.maximum(b2, 0)]
            s2 = m1 & (b2 >= 0) & (rng.random(len(a)) < np.clip(0.60 + 0.04 * (sp2 - LG_SPEED), 0.3, 0.9))
            score_r(s2, b2, r2)
            stay2 = m1 & (b2 >= 0) & ~s2
            sp1 = speed[bt, np.maximum(b1, 0)]
            t13 = m1 & (b1 >= 0) & ~stay2 & (rng.random(len(a)) < np.clip(0.28 + 0.04 * (sp1 - LG_SPEED), 0.05, 0.6))
            nb3 = np.where(m1, np.where(stay2, b2, np.where(t13, b1, -1)), nb3)
            nr3 = np.where(m1, np.where(stay2, r2, np.where(t13, r1, 2)), nr3)
            nb2 = np.where(m1, np.where((b1 >= 0) & ~t13, b1, -1), nb2)
            nr2 = np.where(m1, np.where((b1 >= 0) & ~t13, r1, 2), nr2)
            nb1 = np.where(m1, slot, nb1); nr1 = np.where(m1, new_resp, nr1)
        # double
        m2 = o == S2
        if m2.any():
            score_r(m2, b3, r3); score_r(m2, b2, r2)
            sp1 = speed[bt, np.maximum(b1, 0)]
            h1 = m2 & (b1 >= 0) & (rng.random(len(a)) < np.clip(0.42 + 0.05 * (sp1 - LG_SPEED), 0.15, 0.8))
            score_r(h1, b1, r1)
            nb3 = np.where(m2, np.where((b1 >= 0) & ~h1, b1, -1), nb3)
            nr3 = np.where(m2, np.where((b1 >= 0) & ~h1, r1, 2), nr3)
            nb2 = np.where(m2, slot, nb2); nr2 = np.where(m2, new_resp, nr2)
            nb1 = np.where(m2, -1, nb1)
        # triple / homer
        for code in (S3, HR_):
            mk = o == code
            if mk.any():
                score_r(mk, b3, r3); score_r(mk, b2, r2); score_r(mk, b1, r1)
                if code == HR_:
                    score_r(mk, slot.copy(), new_resp)
                    nb3 = np.where(mk, -1, nb3)
                else:
                    nb3 = np.where(mk, slot, nb3); nr3 = np.where(mk, new_resp, nr3)
                nb2 = np.where(mk, -1, nb2); nb1 = np.where(mk, -1, nb1)
        # strikeout
        mk_k = o == K_
        add_outs += mk_k
        # out in play
        mo = o == OIP_
        if mo.any():
            rnd = rng.random((len(a), 4))
            err = mo & (rnd[:, 0] < 0.022)
            gbm = mo & ~err & (rnd[:, 1] < gbp[ft])
            air = mo & ~err & ~gbm
            # reached on error: everyone up one base
            score_r(err, b3, r3, counts_rbi=False)
            nb3 = np.where(err, b2, nb3); nr3 = np.where(err, r2, nr3)
            nb2 = np.where(err, b1, nb2); nr2 = np.where(err, r1, nr2)
            nb1 = np.where(err, slot, nb1); nr1 = np.where(err, new_resp, nr1)
            # ground ball
            dp_ok = gbm & (b1 >= 0) & (o0 < 2)
            dp = dp_ok & (rnd[:, 2] < np.clip(0.22 * (1 - 0.07 * (bsp - LG_SPEED)), 0.05, 0.4))
            add_outs += np.where(dp, 2, 0)
            sc3 = gbm & (b3 >= 0) & (o0 + np.where(dp, 2, 1) < 3) & (rnd[:, 3] < 0.55)
            score_r(sc3, b3, r3, counts_rbi=False)
            add_outs += np.where(gbm & ~dp, 1, 0)
            # runners after a grounder
            g3 = np.where(sc3, -1, b3)
            g3r = np.where(sc3, 2, r3)
            adv2 = (b2 >= 0) & (g3 < 0) & ((rnd[:, 0] < 0.55) | ((b1 >= 0) & ~dp))   # forced if 1st occupied
            # fielder's choice: lead runner from 1st forced out, batter safe (always when 2nd can't clear)
            fc = gbm & ~dp & (b1 >= 0) & ((rnd[:, 3] < 0.5) | ((b2 >= 0) & ~adv2))
            nb3 = np.where(gbm, np.where(adv2, b2, g3), nb3)
            nr3 = np.where(gbm, np.where(adv2, r2, g3r), nr3)
            nb2 = np.where(gbm, np.where(adv2 | (b2 < 0), np.where((b1 >= 0) & ~dp & ~fc, b1, -1), b2), nb2)
            nr2 = np.where(gbm, np.where(adv2 | (b2 < 0), np.where((b1 >= 0) & ~dp & ~fc, r1, 2), r2), nr2)
            nb1 = np.where(gbm, np.where(fc, slot, -1), nb1)
            nr1 = np.where(gbm, np.where(fc, new_resp, 2), nr1)
            # fly / line out
            add_outs += air
            sf = air & (b3 >= 0) & (o0 < 2) & (rnd[:, 2] < 0.40)
            score_r(sf, b3, r3)
            tag2 = air & (b2 >= 0) & (o0 < 2) & ((b3 < 0) | sf) & (rnd[:, 3] < 0.22)
            nb3 = np.where(air, np.where(sf, np.where(tag2, b2, -1), np.where(tag2, b2, b3)), nb3)
            nr3 = np.where(air, np.where(sf, np.where(tag2, r2, 2), np.where(tag2, r2, r3)), nr3)
            nb2 = np.where(air, np.where(tag2, -1, b2), nb2)
        new_outs = o0 + add_outs
        # steal attempts / wild pitches with a runner on 1st and 2nd open (A: 7% attempt, 78% success)
        rr_ = rng.random((len(a), 2))
        sb_try = ((o == K_) | (o == OIP_)) & (nb1 >= 0) & (nb2 < 0) & (new_outs < 3) & (rr_[:, 0] < 0.07)
        sp_run = speed[bt, np.maximum(nb1, 0)]
        sb_ok = sb_try & (rr_[:, 1] < np.clip(0.78 + 0.04 * (sp_run - LG_SPEED), 0.5, 0.95))
        cs = sb_try & ~sb_ok
        nb2 = np.where(sb_ok, nb1, nb2); nr2 = np.where(sb_ok, nr1, nr2)
        nb1 = np.where(sb_try, -1, nb1)
        new_outs = new_outs + cs
        # extra-base attempts thrown out on hits (A: ~4% of runners who were on base)
        hit_any = np.isin(o, [S1, S2])
        tho = hit_any & (nb3 >= 0) & (nb3 != b3) & (rng.random(len(a)) < 0.08) & (new_outs < 3)
        nb3 = np.where(tho, -1, nb3)       # runner who went for 3rd is thrown out
        new_outs = new_outs + tho
        # apply scoring (runs after the 3rd out on the same play don't count, except none of our plays do that)
        for mk, rs, rr, crbi in score_events:
            mk = mk & (new_outs < 3) | (mk & ((o == HR_) | (o == S1) | (o == S2) | (o == S3) | m_bb))
            if not mk.any():
                continue
            ii = a[mk]
            tb = bt[mk]
            score[ii, tb] += 1
            np.add.at(bat, (ii, tb, rs[mk], B_R), 1)
            runs[mk] += 1
            if crbi:
                rbi[mk] += 1
            st_run = mk & (rr == 0)
            if st_run.any():
                np.add.at(pit, (a[st_run], ft[st_run], P_R), 1)
        # batter stats
        sl = slot
        bat[a, bt, sl, B_PA] += 1
        ishit = np.isin(o, HITS)
        bat[a, bt, sl, B_H] += ishit
        bat[a, bt, sl, B_1B] += o == S1
        bat[a, bt, sl, B_2B] += o == S2
        bat[a, bt, sl, B_3B] += o == S3
        bat[a, bt, sl, B_HR] += o == HR_
        bat[a, bt, sl, B_BB] += o == BB_
        bat[a, bt, sl, B_HBP] += o == HBP_
        bat[a, bt, sl, B_K] += o == K_
        bat[a, bt, sl, B_TB] += np.select([o == S1, o == S2, o == S3, o == HR_], [1, 2, 3, 4], 0)
        bat[a, bt, sl, B_RBI] += rbi.astype(np.int16)
        # starter stats
        if sin.any():
            ia, it, oo = a[sin], ft[sin], o[sin]
            pit[ia, it, P_BF] += 1
            pit[ia, it, P_K] += oo == K_
            pit[ia, it, P_BB] += oo == BB_
            pit[ia, it, P_H] += np.isin(oo, HITS)
            pit[ia, it, P_HR] += oo == HR_
            pit[ia, it, P_OUTS] += np.minimum(new_outs[sin], 3) - o0[sin]
            st_bf[ia, it] += 1
        pit[a, ft, P_PIT] = np.where(st_in[a, ft], pitches[a, ft], pit[a, ft, P_PIT])

        base[a, 0], base[a, 1], base[a, 2] = nb1, nb2, nb3
        resp[a, 0], resp[a, 1], resp[a, 2] = nr1, nr2, nr3
        outs[a] = np.minimum(new_outs, 3)
        bidx[a, bt] = (slot + 1) % 9

        # first to score
        scored = runs > 0
        nf = scored & (first[a] < 0)
        first[a[nf]] = bt[nf]
        # starter lead protection
        for t in (0, 1):
            le = lead_exit[a, t]
            lead_lost[a[le], t] |= (score[a[le], t] - score[a[le], 1 - t]) <= 0

        # walk-off
        wo = (half[a] == 1) & (inning[a] >= 9) & (score[a, 1] > score[a, 0])
        done[a[wo]] = True
        # end of half inning
        e3 = (outs[a] >= 3) & ~wo
        if e3.any():
            ie = a[e3]
            base[ie] = -1
            resp[ie] = 2
            outs[ie] = 0
            top = half[ie] == 0
            it_ = ie[top]
            home_up = (inning[it_] >= 9) & (score[it_, 1] > score[it_, 0])
            done[it_[home_up]] = True
            half[it_[~home_up]] = 1
            ib = ie[~top]
            fin = (inning[ib] >= 9) & (score[ib, 0] != score[ib, 1])
            done[ib[fin]] = True
            r5 = inning[ib] == 5
            f5[ib[r5]] = score[ib[r5]]
            r1_ = inning[ib] == 1
            i1[ib[r1_]] = score[ib[r1_]]
            cont = ib[~fin]
            inning[cont] += 1
            half[cont] = 0
            cap = cont[inning[cont] > 20]
            if len(cap):
                w = rng.integers(0, 2, len(cap))
                score[cap, w] += 1
                done[cap] = True
            # automatic runner (regular season extras)
            if not post:
                newh = ie[~done[ie] & (inning[ie] >= 10)]
                if len(newh):
                    tb2 = half[newh]
                    base[newh, 1] = (bidx[newh, tb2] - 1) % 9
                    resp[newh, 1] = 2

    win = score[:, 1] > score[:, 0]
    st_win = np.zeros((n, 2), bool)
    for t in (0, 1):
        tw = win if t == 1 else ~win
        st_win[:, t] = lead_exit[:, t] & ~lead_lost[:, t] & tw
    return {"score": score, "f5": f5, "i1": i1, "first": first, "bat": bat, "pit": pit,
            "innings": inning, "st_win": st_win, "n": n}


# ================================================================== summaries
def summarize(G, S):
    sc = S["score"].astype(float)
    away, home = sc[:, 0], sc[:, 1]
    tot = away + home
    margin = home - away
    f5 = S["f5"].astype(float)
    i1 = S["i1"].sum(1)
    out = {
        "game": f"{G['away']} @ {G['home']}", "game_pk": G["game_pk"], "date": G["date"],
        "away": G["away"], "home": G["home"],
        "away_sp": G["starter_names"]["away"], "home_sp": G["starter_names"]["home"],
        "home_win": float((margin > 0).mean()), "away_win": float((margin < 0).mean()),
        "away_runs": away.mean(), "home_runs": home.mean(), "total": tot.mean(), "total_median": float(np.median(tot)),
        "home_rl_-1.5": float((margin >= 2).mean()), "away_rl_+1.5": float((margin >= -1).mean()),
        "away_rl_-1.5": float((margin <= -2).mean()), "home_rl_+1.5": float((margin <= 1).mean()),
        "f5_home_win": float((f5[:, 1] > f5[:, 0]).mean()), "f5_away_win": float((f5[:, 0] > f5[:, 1]).mean()),
        "f5_tie": float((f5[:, 0] == f5[:, 1]).mean()), "f5_total": f5.sum(1).mean(),
        "nrfi": float((i1 == 0).mean()), "yrfi": float((i1 > 0).mean()),
        "home_scores_first": float((S["first"] == 1).mean()), "away_scores_first": float((S["first"] == 0).mean()),
        "extras": float((S["innings"] > 9).mean()),
    }
    out["_tot"] = tot
    out["_margin"] = margin
    out["_away"], out["_home"] = away, home
    out["_f5"] = f5
    return out


def prob_over(arr, line):
    arr = np.asarray(arr)
    over = (arr > line).mean()
    push = (arr == line).mean()
    return float(over), float(push)


def player_props(G, S):
    """Per-player distributions from the same simulated games (so same-game combos are consistent)."""
    rows = []
    for ti, side in enumerate(("away", "home")):
        for slot, (pid, name) in enumerate(zip(G["lineup_ids"][side], G["lineup_names"][side])):
            b = S["bat"][:, ti, slot, :].astype(float)
            rows.append({"side": side, "team": G[side], "player": name, "pid": pid, "role": "batter", "slot": slot + 1,
                         "PA": b[:, B_PA].mean(), "hits": b[:, B_H].mean(), "total_bases": b[:, B_TB].mean(),
                         "home_runs": b[:, B_HR].mean(), "runs": b[:, B_R].mean(), "rbi": b[:, B_RBI].mean(),
                         "hrr": (b[:, B_H] + b[:, B_R] + b[:, B_RBI]).mean(), "walks": b[:, B_BB].mean(),
                         "singles": b[:, B_1B].mean(), "doubles": b[:, B_2B].mean(), "strikeouts": b[:, B_K].mean(),
                         "P(1+ hit)": (b[:, B_H] >= 1).mean(), "P(HR)": (b[:, B_HR] >= 1).mean(),
                         "_dist": {"hits": b[:, B_H], "total_bases": b[:, B_TB], "home_runs": b[:, B_HR],
                                   "runs": b[:, B_R], "rbi": b[:, B_RBI], "hrr": b[:, B_H] + b[:, B_R] + b[:, B_RBI],
                                   "walks": b[:, B_BB], "singles": b[:, B_1B], "doubles": b[:, B_2B],
                                   "strikeouts": b[:, B_K]}})
        p = S["pit"][:, ti, :].astype(float)
        pid = G["starters"][side]
        if pid:
            rows.append({"side": side, "team": G[side], "player": G["starter_names"][side], "pid": pid,
                         "role": "pitcher", "slot": 0, "strikeouts": p[:, P_K].mean(), "outs": p[:, P_OUTS].mean(),
                         "hits_allowed": p[:, P_H].mean(), "walks_allowed": p[:, P_BB].mean(),
                         "earned_runs": (p[:, P_R] * 0.93).mean(), "pitches": p[:, P_PIT].mean(),
                         "P(win)": S["st_win"][:, ti].mean(),
                         "_dist": {"strikeouts": p[:, P_K], "outs": p[:, P_OUTS], "hits_allowed": p[:, P_H],
                                   "walks_allowed": p[:, P_BB], "earned_runs": p[:, P_R],
                                   "win": S["st_win"][:, ti].astype(float)}})
    return rows


def calibrate(m, days=30, n_games=250, sims=500, verbose=True):
    """Recent-environment check. Simulates a sample of games from the `days` before the fit date
    with their real lineups and starters, compares simulated per-PA outcome rates with what
    happened, and stores a shrunk per-outcome multiplier (m.level_vec). Uses only games before
    m.asof, so it is safe in backtests. Turn off with m.level_vec = None."""
    m.level_vec = None
    h = m.sched_hist
    h = h[(h.date >= m.asof - dt.timedelta(days=days)) & (h.date < m.asof)]
    h = h.sample(min(n_games, len(h)), random_state=0)
    sim_c, act_c = np.zeros(8), np.zeros(8)
    sim_r, act_r = [], []
    for g in h.itertuples():
        b = m.box.get(int(g.game_pk))
        if not b or any(len(b["teams"][s]["lineup"]) != 9 for s in ("away", "home")):
            continue
        lus = {s: b["teams"][s]["lineup"] for s in ("away", "home")}
        sps = {s: (b["teams"][s]["pitchers"][0]["id"] if b["teams"][s]["pitchers"] else None) for s in ("away", "home")}
        try:
            G = build_game(m, g, lineups=lus, starters=sps)
        except Exception:
            continue
        S = simulate(G, sims, seed=7)
        x = S["bat"].astype(float).sum(2).mean(0).sum(0)
        pa = x[B_PA]
        sc = np.array([x[B_K], x[B_BB], x[B_HBP], x[B_1B], x[B_2B], x[B_3B], x[B_HR]])
        sim_c[:7] += sc / pa
        t = b["teams"]
        f = lambda k: sum(t[s]["bat"][k] for s in t)
        apa = f("plateAppearances")
        ac = np.array([f("strikeOuts"), f("baseOnBalls"), f("hitByPitch"),
                       f("hits") - f("doubles") - f("triples") - f("homeRuns"), f("doubles"), f("triples"), f("homeRuns")])
        act_c[:7] += ac
        sim_r.append(S["score"].sum(1).mean())
        act_r.append(g.away_score + g.home_score)
        sim_c[7] += apa          # PA total for scaling
    n_pa = sim_c[7]
    # expected counts if sim per-PA rates applied to the actual PA
    rate_sim = sim_c[:7] / max(len(act_r), 1)
    exp_cnt = rate_sim * n_pa
    k = np.array([400, 200, 40, 400, 150, 30, 80], float)
    lv = np.ones(8)
    lv[:7] = (act_c[:7] + k) / (exp_cnt + k)
    m.level_vec = np.clip(lv, 0.85, 1.15)
    m.level_info = {"games": len(act_r), "actual_rpg": float(np.mean(act_r)), "sim_rpg_before": float(np.mean(sim_r))}
    if verbose:
        print(f"Recent environment ({len(act_r)} games, last {days} days): actual {np.mean(act_r):.2f} runs/game, "
              f"sim {np.mean(sim_r):.2f} before adjustment")
        print("  per-PA multipliers: " + ", ".join(f"{o} x{v:.3f}" for o, v in zip(OUT[:7], m.level_vec[:7])))
    return m.level_vec


def run_game(m, g, n=6000, seed=None, **kw):
    G = build_game(m, g, **kw)
    S = simulate(G, n, seed)
    res = summarize(G, S)
    res["props"] = player_props(G, S)
    res["G"] = G
    return res
