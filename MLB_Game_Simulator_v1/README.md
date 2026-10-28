# MLB Game Simulator v1

Plate-appearance Monte Carlo for every MLB game, priced against DraftKings.

## Files
| File | What it does |
|---|---|
| `mlb_data.py` | Downloads and caches everything: MLB Stats API (schedule, lineups, umpires, weather, boxscores, platoon splits), Baseball Savant (defense, framing, sprint speed, xwOBA), Open-Meteo forecasts, DraftKings lines and props via ESPN |
| `mlb_game_sim.py` | Model fitting and the simulator |
| `mlb_daily.py` | `daily_report()`, `explore_game()`, `same_game()`, `grade_log()`, `backtest()` |
| `run_day.py` | Command line: `python run_day.py`, `--grade`, `--backtest START END` |
| `MLB_Game_Simulator.ipynb` | Notebook walkthrough |

No API key. `pip install pandas numpy requests openpyxl`. First run pulls ~5,000 boxscores (2-3 min) into `mlb_data/`.

## Betting sections (same as the NFL simulator, plus baseball-specific markets)
Moneyline · run line ±1.5 · game total · team totals · first 5 innings (moneyline, run line, total) · first inning NRFI/YRFI · first team to score · pitcher props (strikeouts, outs, hits allowed, walks allowed, earned runs, to record win) · batter props (hits, total bases, home run, runs, RBIs, H+R+RBI, singles, doubles, walks). Each shows model probability, fair odds, DraftKings price, DraftKings no-vig probability, edge and EV per $1. Same-game combos come from the same simulations, so correlated legs are priced correctly.

## Game factors
Fitted from data and shrunk by sample size: batter and pitcher talent (3 seasons, weighted to this year), platoon splits, switch hitters, Statcast xwOBA luck, per-outcome park factors, temperature and wind relative to each park's normal, home-plate umpire K/BB tendencies, home field, long trips on short rest, day game after night game, team outs above average, each starter's pitch count leash, pitches per PA, short-rest starters, postseason hooks (fitted from last postseason), bullpen availability (back-to-back days or 25+ pitches yesterday = tired), best-3 rested relievers in close late innings, recent 30-day scoring environment.

Assumed constants: times-through-order penalty, catcher framing conversion, baserunning advancement odds (adjusted by sprint speed), double plays, sac flies, steals and caught stealing, runners thrown out, reached on error.

Rules: walk-offs, home team skips the bottom of the 9th when ahead, automatic runner in regular-season extras only.

## Honest limits
- Backtest weather uses conditions reported at first pitch, not the morning forecast.
- Statcast inputs in backtests come from the prior season to avoid leakage; live runs use the current season.
- Props "open" lines are the only lines guaranteed pre-game in ESPN's feed, so prop backtests grade against openers.
- One season of September and playoff games is a few hundred games. ROI ranges in the backtest output show how wide the uncertainty is.
