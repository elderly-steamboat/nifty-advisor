# Architecture

All trading logic lives in the `advisor/` package. It has no UI. The Mac app and its local API
(Plan 2) only call these functions and show the results.

## Data flow (one direction only)

```
          NSE archives (bhavcopy, ind_close_all)
                         │
                    data.py ── calendar.py ── config.py (config.yaml, events.yaml, holidays.yaml)
                    │      │
          analytics.py    forecast.py (Kronos via vendor/kronos)
                    │      │
                    scoring.py
                         │
                    strategy.py
                         │
                    risk.py
                         │
          journal.py ── engine.py   ← public entry points
```

Lower modules never import higher ones. Example: `data.py` never imports `strategy.py`.

## What each file owns

| File | Job | Key functions |
|---|---|---|
| `config.py` | Load and validate YAML; config hash | `load_config`, `config_hash`, `load_events`, `load_holidays` |
| `calendar.py` | Trading days, events in a window | `trading_days_between`, `next_trading_days`, `events_between` |
| `data.py` | Download, cache and parse NSE files; build option chains | `refresh`, `data_status`, `load_chain`, `list_expiries`, `load_index_history` |
| `analytics.py` | Black-76, IV, greeks, PCR, walls, max pain, skew, IV rank | `enrich_chain`, `find_walls`, `implied_range` |
| `forecast.py` | Kronos paths (deterministic), calibration, quantiles, touch probs | `run_forecast`, `touch_prob`, `kronos_sampler` |
| `scoring.py` | Direction score and premium edge | `direction_score`, `premium_edge` |
| `strategy.py` | Strategy table, gates, strikes, walls, wings, sizing | `select_structures`, `build_candidate` |
| `risk.py` | Marking, stress, GREEN/AMBER/RED, pre-entry check, exits and rolls | `assess`, `pre_entry_check`, `decide_action`, `propose_roll` |
| `journal.py` | SQLite trade journal | `open_db`, `add_trade`, `apply_roll`, `close_trade` |
| `engine.py` | Runs everything above in order | `recommend`, `log_trade`, `check_positions`, `recommendation_json` |

## Where things live on disk (`data_dir`, default `~/.nifty-advisor`)

| Path | What |
|---|---|
| `bhavcopy/YYYY-MM-DD.csv.zip` | Raw NSE F&O bhavcopy, one per trading day |
| `indices/YYYY-MM-DD.csv` | Raw NSE `ind_close_all` file, one per trading day |
| `index_daily.csv` | Nifty 50 OHLCV + India VIX, one row per day (built from `indices/`) |
| `metrics_daily.csv` | Cached front-month ATM IV and 25Δ risk reversal per day (for IV rank and skew) |
| `forecasts/*.npz` | Cached raw Kronos paths, keyed by date, expiry and Kronos settings |
| `journal.sqlite` | Trade journal (created by the app) |

You can delete any cache file and it will be rebuilt. **Never delete `journal.sqlite`. Back it up instead.**

## Determinism

- Kronos runs on CPU, with `torch.use_deterministic_algorithms(True)`.
- Each forecast gets one seed, built from `base_seed | date | expiry`.
- Library versions are pinned in `uv.lock`, and Kronos is pinned at commit `67b630e`.
- `recommendation_json` sorts keys and rounds floats to 6 decimals. Rule thresholds are compared
  on those rounded values, so the same inputs always give byte-identical JSON.
