# Runbook

## Daily use (before the Mac app exists)
```python
from datetime import date
from advisor.config import load_config, load_holidays
from advisor.data import refresh
from advisor.engine import recommend, recommendation_json, log_trade, check_positions
from advisor.journal import open_db

cfg = load_config()
refresh(date(2023, 9, 1), date.today(), cfg, load_holidays("holidays.yaml"))   # first time: ~3 years; later only missing days download
rec = recommend(date(2026, 11, 17), date(2026, 12, 29), cfg)                    # first run per date+expiry is slow
print(recommendation_json(rec))
conn = open_db(cfg.data_dir / "journal.sqlite")
check_positions(conn, date(2026, 11, 17), cfg)
```

## Refreshing data
- `refresh(start, end, cfg, holidays)` downloads only the trading days that are missing.
- NSE usually publishes the bhavcopy and index files by about 6–7 pm IST. Before then, the
  log says "not published".
- First setup needs about 3 years of data: 512 trading days of history for Kronos, plus 252
  days for IV rank.

## Kronos speed
- **Kronos-base:** 200 paths over a 25-day horizon took about **10 minutes** on this Mac's CPU
  (a full first scan, including building 252 days of IV history, took 12 minutes)
  (measured with other work running at the same time).
- **After the first run:** results are cached in `forecasts/`, so later runs for the same date
  and expiry are instant.
- **Faster option:** set `kronos.model_id: NeoQuasar/Kronos-small`, or lower `kronos.n_paths` to
  100. Either change also changes the config hash.
- **CPU only:** keep `kronos.device: cpu`. GPU and MPS results are not exactly repeatable.

## Yearly upkeep (each December)
- **`holidays.yaml`:** add next year's NSE trading holidays from the NSE circular. A missing
  holiday makes trading-day counts off by one.
- **`events.yaml`:** add next year's RBI MPC decision dates (RBI site), FOMC decision dates
  (federalreserve.gov), the Union Budget date, and any election-result dates.

## When NSE changes something
- **Lot size:**
  - New bhavcopy files include the lot size, so nothing needs changing.
  - `lot_size_fallback` is only for old files; update it if the circular changes it.
- **Expiry day:** expiries come from the bhavcopy, so nothing needs changing.
- **Bhavcopy file format:** `parse_bhavcopy` raises "unknown bhavcopy format". Add a new branch
  in `data.parse_bhavcopy` and a trimmed sample file under `tests/fixtures/`.
- **Download URLs:** they are in `data._bhav_urls` and `data.download_index_close`.

## Upgrading Kronos
1. `cd vendor/kronos && git fetch && git checkout <new sha>`, then update the SHA in `README.md`.
2. Bump `FORECAST_VERSION` in `advisor/forecast.py` so old cached forecasts aren't reused.
3. Run `RUN_SLOW=1 uv run pytest tests/test_forecast_kronos.py`.

## Caches and backups
- **Safe to delete:** `forecasts/`, `metrics_daily.csv` and `index_daily.csv` are all rebuilt
  automatically. Forecasts are keyed on the exact history and future dates, so editing
  `holidays.yaml` or fixing index data makes the next scan re-run Kronos. `index_daily.csv` is rebuilt from `indices/` by re-running `refresh`.
- **Back up regularly:** `journal.sqlite`. It is your trade history. Copy the file while the app
  is closed.

## When a download fails
- **What the error looks like:** `DataError` names the date and the URL.
- **Bad files are never cached:** each download is checked first (it must parse, and the index
  file must contain both Nifty 50 and India VIX). A bad file is thrown away. An index file
  cached without the VIX row is downloaded again on the next refresh.
- **Old corrupt files:** a file cached before this check existed that won't parse shows as
  "unknown bhavcopy format". Delete that one file and refresh.
- **What to do:**
  - Retry later.
  - If it keeps failing, check whether NSE has moved the file: open the URL from the error in a
    browser.

## Tests
- `uv run pytest`: the full fast suite, no network.
- `RUN_SLOW=1 uv run pytest`: also runs the real Kronos-small model, which downloads its weights once.
