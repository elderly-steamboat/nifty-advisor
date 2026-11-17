# Nifty Options Advisor

Deterministic, local advisor for selling Nifty 50 monthly options. Combines a Kronos
price-range forecast with option-chain analytics (IV, OI walls, PCR, skew) and recommends
a trade (or no trade), with legs, sizing, an exit plan, and early risk warnings on open trades.

Design and plans: `~/Documents/trading/specs/`.

Status: core engine complete (Plan 1). Mac app: Plan 2. Backtest + calibration: Plan 3.

## Install

```bash
git clone --recurse-submodules <this repo>   # or: git submodule update --init
uv sync                                       # Python 3.12, pinned deps
uv run pytest
```

Kronos is vendored at `vendor/kronos`, pinned to commit `67b630e`.

## Docs

- `docs/architecture.md`: which file does what, the data flow, and where files are stored
- `docs/rules.md`: every trading rule in plain English, with its config keys
- `docs/config.md`: every setting
- `docs/runbook.md`: daily use, yearly upkeep, NSE changes, Kronos upgrades, backups

## Quick use (Python)

```python
from datetime import date
from advisor.config import load_config
from advisor.engine import recommend, recommendation_json
cfg = load_config()
print(recommendation_json(recommend(date(2026, 11, 17), date(2026, 12, 29), cfg)))
```
