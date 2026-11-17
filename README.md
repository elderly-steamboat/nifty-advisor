# Nifty Options Advisor

Deterministic, local advisor for selling Nifty 50 monthly options. Combines a Kronos
price-range forecast with option-chain analytics (IV, OI walls, PCR, skew) and recommends
a trade (or no trade), with legs, sizing, an exit plan, and early risk warnings on open trades.

Design and plans: `~/Documents/trading/specs/`.

Status: core engine in progress (Plan 1).

## Install

```bash
git clone --recurse-submodules <this repo>   # or: git submodule update --init
uv sync                                       # Python 3.12, pinned deps
uv run pytest
```

Kronos is vendored at `vendor/kronos`, pinned to commit `67b630e`.
