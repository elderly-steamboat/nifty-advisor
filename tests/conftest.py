import shutil
from pathlib import Path

import pytest
import yaml

from advisor.config import load_config

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures"


@pytest.fixture
def tmp_cfg(tmp_path):
    """Repo config with data_dir pointed at an empty temp dir."""
    data = yaml.safe_load((ROOT / "config.yaml").read_text())
    data["data_dir"] = str(tmp_path / "data")
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(data))
    return load_config(p)


@pytest.fixture
def seeded_cfg(tmp_cfg):
    """tmp_cfg with the real 2026-11-17 and 2023-06-15 fixture files placed in the cache."""
    from advisor.data import ingest_index_file
    d = tmp_cfg.data_dir
    (d / "bhavcopy").mkdir(parents=True)
    shutil.copy(FIX / "udiff_2026-11-17.csv.zip", d / "bhavcopy" / "2026-11-17.csv.zip")
    shutil.copy(FIX / "legacy_2023-06-15.csv.zip", d / "bhavcopy" / "2023-06-15.csv.zip")
    for day in ("2026-11-17", "2023-06-15"):
        ingest_index_file(FIX / f"ind_close_all_{day}.csv", d)
    return tmp_cfg


# ---------- synthetic option chain ----------
import math as _m
from datetime import date as _date

import numpy as _np
import pandas as _pd


def make_chain(F=24000.0, iv_atm=0.14, skew=-0.5, days=40, d=_date(2026, 11, 18),
               spikes=None, r=0.065, lo=21000, hi=27000, step=50, lot_size=65):
    """Chain priced with Black-76 on a linear-in-log-moneyness smile.
    spikes: {(strike, 'CE'|'PE'): oi} overrides; base OI is a hump around F."""
    from advisor.analytics import black76_price
    from advisor.data import Chain
    from datetime import timedelta
    expiry = d + timedelta(days=days)
    t = days / 365.0
    rows = []
    for k in range(lo, hi + step, step):
        k = float(k)
        iv = max(0.05, iv_atm + skew * _m.log(k / F))
        for typ in ("CE", "PE"):
            price = black76_price(F, k, t, r, iv, typ)
            oi = 60000.0 + 200000.0 * _m.exp(-((k - F) / 1200.0) ** 2)
            if spikes and (k, typ) in spikes:
                oi = float(spikes[(k, typ)])
            rows.append({"date": d, "instrument": "OPT", "expiry": expiry, "strike": k,
                         "option_type": typ, "open": price, "high": price, "low": price,
                         "close": price, "settle": round(price, 2), "oi": oi, "chg_oi": 0.0,
                         "volume": 1000.0, "lot_size": lot_size})
    return Chain(d, expiry, F, F, lot_size, _pd.DataFrame(rows))


@pytest.fixture
def synthetic_chain():
    return make_chain()


# ---------- slow marker + synthetic index history ----------
import os as _os


def pytest_collection_modifyitems(config, items):
    if _os.environ.get("RUN_SLOW") == "1":
        return
    skip = pytest.mark.skip(reason="slow: set RUN_SLOW=1")
    for it in items:
        if "slow" in it.keywords:
            it.add_marker(skip)


def make_index_history(n=600, end=_date(2026, 11, 17), start_px=22000.0, vol=0.01, seed=7):
    """Synthetic daily OHLCV on weekdays ending at `end` (no holidays)."""
    from datetime import timedelta
    rng = _np.random.default_rng(seed)
    days, d = [], end
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d -= timedelta(days=1)
    days.reverse()
    closes = start_px * _np.exp(_np.cumsum(rng.normal(0, vol, n)))
    opens = _np.r_[closes[0], closes[:-1]]
    highs = _np.maximum(opens, closes) * (1 + _np.abs(rng.normal(0, vol / 2, n)))
    lows = _np.minimum(opens, closes) * (1 - _np.abs(rng.normal(0, vol / 2, n)))
    return _pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes,
                          "volume": _np.full(n, 3e8)}, index=days)


# ---------- fake Kronos sampler + index writer ----------

def make_fake_sampler(daily_vol=0.01, drift=0.0, counter=None):
    """GBM sampler with the Sampler signature; deterministic in seed."""
    def sample(history, future_dates, n_paths, seed):
        if counter is not None:
            counter.append(seed)
        rng = _np.random.default_rng(seed)
        h = len(future_dates)
        last = float(history.close.iloc[-1])
        rets = rng.normal(drift, daily_vol, (n_paths, h))
        close = last * _np.exp(_np.cumsum(rets, axis=1))
        wig = _np.abs(rng.normal(0, daily_vol / 3, (n_paths, h)))
        return _np.stack([close * (1 + wig), close * (1 - wig), close], axis=2)
    return sample


@pytest.fixture
def fake_sampler():
    return make_fake_sampler()


def write_index_history(cfg, hist, vix=15.0):
    """Write a synthetic history into cfg.data_dir/index_daily.csv."""
    from advisor.data import INDEX_COLS
    df = hist.copy()
    df.index.name = "date"
    df = df.reset_index()
    df["date"] = [d.isoformat() for d in df.date]
    for c in ("vix_open", "vix_high", "vix_low", "vix_close"):
        df[c] = vix
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    df[INDEX_COLS].to_csv(cfg.data_dir / "index_daily.csv", index=False)


def uniform_forecast(center=24000.0, half=1000.0, n=401, horizon=5, spot=None,
                     d=_date(2026, 11, 18), expiry=None):
    """Forecast whose terminal closes are uniform on [center−half, center+half]; paths flat."""
    from datetime import timedelta
    from advisor.forecast import Forecast
    term = _np.linspace(center - half, center + half, n)
    closes = _np.tile(term[:, None], (1, horizon))
    return Forecast(d, expiry or d + timedelta(days=40), spot or center, [d] * horizon,
                    closes.copy(), closes.copy(), closes, 1.0, {})


def cfg_with(cfg, section, **kw):
    """Copy of a frozen Config with one section's fields replaced."""
    return cfg.model_copy(update={section: getattr(cfg, section).model_copy(update=kw)})
