# Kronos price-path forecasting: deterministic sampling of N paths, calibration,
# quantiles and touch probabilities, with an on-disk cache of raw paths.
from __future__ import annotations

import hashlib
import sys
from datetime import date
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

VENDOR = Path(__file__).resolve().parents[1] / "vendor" / "kronos"
CHUNK = 50  # paths per Kronos batch (memory bound); fixed so results are deterministic

Sampler = Callable[[pd.DataFrame, list, int, int], np.ndarray]
_MODELS: dict = {}


def forecast_seed(base_seed: int, d: date, expiry: date) -> int:
    """Deterministic 32-bit seed for one forecast: sha256 of base_seed|date|expiry."""
    h = hashlib.sha256(f"{base_seed}|{d.isoformat()}|{expiry.isoformat()}".encode()).hexdigest()
    return int(h[:8], 16)


def _load_kronos(cfg):
    key = (cfg.kronos.model_id, cfg.kronos.tokenizer_id, cfg.kronos.device)
    if key not in _MODELS:
        if str(VENDOR) not in sys.path:
            sys.path.insert(0, str(VENDOR))
        from model import Kronos, KronosTokenizer  # vendored, pinned commit
        tok = KronosTokenizer.from_pretrained(cfg.kronos.tokenizer_id).to(cfg.kronos.device).eval()
        mdl = Kronos.from_pretrained(cfg.kronos.model_id).to(cfg.kronos.device).eval()
        _MODELS[key] = (tok, mdl)
    return _MODELS[key]


def _stamps(dates) -> np.ndarray:
    ts = pd.to_datetime(pd.Series(list(dates)))
    return np.stack([ts.dt.minute, ts.dt.hour, ts.dt.day_of_week, ts.dt.day, ts.dt.month],
                    axis=1).astype(np.float32)


def kronos_sampler(cfg) -> Sampler:
    """Sampler running Kronos on CPU: returns (n_paths, horizon, 3) array of (high, low, close).
    One torch seed per call; paths are a batch of identical histories with sample_count=1,
    so each row is an independent sample (Kronos's predict() would average them)."""
    import torch

    def sample(history: pd.DataFrame, future_dates: list, n_paths: int, seed: int) -> np.ndarray:
        if str(VENDOR) not in sys.path:
            sys.path.insert(0, str(VENDOR))
        from model.kronos import auto_regressive_inference
        tok, mdl = _load_kronos(cfg)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(seed)
        h = history[["open", "high", "low", "close", "volume"]].astype(float).copy()
        h["amount"] = h.volume * h[["open", "high", "low", "close"]].mean(axis=1)
        x = h.to_numpy(np.float32)
        mean, std = x.mean(axis=0), x.std(axis=0)
        xn = np.clip((x - mean) / (std + 1e-5), -5, 5).astype(np.float32)
        xs, ys = _stamps(history.index), _stamps(future_dates)
        horizon = len(future_dates)
        out = []
        for start in range(0, n_paths, CHUNK):
            b = min(CHUNK, n_paths - start)
            preds = auto_regressive_inference(
                tok, mdl,
                torch.from_numpy(np.repeat(xn[None], b, 0)).to(cfg.kronos.device),
                torch.from_numpy(np.repeat(xs[None], b, 0)).to(cfg.kronos.device),
                torch.from_numpy(np.repeat(ys[None], b, 0)).to(cfg.kronos.device),
                cfg.kronos.context, horizon, 5, cfg.kronos.temperature, 0,
                cfg.kronos.top_p, 1, False)
            preds = preds[:, -horizon:, :] * (std + 1e-5) + mean
            out.append(preds[:, :, [1, 2, 3]])
        return np.concatenate(out, axis=0).astype(np.float64)

    return sample


# ---------- forecast object, cache, statistics ----------

import math  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402

from scipy.stats import norm  # noqa: E402

from advisor.calendar import next_trading_days, trading_days_between  # noqa: E402
from advisor.data import DataError, load_index_history  # noqa: E402

FORECAST_VERSION = "1"  # bump when sampling code changes; part of the cache key


@dataclass
class Forecast:
    date: date
    expiry: date
    spot: float
    dates: list
    highs: np.ndarray   # (n_paths, horizon), calibrated
    lows: np.ndarray
    closes: np.ndarray
    k: float
    meta: dict = field(default_factory=dict)


def _cache_key(d: date, expiry: date, cfg, dates: list, hist: pd.DataFrame) -> str:
    """Key covers every input to the sampler: settings, the future dates (so holiday edits
    change it) and a hash of the exact history values (so corrected data changes it)."""
    kc = cfg.kronos
    hist_hash = hashlib.sha256(np.ascontiguousarray(hist.to_numpy(np.float64)).tobytes()
                               + ",".join(str(x) for x in hist.index).encode()).hexdigest()[:16]
    parts = [d.isoformat(), expiry.isoformat(), kc.model_id, kc.tokenizer_id, str(kc.context),
             str(kc.n_paths), str(kc.base_seed), repr(kc.temperature), repr(kc.top_p), FORECAST_VERSION,
             ",".join(x.isoformat() for x in dates), hist_hash]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:20]


def apply_calibration(raw_highs, raw_lows, raw_closes, k: float):
    """Scale every path's deviation from the per-step median close by k (§9 calibration)."""
    m = np.median(raw_closes, axis=0)
    f = lambda x: m + k * (x - m)
    return f(raw_highs), f(raw_lows), f(raw_closes)


def run_forecast(d: date, expiry: date, cfg, holidays: set, sampler: Sampler | None = None) -> Forecast:
    """Kronos forecast from close of d to expiry. Raw paths cached on disk; calibration
    multiplier cfg.calibration.k applied after loading. Config: kronos.*, calibration.k."""
    hist = load_index_history(d, cfg.kronos.context, cfg.data_dir)
    if hist.empty or hist.index[-1] != d:
        raise DataError(f"index history does not end on {d.isoformat()} — refresh data")
    if len(hist) < cfg.kronos.context:
        raise DataError(f"Kronos needs {cfg.kronos.context} index bars, have {len(hist)} — refresh more history")
    expected = trading_days_between(hist.index[0], d, holidays) + 1
    if expected != len(hist):
        raise DataError(f"index history has gaps: {expected - len(hist)} trading days missing "
                        f"between {hist.index[0]} and {d} — refresh data (or fix holidays.yaml)")
    horizon = trading_days_between(d, expiry, holidays)
    if horizon < 1:
        raise DataError(f"expiry {expiry} is not after {d}")
    dates = next_trading_days(d, horizon, holidays)
    cache = Path(cfg.data_dir) / "forecasts" / f"{_cache_key(d, expiry, cfg, dates, hist)}.npz"
    raw = np.load(cache)["paths"] if cache.exists() else None
    if raw is None or raw.shape[1] != horizon:
        sampler = sampler or kronos_sampler(cfg)
        raw = sampler(hist, dates, cfg.kronos.n_paths, forecast_seed(cfg.kronos.base_seed, d, expiry))
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(".part.npz")
        np.savez_compressed(tmp, paths=raw)
        tmp.replace(cache)
    k = cfg.calibration.k
    hi, lo, cl = apply_calibration(raw[:, :, 0], raw[:, :, 1], raw[:, :, 2], k)
    return Forecast(d, expiry, float(hist.close.iloc[-1]), dates, hi, lo, cl, k,
                    {"cache": cache.name, "model_id": cfg.kronos.model_id, "n_paths": int(raw.shape[0])})


def terminal_quantiles(fc: Forecast, qs: list[float]) -> dict[float, float]:
    """Quantiles of the final-day close across paths."""
    term = fc.closes[:, -1]
    return {q: float(np.quantile(term, q)) for q in qs}


def touch_prob(fc: Forecast, level: float, side: str, days: int | None) -> float:
    """Share of paths whose low ('below') / high ('above') reaches level within the first
    `days` steps (None = to expiry)."""
    n = fc.closes.shape[1] if days is None else min(days, fc.closes.shape[1])
    if side == "below":
        return float(np.mean(fc.lows[:, :n].min(axis=1) <= level))
    return float(np.mean(fc.highs[:, :n].max(axis=1) >= level))


def iv_touch_prob(F: float, K: float, iv: float, t: float) -> float:
    """Reflection-principle touch probability: min(1, 2·N(−|ln(K/F)|/(iv·√t)))."""
    if t <= 0 or iv <= 0 or not np.isfinite(iv):
        return 1.0 if K == F else 0.0
    return float(min(1.0, 2 * norm.cdf(-abs(math.log(K / F)) / (iv * math.sqrt(t)))))


def expected_payoff(fc: Forecast, payoff_fn: Callable[[np.ndarray], np.ndarray]) -> float:
    """Mean payoff over final-day closes."""
    return float(np.mean(payoff_fn(fc.closes[:, -1])))
