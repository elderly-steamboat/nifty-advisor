# Option analytics: Black-76 pricing/IV/greeks and option-chain metrics
# (ATM IV, implied range, PCR, OI walls, max pain, 25-delta skew, IV rank).
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

IV_LO, IV_HI = 0.01, 3.0


def _d1d2(F, K, t, sigma):
    st = sigma * np.sqrt(t)
    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * t) / st
    return d1, d1 - st


def black76_price(F, K, t, r, sigma, option_type: str):
    """Black-76 price of a European option on a forward. t in years (calendar days/365).
    At t <= 0 returns undiscounted intrinsic. Accepts scalars or numpy arrays."""
    F, K, sigma = np.asarray(F, float), np.asarray(K, float), np.asarray(sigma, float)
    call = option_type == "CE"
    if t <= 0:
        out = np.maximum(F - K, 0.0) if call else np.maximum(K - F, 0.0)
        return float(out) if out.ndim == 0 else out
    d1, d2 = _d1d2(F, K, t, sigma)
    disc = math.exp(-r * t)
    out = disc * (F * norm.cdf(d1) - K * norm.cdf(d2)) if call \
        else disc * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
    return float(out) if np.ndim(out) == 0 else out


def black76_greeks(F, K, t, r, sigma, option_type: str) -> dict:
    """delta, gamma, theta (per calendar day), vega (per 1.00 vol) under Black-76."""
    t = max(t, 1e-8)
    d1, d2 = _d1d2(F, K, t, sigma)
    disc = math.exp(-r * t)
    pdf = norm.pdf(d1)
    call = option_type == "CE"
    delta = disc * norm.cdf(d1) if call else -disc * norm.cdf(-d1)
    gamma = disc * pdf / (F * sigma * math.sqrt(t))
    vega = disc * F * pdf * math.sqrt(t)
    price = black76_price(F, K, t, r, sigma, option_type)
    theta_yr = -disc * F * pdf * sigma / (2 * math.sqrt(t)) + r * price
    return {"delta": float(delta), "gamma": float(gamma),
            "theta": float(theta_yr / 365.0), "vega": float(vega)}


def implied_vol(price, F, K, t, r, option_type: str) -> float:
    """Black-76 implied vol via Brent on [0.01, 3.0]; NaN if price <= 0, at/below
    discounted intrinsic, or no root in range."""
    if t <= 0 or price is None or not np.isfinite(price) or price <= 0:
        return math.nan
    disc = math.exp(-r * t)
    intrinsic = disc * max(F - K, 0.0) if option_type == "CE" else disc * max(K - F, 0.0)
    if price <= intrinsic + 1e-9:
        return math.nan
    f = lambda s: black76_price(F, K, t, r, s, option_type) - price
    lo, hi = f(IV_LO), f(IV_HI)
    if lo * hi > 0:
        return math.nan
    return float(brentq(f, IV_LO, IV_HI, xtol=1e-10))


# ---------- chain metrics ----------

import pandas as pd  # noqa: E402


def enrich_chain(chain, r: float) -> pd.DataFrame:
    """Copy of chain.rows with t, iv, delta, gamma, theta, vega from settle prices.
    Rows with oi == 0, settle <= 0, or failed IV inversion get NaN iv/greeks."""
    e = chain.rows.copy()
    t = max((chain.expiry - chain.date).days, 0) / 365.0
    e["t"] = t
    F = chain.future
    ivs, greeks = [], []
    for k, typ, px, oi in zip(e.strike, e.option_type, e.settle, e.oi):
        iv = implied_vol(px, F, k, t, r, typ) if oi > 0 and px > 0 else math.nan
        ivs.append(iv)
        greeks.append(black76_greeks(F, k, t, r, iv, typ) if np.isfinite(iv)
                      else dict(delta=math.nan, gamma=math.nan, theta=math.nan, vega=math.nan))
    e["iv"] = ivs
    for g in ("delta", "gamma", "theta", "vega"):
        e[g] = [x[g] for x in greeks]
    return e


def atm_strike(rows: pd.DataFrame, F: float, step: float) -> float:
    """Listed strike nearest to F (ties → lower strike)."""
    strikes = np.sort(rows.strike.dropna().unique())
    return float(strikes[np.argmin(np.abs(strikes - F))])


def atm_iv(enriched: pd.DataFrame, F: float) -> float:
    """Mean of CE and PE IV at the ATM strike (NaNs ignored)."""
    k = atm_strike(enriched, F, 0)
    return float(enriched[enriched.strike == k].iv.mean())


def implied_move(F: float, iv: float, t: float) -> float:
    """One-sigma move to expiry: F·iv·√t."""
    return F * iv * math.sqrt(t)


def implied_range(F: float, iv: float, t: float, q: float) -> tuple[float, float]:
    """Market-implied (q, 1−q) range: F·exp(∓z·iv·√t), z = Φ⁻¹(1−q)."""
    z = norm.ppf(1 - q)
    s = iv * math.sqrt(t)
    return F * math.exp(-z * s), F * math.exp(z * s)


def strike_for_delta(enriched: pd.DataFrame, option_type: str, target_abs_delta: float) -> float:
    """OTM strike (PE below F / CE above F, F inferred from |Δ|<0.5) whose |delta| is closest to target."""
    side = enriched[(enriched.option_type == option_type) & enriched.iv.notna()]
    side = side[side.delta.abs() <= 0.5]
    if side.empty:
        return math.nan
    i = (side.delta.abs() - target_abs_delta).abs().to_numpy().argmin()
    return float(side.strike.iloc[i])


def _window(enriched, spot, window_pct):
    return enriched[(enriched.strike >= spot * (1 - window_pct)) &
                    (enriched.strike <= spot * (1 + window_pct))]


def pcr(enriched: pd.DataFrame, spot: float, window_pct: float) -> float:
    """Put OI / call OI over strikes within ±window_pct of spot."""
    w = _window(enriched, spot, window_pct)
    c = w[w.option_type == "CE"].oi.sum()
    return float(w[w.option_type == "PE"].oi.sum() / c) if c > 0 else math.nan


def find_walls(enriched: pd.DataFrame, spot: float, window_pct: float,
               oi_multiple: float, top_n: int) -> dict:
    """OI walls (§11): puts below spot / calls above spot within the window whose OI is
    ≥ oi_multiple × that side's median OI and among the side's top_n by OI. Nearest first."""
    w = _window(enriched, spot, window_pct)
    out = {}
    for side, typ, cond in (("put", "PE", w.strike < spot), ("call", "CE", w.strike > spot)):
        s = w[(w.option_type == typ) & cond]
        if s.empty:
            out[side] = []
            continue
        med = s.oi.median()
        top = set(s.nlargest(top_n, "oi").strike)
        walls = [k for k, oi in zip(s.strike, s.oi) if k in top and oi >= oi_multiple * med]
        out[side] = sorted(walls, key=lambda k: abs(k - spot))
    return out


def max_pain(rows: pd.DataFrame) -> float:
    """Strike minimising total intrinsic payout to option holders at expiry."""
    strikes = np.sort(rows.strike.unique())
    ce = rows[rows.option_type == "CE"]
    pe = rows[rows.option_type == "PE"]
    pay = [(ce.oi * np.maximum(s - ce.strike, 0)).sum() + (pe.oi * np.maximum(pe.strike - s, 0)).sum()
           for s in strikes]
    return float(strikes[int(np.argmin(pay))])


def risk_reversal_25d(enriched: pd.DataFrame) -> float:
    """IV(25Δ put) − IV(25Δ call). Positive = puts richer."""
    kp = strike_for_delta(enriched, "PE", 0.25)
    kc = strike_for_delta(enriched, "CE", 0.25)
    ivp = enriched[(enriched.strike == kp) & (enriched.option_type == "PE")].iv
    ivc = enriched[(enriched.strike == kc) & (enriched.option_type == "CE")].iv
    if ivp.empty or ivc.empty:
        return math.nan
    return float(ivp.iloc[0] - ivc.iloc[0])


def iv_rank(history: pd.Series, current: float) -> float:
    """Percent position of current within the trailing 252-value min–max range (50 if flat)."""
    h = pd.Series(history).dropna().tail(252)
    lo, hi = float(h.min()), float(h.max())
    if not np.isfinite(lo) or hi - lo <= 1e-12:
        return 50.0
    return float(np.clip((current - lo) / (hi - lo) * 100.0, 0.0, 100.0))
