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
