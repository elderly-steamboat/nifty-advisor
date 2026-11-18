# Direction score (−100 bearish … +100 bullish) and premium edge (§10.1, §10.2).
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from advisor.analytics import find_walls, implied_range, pcr, risk_reversal_25d
from advisor.forecast import Forecast, terminal_quantiles


@dataclass
class Component:
    name: str
    raw: float
    normalized: float
    weight: float
    contribution: float


@dataclass
class DirectionScore:
    value: float
    band: str
    components: list[Component]


@dataclass
class Edge:
    r: float
    ivr: float
    good: bool
    implied_width: float
    kronos_width: float


def _clip(x: float) -> float:
    return float(np.clip(x, -1.0, 1.0)) if np.isfinite(x) else 0.0


def band_for(value: float, cfg) -> str:
    """Band from |value|: < neutral_band neutral, < strong_band mild, else strong. Config: scoring.*_band."""
    a = abs(value)
    if a < cfg.scoring.neutral_band:
        return "neutral"
    side = "bullish" if value > 0 else "bearish"
    return f"{'mild' if a < cfg.scoring.strong_band else 'strong'}_{side}"


def _oi_component(spot: float, enriched: pd.DataFrame, cfg) -> tuple[float, float]:
    s = cfg.scoring
    p = pcr(enriched, spot, s.oi_window_pct)
    pcr_s = _clip(2 * (p - s.pcr_low) / (s.pcr_high - s.pcr_low) - 1)
    walls = find_walls(enriched, spot, s.oi_window_pct, cfg.walls.oi_multiple, cfg.walls.top_n)
    w = enriched[(enriched.strike >= spot * (1 - s.oi_window_pct)) &
                 (enriched.strike <= spot * (1 + s.oi_window_pct))]

    def nearest(side, typ, cond):
        if walls[side]:
            return walls[side][0]
        sub = w[(w.option_type == typ) & cond]
        return float(sub.loc[sub.oi.idxmax(), "strike"]) if not sub.empty else math.nan

    put = nearest("put", "PE", w.strike < spot)
    call = nearest("call", "CE", w.strike > spot)
    d_put, d_call = spot - put, call - spot
    asym = _clip((d_call - d_put) / (d_call + d_put)) if np.isfinite(d_put + d_call) and d_put + d_call > 0 else 0.0
    return p, (pcr_s + asym) / 2


def _trend_component(index_hist: pd.DataFrame, cfg) -> tuple[float, float]:
    c = index_hist.close.astype(float)
    a, b, s20 = cfg.scoring.trend_scales
    last = float(c.iloc[-1])
    parts = []
    if len(c) >= 20:
        parts.append(_clip((last / c.tail(20).mean() - 1) / a))
    if len(c) >= 50:
        parts.append(_clip((last / c.tail(50).mean() - 1) / b))
    if len(c) >= 21:
        parts.append(_clip((last / float(c.iloc[-21]) - 1) / s20))
    raw = last / c.tail(20).mean() - 1 if len(c) >= 20 else math.nan
    return raw, float(np.mean(parts)) if parts else 0.0


def direction_score(spot: float, implied_move: float, fc: Forecast, enriched: pd.DataFrame,
                    index_hist: pd.DataFrame, rr_history: pd.Series, cfg) -> DirectionScore:
    """Weighted direction score (§10.1). Config: scoring.weights, pcr_low/high, oi_window_pct,
    trend_scales, skew_scale, neutral_band, strong_band; walls.oi_multiple/top_n."""
    w = cfg.scoring.weights
    p50 = terminal_quantiles(fc, [0.5])[0.5]
    drift_raw = (p50 - spot) / implied_move if implied_move > 0 else 0.0
    pcr_raw, oi_n = _oi_component(spot, enriched, cfg)
    trend_raw, trend_n = _trend_component(index_hist, cfg)
    rr = risk_reversal_25d(enriched)
    hist = pd.Series(rr_history).dropna().tail(252)
    med = float(hist.median()) if len(hist) else rr
    skew_raw = rr - med if np.isfinite(rr) else math.nan
    skew_n = -_clip(skew_raw / cfg.scoring.skew_scale) if np.isfinite(skew_raw) else 0.0
    comps = []
    for name, raw, n in (("kronos_drift", drift_raw, _clip(drift_raw)), ("oi", pcr_raw, oi_n),
                         ("trend", trend_raw, trend_n), ("skew", skew_raw, skew_n)):
        wt = getattr(w, name)
        n = n + 0.0  # normalise -0.0
        comps.append(Component(name, float(raw), n, wt, wt * n))
    value = 100.0 * sum(c.contribution for c in comps)
    return DirectionScore(value, band_for(value, cfg), comps)


def premium_edge(F: float, atm_iv: float, t: float, fc: Forecast, ivr: float, cfg) -> Edge:
    """R = implied P10–P90 width / Kronos P10–P90 width; good if R ≥ r_good and IVR ≥ ivr_good (§10.2)."""
    lo, hi = implied_range(F, atm_iv, t, 0.10)
    q = terminal_quantiles(fc, [0.10, 0.90])
    kw = q[0.90] - q[0.10]
    r = round((hi - lo) / kw, 6) if kw > 0 else math.inf  # rounded: thresholds compare on 6 dp, as in JSON
    good = r >= cfg.edge.r_good and ivr >= cfg.edge.ivr_good
    return Edge(float(r), float(ivr), bool(good), float(hi - lo), float(kw))
