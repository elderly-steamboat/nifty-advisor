from datetime import date

import numpy as np
import pandas as pd
import pytest

from advisor.analytics import enrich_chain, implied_move
from advisor.forecast import Forecast
from advisor.scoring import band_for, direction_score, premium_edge
from tests.conftest import make_chain, make_index_history

D, E = date(2026, 11, 18), date(2026, 12, 28)


def _fc(center, spread=300.0, n=401, spot=24000.0):
    term = np.linspace(center - 2 * spread, center + 2 * spread, n)
    closes = np.tile(term[:, None], (1, 5))
    return Forecast(D, E, spot, [D] * 5, closes, closes, closes, 1.0, {})


@pytest.fixture
def inputs(tmp_cfg):
    ch = make_chain(F=24000.0, days=40, d=D)
    e = enrich_chain(ch, tmp_cfg.risk_free_rate)
    hist = make_index_history(120, end=D, start_px=24000.0, vol=0.0)  # flat history
    return ch, e, hist, tmp_cfg


def test_contributions_sum_to_value(inputs):
    ch, e, hist, cfg = inputs
    s = direction_score(24000.0, 500.0, _fc(24100.0), e, hist, pd.Series([0.02] * 50), cfg)
    assert s.value == pytest.approx(100 * sum(c.contribution for c in s.components))
    assert [c.name for c in s.components] == ["kronos_drift", "oi", "trend", "skew"]
    assert all(c.weight == getattr(cfg.scoring.weights, c.name) for c in s.components)


def test_band_edges(tmp_cfg):
    assert band_for(19.99, tmp_cfg) == "neutral"
    assert band_for(20.0, tmp_cfg) == "mild_bullish"
    assert band_for(50.0, tmp_cfg) == "strong_bullish"
    assert band_for(-50.0, tmp_cfg) == "strong_bearish"
    assert band_for(-20.0, tmp_cfg) == "mild_bearish"


def test_drift_clipped(inputs):
    ch, e, hist, cfg = inputs
    s = direction_score(24000.0, 500.0, _fc(26000.0), e, hist, pd.Series([0.0] * 50), cfg)
    assert s.components[0].normalized == 1.0


def test_flat_inputs_near_zero(inputs):
    ch, e, hist, cfg = inputs
    rr_now = 0.0
    from advisor.analytics import risk_reversal_25d
    rr_now = risk_reversal_25d(e)
    s = direction_score(24000.0, 500.0, _fc(24000.0), e, hist, pd.Series([rr_now] * 50), cfg)
    assert s.components[0].normalized == pytest.approx(0.0, abs=1e-9)
    assert s.components[2].normalized == pytest.approx(0.0, abs=1e-9)
    assert s.components[3].normalized == pytest.approx(0.0, abs=1e-9)


def test_rich_puts_bearish(inputs):
    ch, e, hist, cfg = inputs
    s = direction_score(24000.0, 500.0, _fc(24000.0), e, hist, pd.Series([-0.05] * 50), cfg)
    assert s.components[3].normalized < 0


def test_uptrend_bullish(tmp_cfg):
    hist = make_index_history(120, end=D, start_px=20000.0, vol=0.0)
    hist["close"] = np.linspace(20000, 24000, 120)
    e = enrich_chain(make_chain(d=D), tmp_cfg.risk_free_rate)
    s = direction_score(24000.0, 500.0, _fc(24000.0), e, hist, pd.Series([0.0] * 50), tmp_cfg)
    assert s.components[2].normalized > 0.5


def test_edge_good_and_low(tmp_cfg):
    F, iv, t = 24000.0, 0.14, 40 / 365
    from advisor.analytics import implied_range
    lo, hi = implied_range(F, iv, t, 0.10)
    width = hi - lo
    # Build a forecast whose P10–P90 width is width / 1.10 → R = 1.10
    n = 1001
    target = width / 1.10
    term = np.linspace(F - target / 1.6, F + target / 1.6, n)  # P10–P90 of uniform = 0.8 × full range
    closes = np.tile(term[:, None], (1, 3))
    fc = Forecast(D, E, F, [D] * 3, closes, closes, closes, 1.0, {})
    good = premium_edge(F, iv, t, fc, 30.0, tmp_cfg)
    assert good.r == pytest.approx(1.10, rel=1e-3) and good.good
    assert not premium_edge(F, iv, t, fc, 29.9, tmp_cfg).good
    fc2 = Forecast(D, E, F, [D] * 3, closes * 0 + closes.mean() + (closes - closes.mean()) * 1.01,
                   closes, closes * 0 + closes.mean() + (closes - closes.mean()) * 1.01, 1.0, {})
    assert not premium_edge(F, iv, t, fc2, 80.0, tmp_cfg).good  # R ≈ 1.089
