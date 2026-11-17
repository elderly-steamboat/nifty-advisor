import math

import numpy as np
import pandas as pd
import pytest

from advisor.analytics import (atm_iv, atm_strike, enrich_chain, find_walls, implied_move,
                               implied_range, iv_rank, max_pain, pcr, risk_reversal_25d,
                               strike_for_delta)
from tests.conftest import make_chain

R = 0.065


def test_implied_range_q10():
    lo, hi = implied_range(24000, 0.14, 40 / 365, 0.10)
    assert hi / 24000 == pytest.approx(math.exp(1.2816 * 0.14 * math.sqrt(40 / 365)), rel=1e-4)
    assert lo / 24000 == pytest.approx(math.exp(-1.2816 * 0.14 * math.sqrt(40 / 365)), rel=1e-4)


def test_implied_move():
    assert implied_move(24000, 0.14, 0.25) == pytest.approx(24000 * 0.14 * 0.5)


def test_enrich_recovers_smile(synthetic_chain):
    e = enrich_chain(synthetic_chain, R)
    for c in ["t", "iv", "delta", "gamma", "theta", "vega"]:
        assert c in e.columns
    atm = e[(e.strike == 24000) & (e.option_type == "CE")].iv.iloc[0]
    assert atm == pytest.approx(0.14, abs=2e-3)


def test_enrich_marks_bad_rows_nan(synthetic_chain):
    rows = synthetic_chain.rows
    rows.loc[(rows.strike == 23000) & (rows.option_type == "PE"), "oi"] = 0
    rows.loc[(rows.strike == 25000) & (rows.option_type == "CE"), "settle"] = 0
    rows.loc[(rows.strike == 21000) & (rows.option_type == "CE"), "settle"] = 1.0  # below intrinsic
    e = enrich_chain(synthetic_chain, R)
    for k, typ in ((23000, "PE"), (25000, "CE"), (21000, "CE")):
        assert math.isnan(e[(e.strike == k) & (e.option_type == typ)].iv.iloc[0])


def test_atm(synthetic_chain):
    e = enrich_chain(synthetic_chain, R)
    assert atm_strike(e, 24020.0, 50) == 24000.0
    assert atm_iv(e, 24000.0) == pytest.approx(0.14, abs=2e-3)


def test_walls_detected():
    ch = make_chain(spikes={(23500.0, "PE"): 900000, (24800.0, "CE"): 800000})
    w = find_walls(enrich_chain(ch, R), 24000.0, 0.10, 2.0, 3)
    assert w["put"][0] == 23500.0
    assert w["call"][0] == 24800.0


def test_no_spike_no_walls(synthetic_chain):
    w = find_walls(enrich_chain(synthetic_chain, R), 24000.0, 0.10, 2.0, 3)
    assert w == {"put": [], "call": []}


def test_walls_sorted_nearest_first():
    ch = make_chain(spikes={(23000.0, "PE"): 900000, (23600.0, "PE"): 700000})
    w = find_walls(enrich_chain(ch, R), 24000.0, 0.10, 2.0, 3)
    assert w["put"] == [23600.0, 23000.0]


def test_max_pain_known_profile():
    ch = make_chain()
    ch.rows["oi"] = 0.0
    ch.rows.loc[(ch.rows.strike == 24500) & (ch.rows.option_type == "CE"), "oi"] = 1e6
    ch.rows.loc[(ch.rows.strike == 23500) & (ch.rows.option_type == "PE"), "oi"] = 1e6
    mp = max_pain(ch.rows)
    assert 23500 <= mp <= 24500


def test_pcr_window():
    ch = make_chain()
    ch.rows.loc[ch.rows.option_type == "PE", "oi"] *= 2
    assert pcr(enrich_chain(ch, R), 24000.0, 0.10) == pytest.approx(2.0)


def test_strike_for_delta_otm(synthetic_chain):
    e = enrich_chain(synthetic_chain, R)
    kp = strike_for_delta(e, "PE", 0.20)
    kc = strike_for_delta(e, "CE", 0.20)
    assert kp < 24000 < kc
    dp = e[(e.strike == kp) & (e.option_type == "PE")].delta.iloc[0]
    assert abs(abs(dp) - 0.20) < 0.03


def test_risk_reversal_sign():
    e = enrich_chain(make_chain(skew=-0.5), R)
    assert risk_reversal_25d(e) > 0  # puts richer
    e0 = enrich_chain(make_chain(skew=0.0), R)
    assert risk_reversal_25d(e0) == pytest.approx(0.0, abs=2e-3)


def test_iv_rank_bounds():
    h = pd.Series(np.linspace(0.10, 0.20, 300))
    assert iv_rank(h, 0.20) == pytest.approx(100.0)
    assert iv_rank(h, 0.05) == 0.0
    assert iv_rank(pd.Series([0.15] * 10), 0.15) == 50.0
