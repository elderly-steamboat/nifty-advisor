import math

import numpy as np
import pytest

from advisor.analytics import black76_greeks, black76_price, implied_vol

F, K, T, R = 24000.0, 24000.0, 40 / 365, 0.065


def test_put_call_parity():
    for k in (22000.0, 24000.0, 26000.0):
        c = black76_price(F, k, T, R, 0.15, "CE")
        p = black76_price(F, k, T, R, 0.15, "PE")
        assert c - p == pytest.approx(math.exp(-R * T) * (F - k), abs=1e-9)


def test_iv_roundtrip():
    for k, typ in ((23000.0, "PE"), (24000.0, "CE"), (25200.0, "CE")):
        price = black76_price(F, k, T, R, 0.14, typ)
        assert implied_vol(price, F, k, T, R, typ) == pytest.approx(0.14, abs=1e-6)


def test_iv_nan_below_intrinsic():
    intrinsic = math.exp(-R * T) * (F - 23000.0)
    assert math.isnan(implied_vol(intrinsic - 5, F, 23000.0, T, R, "CE"))
    assert math.isnan(implied_vol(0.0, F, 25000.0, T, R, "CE"))


def test_delta_signs_and_atm():
    g_c = black76_greeks(F, K, T, R, 0.15, "CE")
    g_p = black76_greeks(F, K, T, R, 0.15, "PE")
    disc = math.exp(-R * T)
    assert g_c["delta"] == pytest.approx(0.5 * disc, abs=0.02)
    assert g_p["delta"] == pytest.approx(-0.5 * disc, abs=0.02)
    assert g_c["gamma"] > 0 and g_c["vega"] > 0 and g_c["theta"] < 0


def test_vectorised_price():
    ks = np.array([23000.0, 24000.0, 25000.0])
    out = black76_price(F, ks, T, R, np.array([0.15, 0.14, 0.13]), "CE")
    assert out.shape == (3,) and out[0] > out[1] > out[2]


def test_expired_option_is_intrinsic():
    assert black76_price(F, 23000.0, 0.0, R, 0.15, "CE") == pytest.approx(1000.0)
    assert black76_price(F, 25000.0, 0.0, R, 0.15, "CE") == 0.0
