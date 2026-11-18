from datetime import date

import numpy as np
import pytest

from advisor.calendar import trading_days_between
from advisor.data import DataError
from advisor.forecast import (Forecast, apply_calibration, expected_payoff, iv_touch_prob,
                              run_forecast, terminal_quantiles, touch_prob)
from tests.conftest import make_fake_sampler, make_index_history, write_index_history

D, E = date(2026, 11, 17), date(2026, 12, 29)


@pytest.fixture
def cfg_hist(tmp_cfg):
    write_index_history(tmp_cfg, make_index_history(600, end=D))
    return tmp_cfg


def test_horizon_matches_trading_days(cfg_hist, fake_sampler):
    hol = {date(2026, 11, 24), date(2026, 12, 25)}
    fc = run_forecast(D, E, cfg_hist, hol, sampler=fake_sampler)
    assert fc.closes.shape == (cfg_hist.kronos.n_paths, trading_days_between(D, E, hol))
    assert fc.dates[0] == date(2026, 11, 18) and date(2026, 11, 24) not in fc.dates
    assert fc.dates[-1] == E


def test_cache_hit_skips_sampler(cfg_hist):
    calls = []
    s = make_fake_sampler(counter=calls)
    a = run_forecast(D, E, cfg_hist, set(), sampler=s)
    b = run_forecast(D, E, cfg_hist, set(), sampler=s)
    assert len(calls) == 1
    assert np.array_equal(a.closes, b.closes)


def test_deterministic_with_fake_sampler(cfg_hist, tmp_path, fake_sampler):
    a = run_forecast(D, E, cfg_hist, set(), sampler=fake_sampler)
    for f in (cfg_hist.data_dir / "forecasts").glob("*"):
        f.unlink()
    b = run_forecast(D, E, cfg_hist, set(), sampler=fake_sampler)
    assert np.array_equal(a.closes, b.closes) and np.array_equal(a.highs, b.highs)


def test_run_forecast_requires_history_on_date(cfg_hist, fake_sampler):
    with pytest.raises(DataError):
        run_forecast(date(2026, 11, 18), E, cfg_hist, set(), sampler=fake_sampler)


def test_calibration_k1_identity():
    rng = np.random.default_rng(0)
    c = rng.normal(100, 5, (50, 4))
    h, l, c2 = apply_calibration(c + 1, c - 1, c, 1.0)
    assert np.array_equal(c2, c) and np.array_equal(h, c + 1)


def test_calibration_k2_doubles_spread_around_median():
    rng = np.random.default_rng(0)
    c = rng.normal(100, 5, (51, 3))
    _, _, c2 = apply_calibration(c, c, c, 2.0)
    m = np.median(c, axis=0)
    assert np.allclose(c2 - m, 2 * (c - m))
    assert np.allclose(np.median(c2, axis=0), m)


def _fc(closes, highs=None, lows=None, spot=100.0):
    closes = np.asarray(closes, float)
    return Forecast(D, E, spot, [D] * closes.shape[1], closes if highs is None else highs,
                    closes if lows is None else lows, closes, 1.0, {})


def test_terminal_quantiles():
    fc = _fc(np.arange(1, 102, dtype=float)[:, None])
    q = terminal_quantiles(fc, [0.1, 0.5, 0.9])
    assert q[0.5] == pytest.approx(51.0) and q[0.1] == pytest.approx(11.0)


def test_touch_prob_monotonic_and_window():
    rng = np.random.default_rng(1)
    paths = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, (400, 20)), axis=1))
    fc = _fc(paths)
    assert touch_prob(fc, 95, "below", None) > touch_prob(fc, 92, "below", None)
    assert touch_prob(fc, 95, "below", 3) <= touch_prob(fc, 95, "below", None)
    assert touch_prob(fc, 105, "above", None) > 0
    assert touch_prob(fc, 1000, "above", None) == 0.0


def test_iv_touch_prob():
    assert iv_touch_prob(100, 100, 0.2, 0.1) == 1.0
    assert 0 < iv_touch_prob(100, 90, 0.2, 0.1) < iv_touch_prob(100, 95, 0.2, 0.1) < 1


def test_expected_payoff():
    fc = _fc(np.array([[90.0], [100.0], [110.0]]))
    assert expected_payoff(fc, lambda s: np.maximum(s - 100, 0)) == pytest.approx(10 / 3)
