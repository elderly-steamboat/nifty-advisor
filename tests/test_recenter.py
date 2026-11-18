"""Option A: Kronos supplies only the spread; the centre is the futures price."""
from datetime import date

import numpy as np
import pytest

from advisor.engine import recommend
from advisor.forecast import run_forecast, terminal_quantiles
from tests.conftest import cfg_with, make_fake_sampler, make_index_history, write_index_history

D, E = date(2026, 11, 17), date(2026, 12, 29)
HOL = {date(2026, 11, 24), date(2026, 12, 25)}


@pytest.fixture
def cfg_hist(tmp_cfg):
    write_index_history(tmp_cfg, make_index_history(600, end=D))
    return tmp_cfg


def test_recenter_puts_every_step_median_on_spot_to_future_path(cfg_hist):
    drifty = make_fake_sampler(daily_vol=0.01, drift=-0.003)  # Kronos-like downward lean
    fc = run_forecast(D, E, cfg_hist, set(), sampler=drifty, center=26000.0)
    med = np.median(fc.closes, axis=0)
    h = med.size
    target = fc.spot * (26000.0 / fc.spot) ** (np.arange(1, h + 1) / h)
    assert np.allclose(med, target, rtol=1e-9)
    assert fc.meta["center"] == 26000.0


def test_recenter_keeps_relative_spread(cfg_hist):
    drifty = make_fake_sampler(daily_vol=0.01, drift=-0.003)
    raw = run_forecast(D, E, cfg_hist, set(), sampler=drifty)
    cen = run_forecast(D, E, cfg_hist, set(), sampler=drifty, center=26000.0)
    qr, qc = terminal_quantiles(raw, [0.1, 0.5, 0.9]), terminal_quantiles(cen, [0.1, 0.5, 0.9])
    assert qc[0.9] / qc[0.1] == pytest.approx(qr[0.9] / qr[0.1], rel=1e-9)
    assert qc[0.5] == pytest.approx(26000.0, rel=1e-9)
    assert np.all(cen.lows <= cen.closes) and np.all(cen.closes <= cen.highs)


def test_recenter_off_in_config_keeps_raw_centre(cfg_hist):
    cfg = cfg_with(cfg_hist, "kronos", recenter=False)
    drifty = make_fake_sampler(daily_vol=0.01, drift=-0.003)
    a = run_forecast(D, E, cfg, set(), sampler=drifty)
    b = run_forecast(D, E, cfg, set(), sampler=drifty, center=26000.0)
    assert np.array_equal(a.closes, b.closes) and "center" not in b.meta


def test_recommend_centres_on_future_and_ignores_kronos_drift(seeded_long_cfg):
    drifty = make_fake_sampler(daily_vol=0.01, drift=-0.003)
    rec = recommend(D, E, seeded_long_cfg, sampler=drifty, holidays=HOL, events=[])
    assert rec.data_ok, rec.data_reason
    assert rec.quantiles["p50"] == pytest.approx(rec.future, rel=1e-9)
    kd = next(c for c in rec.direction.components if c.name == "kronos_drift")
    assert kd.weight == 0.0
