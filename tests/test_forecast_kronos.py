from datetime import date

import numpy as np
import pytest
import yaml

from advisor.calendar import next_trading_days
from advisor.config import load_config
from advisor.forecast import forecast_seed, kronos_sampler
from tests.conftest import ROOT, make_index_history


def test_forecast_seed_deterministic_and_distinct():
    a = forecast_seed(1, date(2026, 11, 17), date(2026, 12, 29))
    assert a == forecast_seed(1, date(2026, 11, 17), date(2026, 12, 29))
    assert a != forecast_seed(1, date(2026, 11, 18), date(2026, 12, 29))
    assert 0 <= a < 2 ** 32


@pytest.fixture
def cfg_small(tmp_path):
    d = yaml.safe_load((ROOT / "config.yaml").read_text())
    d["kronos"]["model_id"] = "NeoQuasar/Kronos-small"
    d["kronos"]["n_paths"] = 8
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(d))
    return load_config(p)


@pytest.mark.slow
def test_kronos_deterministic(cfg_small):
    hist = make_index_history(300)
    dates5 = next_trading_days(hist.index[-1], 5, set())
    s = kronos_sampler(cfg_small)
    a = s(hist, dates5, 8, 123)
    b = s(hist, dates5, 8, 123)
    assert a.shape == (8, 5, 3)
    assert np.array_equal(a, b)
    assert not np.array_equal(a[0], a[1])
    c = s(hist, dates5, 8, 124)
    assert not np.array_equal(a, c)
    last = hist.close.iloc[-1]
    assert np.all(np.abs(a[:, 0, 2] / last - 1) < 0.15)  # first-day closes near last close
