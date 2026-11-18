"""Regression tests for the final-review findings (Important 1–6)."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from advisor.analytics import enrich_chain
from advisor.data import DataError, download_bhavcopy, download_index_close, data_status, parse_bhavcopy
from tests.conftest import FIX, make_chain, make_fake_sampler, make_index_history, uniform_forecast, write_index_history

D, E = date(2026, 11, 17), date(2026, 12, 29)
HOL = {date(2026, 11, 24), date(2026, 12, 25)}


class _Resp:
    def __init__(self, body, code=200):
        self.status_code, self.content = code, body


class _Session:
    def __init__(self, body):
        self.body, self.calls = body, 0

    def get(self, *a, **k):
        self.calls += 1
        return _Resp(self.body)


# ---- Important 1: bad downloads never poison the cache ----

def test_html_body_rejected_and_not_cached(tmp_cfg):
    with pytest.raises(DataError, match="invalid bhavcopy"):
        download_bhavcopy(date(2026, 11, 18), tmp_cfg.data_dir, session=_Session(b"<html>blocked</html>"))
    assert not data_status(date(2026, 11, 18), tmp_cfg.data_dir).has_bhavcopy
    assert list((tmp_cfg.data_dir / "bhavcopy").glob("*")) == []


def test_parse_missing_column_is_dataerror(tmp_path):
    import io, zipfile
    raw = pd.read_csv(FIX / "udiff_2026-11-17.csv.zip", compression="zip").drop(columns=["XpryDt"])
    p = tmp_path / "x.csv.zip"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("x.csv", raw.to_csv(index=False))
    with pytest.raises(DataError, match="unknown bhavcopy format"):
        parse_bhavcopy(p)


def test_parse_non_zip_is_dataerror(tmp_path):
    p = tmp_path / "x.csv.zip"
    p.write_bytes(b"<html></html>")
    with pytest.raises(DataError):
        parse_bhavcopy(p)


def test_index_file_without_vix_redownloaded(tmp_cfg):
    good = (FIX / "ind_close_all_2026-11-17.csv").read_bytes()
    no_vix = b"\n".join(l for l in good.split(b"\n") if b"India VIX" not in l)
    dest = tmp_cfg.data_dir / "indices" / "2026-11-17.csv"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(no_vix)
    s = _Session(good)
    download_index_close(D, tmp_cfg.data_dir, session=s)
    assert s.calls == 1 and data_status(D, tmp_cfg.data_dir).has_vix


# ---- Important 2: forecast cache keyed on horizon and history ----

def test_cache_misses_when_horizon_or_history_changes(tmp_cfg):
    from advisor.forecast import run_forecast
    write_index_history(tmp_cfg, make_index_history(600, end=D))
    calls = []
    s = make_fake_sampler(counter=calls)
    a = run_forecast(D, E, tmp_cfg, set(), sampler=s)
    b = run_forecast(D, E, tmp_cfg, {date(2026, 12, 1)}, sampler=s)  # new holiday → shorter horizon
    assert len(calls) == 2 and b.closes.shape[1] == a.closes.shape[1] - 1
    write_index_history(tmp_cfg, make_index_history(600, end=D, seed=99))  # corrected history
    run_forecast(D, E, tmp_cfg, set(), sampler=s)
    assert len(calls) == 3


# ---- Important 4: minimum history ----

def test_forecast_requires_full_context(tmp_cfg):
    from advisor.forecast import run_forecast
    write_index_history(tmp_cfg, make_index_history(100, end=D))
    with pytest.raises(DataError, match="512"):
        run_forecast(D, E, tmp_cfg, set(), sampler=make_fake_sampler())


def test_forecast_rejects_gaps(tmp_cfg):
    from advisor.forecast import run_forecast
    h = make_index_history(600, end=D)
    write_index_history(tmp_cfg, h.drop(h.index[-10]))
    with pytest.raises(DataError, match="gap"):
        run_forecast(D, E, tmp_cfg, set(), sampler=make_fake_sampler())


def test_recommend_short_history_not_ok(seeded_cfg):
    from advisor.engine import recommend
    rec = recommend(D, E, seeded_cfg, sampler=make_fake_sampler(), holidays=HOL, events=[])
    assert not rec.data_ok and "512" in rec.data_reason


def test_ivr_needs_200_days(seeded_long_cfg):
    from advisor.engine import iv_rank_for
    hist = pd.DataFrame({"date": ["2026-11-17"], "atm_iv": [0.13], "rr25": [0.01]})
    ivr, src, n = iv_rank_for(D, hist, seeded_long_cfg)
    assert src == "india_vix" and n >= 200
    short = pd.Series([15.0] * 50)
    with pytest.raises(DataError, match="IV history"):
        iv_rank_for(D, hist, seeded_long_cfg, vix=short)


# ---- Important 3: IV rank uses front-monthly IV on d, not the selected expiry ----

def test_ivr_uses_front_month_row(seeded_long_cfg):
    from advisor.engine import iv_rank_for
    days = pd.bdate_range(end="2026-11-17", periods=220)
    hist = pd.DataFrame({"date": [x.date().isoformat() for x in days],
                         "atm_iv": np.linspace(0.10, 0.20, 220), "rr25": 0.0})
    ivr, src, n = iv_rank_for(D, hist, seeded_long_cfg)  # ranks front-month IV on d (0.20 = max)
    assert src == "atm_iv" and ivr == pytest.approx(100.0) and n == 220


# ---- Important 5: wall-shift false alarms ----

def _ic(e, put, call, width=200.0):
    from advisor.strategy import Leg
    px = lambda t, k: float(e[(e.option_type == t) & (e.strike == k)].settle.iloc[0])
    return [Leg("PE", put - width, "BUY", 1, px("PE", put - width)), Leg("PE", put, "SELL", 1, px("PE", put)),
            Leg("CE", call, "SELL", 1, px("CE", call)), Leg("CE", call + width, "BUY", 1, px("CE", call + width))]


def test_wall_inside_at_entry_no_alarm(tmp_cfg):
    from advisor.risk import Position, assess, baseline_for
    from advisor.strategy import IRON_CONDOR
    ch = make_chain(d=date(2026, 11, 18), spikes={(23600.0, "PE"): 900000, (22800.0, "PE"): 800000})
    e = enrich_chain(ch, 0.065)
    fc = uniform_forecast(half=300, d=ch.date)
    legs = _ic(e, 23000.0, 25000.0)
    p = Position(1, IRON_CONDOR, ch.expiry, legs, sum(l.price if l.side == "SELL" else -l.price for l in legs), 0, {})
    p.baseline = baseline_for(p, ch, e, fc, tmp_cfg)
    r = assess(p, ch, e, fc, [], ch.date, tmp_cfg)
    assert not any(t.startswith("wall shift") for t in r.triggers), r.triggers


def test_new_wall_inside_still_alarms(tmp_cfg):
    from advisor.risk import Position, assess, baseline_for
    from advisor.strategy import IRON_CONDOR
    ch0 = make_chain(d=date(2026, 11, 18), spikes={(22800.0, "PE"): 800000})
    e0 = enrich_chain(ch0, 0.065)
    fc = uniform_forecast(half=300, d=ch0.date)
    legs = _ic(e0, 23000.0, 25000.0)
    p = Position(1, IRON_CONDOR, ch0.expiry, legs, sum(l.price if l.side == "SELL" else -l.price for l in legs), 0, {})
    p.baseline = baseline_for(p, ch0, e0, fc, tmp_cfg)
    ch1 = make_chain(d=date(2026, 11, 18), spikes={(22800.0, "PE"): 800000, (23500.0, "PE"): 1500000})
    r = assess(p, ch1, enrich_chain(ch1, 0.065), fc, [], ch1.date, tmp_cfg)
    assert any("inside" in t for t in r.triggers)


def test_roll_persists_tested_side_baseline(tmp_path, tmp_cfg):
    from advisor.journal import apply_roll, add_trade, get_open_positions, open_db
    from advisor.risk import Roll, rolled_baseline, apply_roll_to_position, baseline_for, Position
    from advisor.strategy import IRON_CONDOR, Leg
    ch = make_chain(d=date(2026, 11, 18))
    e = enrich_chain(ch, 0.065)
    fc = uniform_forecast(half=300, d=ch.date)
    legs = _ic(e, 23700.0, 25000.0)
    pos = Position(None, IRON_CONDOR, ch.expiry, legs, sum(l.price if l.side == "SELL" else -l.price for l in legs), 0, {})
    base = baseline_for(pos, ch, e, fc, tmp_cfg)
    conn = open_db(tmp_path / "j.sqlite")
    tid = add_trade(conn, IRON_CONDOR, ch.expiry, ch.date, legs, base, 65)
    [p] = get_open_positions(conn)
    new_legs = _ic(e, 23300.0, 25000.0)[:2]
    roll = Roll("put", [p.legs[0], p.legs[1]], new_legs, 1.0)
    nb = rolled_baseline(p, roll, ch, e, fc, tmp_cfg)
    assert nb["delta"]["put"] < base["delta"]["put"] and nb["delta"]["call"] == base["delta"]["call"]
    apply_roll(conn, tid, roll, ch.date, {p.legs[0].id: 1.0, p.legs[1].id: 2.0}, baseline=nb)
    [p2] = get_open_positions(conn)
    assert p2.baseline["delta"]["put"] == pytest.approx(nb["delta"]["put"])


# ---- Important 6: one bad position doesn't abort the others ----

def test_check_positions_survives_missing_chain(seeded_long_cfg, tmp_path):
    from advisor.engine import check_positions, log_trade
    from advisor.journal import add_trade, open_db
    from advisor.data import load_chain
    from advisor.strategy import IRON_CONDOR, Leg
    conn = open_db(tmp_path / "j.sqlite")
    add_trade(conn, IRON_CONDOR, date(2027, 1, 25), D, [Leg("PE", 24000.0, "SELL", 1, 50.0)], {}, 65)  # expiry not in fixture
    ch = load_chain(D, E, seeded_long_cfg)
    px = lambda t, k: float(ch.rows[(ch.rows.option_type == t) & (ch.rows.strike == k)].settle.iloc[0])
    legs = [Leg("PE", 24000.0, "BUY", 1, px("PE", 24000.0)), Leg("PE", 24200.0, "SELL", 1, px("PE", 24200.0)),
            Leg("CE", 26800.0, "SELL", 1, px("CE", 26800.0)), Leg("CE", 27000.0, "BUY", 1, px("CE", 27000.0))]
    log_trade(conn, D, IRON_CONDOR, E, legs, seeded_long_cfg, sampler=make_fake_sampler(), holidays=HOL)
    out = check_positions(conn, D, seeded_long_cfg, sampler=make_fake_sampler(), holidays=HOL, events=[])
    assert len(out) == 2
    bad = next(r for p, r, a in out if p.expiry == date(2027, 1, 25))
    assert bad.level == "UNKNOWN" and "data missing" in bad.triggers[0]
