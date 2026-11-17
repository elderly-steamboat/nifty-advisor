from datetime import date

import pytest

from advisor.analytics import enrich_chain
from advisor.config import Event
from advisor.risk import (Position, assess, danger_levels, level_from_metrics, mark, pnl,
                          pre_entry_check, stress)
from advisor.strategy import IRON_BUTTERFLY, IRON_CONDOR, NAKED_STRANGLE, BULL_PUT, Leg, build_candidate
from tests.conftest import make_chain, uniform_forecast

D = date(2026, 11, 18)
R = 0.065


def _px(e, typ, k):
    return float(e[(e.option_type == typ) & (e.strike == k)].settle.iloc[0])


def condor(e, put=23000.0, call=25000.0, width=200.0, credit=None):
    legs = [Leg("PE", put - width, "BUY", 1, _px(e, "PE", put - width)),
            Leg("PE", put, "SELL", 1, _px(e, "PE", put)),
            Leg("CE", call, "SELL", 1, _px(e, "CE", call)),
            Leg("CE", call + width, "BUY", 1, _px(e, "CE", call + width))]
    nc = credit if credit is not None else sum(l.price if l.side == "SELL" else -l.price for l in legs)
    return Position(1, IRON_CONDOR, D, legs, nc, 0, {})


@pytest.fixture
def env(tmp_cfg):
    ch = make_chain(d=D)
    return ch, enrich_chain(ch, R), tmp_cfg


def side(touch=0.01, delta=0.10, frac=None):
    return {"touch_3d": touch, "delta": delta, "frac": frac}


def base_metrics(**kw):
    m = {"sides": {"put": side(), "call": side()}, "loss_mult": 0.0, "gap_loss_mult": 0.0,
         "event": None, "wall_shift": None}
    m.update(kw)
    return m


def test_green_when_far_otm(env):
    ch, e, cfg = env
    pos = condor(e)
    r = assess(pos, ch, e, uniform_forecast(half=300, d=D), [], D, cfg)
    assert r.level == "GREEN", r.triggers


def test_amber_needs_both_absolute_and_relative(tmp_cfg):
    m = base_metrics(sides={"put": side(touch=0.18), "call": side()})
    b = {"touch_3d": {"put": 0.15, "call": 0.01}, "delta": {"put": 0.10, "call": 0.10}}
    assert level_from_metrics(m, b, IRON_CONDOR, tmp_cfg)[0] == "GREEN"
    b["touch_3d"]["put"] = 0.10
    lvl, trig = level_from_metrics(m, b, IRON_CONDOR, tmp_cfg)
    assert lvl == "AMBER" and any(t.startswith("touch_3d put") for t in trig)


def test_credit_spread_relative_delta(tmp_cfg):
    b = {"touch_3d": {"put": 0.05}, "delta": {"put": 0.28}}
    def lvl(dl):
        m = base_metrics(sides={"put": side(delta=dl)})
        return level_from_metrics(m, b, BULL_PUT, tmp_cfg)[0]
    assert lvl(0.35) == "GREEN"
    assert lvl(0.37) == "AMBER"
    assert lvl(0.44) == "RED"


def test_butterfly_uses_frac_not_delta(tmp_cfg):
    b = {"touch_3d": {"put": 0.3, "call": 0.3}, "delta": {"put": 0.5, "call": 0.5}}
    m = base_metrics(sides={"put": side(touch=0.3, delta=0.5, frac=0.1), "call": side(touch=0.3, delta=0.5, frac=0.0)})
    assert level_from_metrics(m, b, IRON_BUTTERFLY, tmp_cfg)[0] == "GREEN"
    m["sides"]["put"]["frac"] = 0.6
    assert level_from_metrics(m, b, IRON_BUTTERFLY, tmp_cfg)[0] == "AMBER"
    m["sides"]["put"]["frac"] = 0.85
    assert level_from_metrics(m, b, IRON_BUTTERFLY, tmp_cfg)[0] == "RED"


def test_red_on_loss_multiple(tmp_cfg):
    assert level_from_metrics(base_metrics(loss_mult=1.6), {}, IRON_CONDOR, tmp_cfg)[0] == "RED"
    assert level_from_metrics(base_metrics(loss_mult=1.4), {}, IRON_CONDOR, tmp_cfg)[0] == "GREEN"


def test_gap_amber(tmp_cfg):
    assert level_from_metrics(base_metrics(gap_loss_mult=1.2), {}, IRON_CONDOR, tmp_cfg)[0] == "AMBER"


def test_event_amber(env):
    ch, e, cfg = env
    ev = [Event(date(2026, 11, 20), "RBI MPC", "rbi")]
    r = assess(condor(e), ch, e, uniform_forecast(half=300, d=D), ev, D, cfg)
    assert r.level == "AMBER" and any("event" in t for t in r.triggers)


def test_wall_shift_amber(tmp_cfg):
    m = base_metrics(wall_shift="call wall OI 900000 → 500000")
    assert level_from_metrics(m, {}, IRON_CONDOR, tmp_cfg)[0] == "AMBER"


def test_wall_oi_drop_detected(tmp_cfg):
    ch0 = make_chain(d=D, spikes={(25200.0, "CE"): 900000})
    e0 = enrich_chain(ch0, R)
    pos = condor(e0, call=25200.0)
    pos.baseline = {"wall_oi": {"call": 900000.0}, "wall_strike": {"call": 25200.0}}
    ch1 = make_chain(d=D, spikes={(25200.0, "CE"): 500000})
    e1 = enrich_chain(ch1, R)
    r = assess(pos, ch1, e1, uniform_forecast(half=300, d=D), [], D, tmp_cfg)
    assert any(t.startswith("wall shift") for t in r.triggers)


def test_naked_tighter_thresholds(tmp_cfg):
    m = base_metrics(sides={"put": side(touch=0.12), "call": side()})
    assert level_from_metrics(m, {}, IRON_CONDOR, tmp_cfg)[0] == "GREEN"
    assert level_from_metrics(m, {}, NAKED_STRANGLE, tmp_cfg)[0] == "AMBER"
    m = base_metrics(sides={"put": side(delta=0.26), "call": side()})
    assert level_from_metrics(m, {}, NAKED_STRANGLE, tmp_cfg)[0] == "RED"


def test_mark_and_pnl(env):
    ch, e, cfg = env
    pos = condor(e)
    assert mark(pos, e) == pytest.approx(pos.net_credit)
    assert pnl(pos, e) == pytest.approx(0.0)


def test_danger_levels_butterfly(env):
    ch, e, cfg = env
    legs = [Leg("PE", 23800, "BUY", 1, 50), Leg("PE", 24000, "SELL", 1, 150),
            Leg("CE", 24000, "SELL", 1, 150), Leg("CE", 24200, "BUY", 1, 50)]
    pos = Position(1, IRON_BUTTERFLY, D, legs, 200.0, 0, {})
    assert danger_levels(pos) == {"put": 23800.0, "call": 24200.0}
    assert danger_levels(condor(e)) == {"put": 23000.0, "call": 25000.0}


def test_stress_keys_and_sign(env):
    ch, e, cfg = env
    s = stress(condor(e), ch, e, cfg)
    assert list(s) == ["-5%", "-3%", "-2%", "+2%", "+3%", "+5%", "iv+3"]
    assert s["-5%"] < 0 and s["iv+3"] < 0


def test_pre_entry_event_fails(env):
    ch, e, cfg = env
    c, _ = build_candidate(IRON_CONDOR, e, ch, uniform_forecast(half=600, d=D), cfg, 0.0)
    ok, why = pre_entry_check(c, ch, e, uniform_forecast(half=600, d=D), [], D, cfg)
    assert ok, why
    ev = [Event(date(2026, 11, 19), "FOMC", "fomc")]
    ok, why = pre_entry_check(c, ch, e, uniform_forecast(half=600, d=D), ev, D, cfg)
    assert not ok and any("event" in w for w in why)


def test_pre_entry_touch_fails(env):
    ch, e, cfg = env
    c, _ = build_candidate(IRON_CONDOR, e, ch, uniform_forecast(half=600, d=D), cfg, 0.0)
    wide = uniform_forecast(half=6000, d=D)  # most paths beyond the shorts on day 1
    ok, why = pre_entry_check(c, ch, e, wide, [], D, cfg)
    assert not ok and any(w.startswith("touch_3d") for w in why)
